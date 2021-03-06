import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet', 'resnetK']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        self.drop_rate = 0.4
        self.drop_out = nn.Dropout(p=self.drop_rate)

    def d_rate(self, r):
        self.drop_rate = r
        self.drop_out = nn.Dropout(p=r)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.drop_out(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[2], stride=2)

        self.avgpool = nn.AvgPool2d(7)

        self.drop_rate = 0
        self.drop_out = nn.Dropout(p=self.drop_rate)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def d_rate(self, r):
        print('drop rate: {}'.format(r))
        self.drop_rate = r
        self.drop_out = nn.Dropout(p=r)

        self.layer1[0].d_rate(r)
        self.layer2[0].d_rate(r)
        self.layer3[0].d_rate(r)
        self.layer4[0].d_rate(r)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.drop_out(x)
        x = self.layer2(x)
        x = self.drop_out(x)
        x = self.layer3(x)
        x = self.drop_out(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


def resnetK(F, pretrained=True, **kwargs):
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    if pretrained:
        pretr_dict = model_zoo.load_url(model_urls['resnet18'])
        model_dict = model.state_dict()
        # filter out unnecessary keys
        pretr_dict = {k: v for k, v in pretr_dict.items() if k in model_dict}
        # overwrite entries in the existing state dict
        model_dict.update(pretr_dict)
        # load the new state dict
        model.load_state_dict(pretr_dict)

    return model


class VNetA(nn.Module):
    def __init__(self, F, C):
        super(VNetA, self).__init__()
        self.vote = nn.AvgPool2d((9, 1))
        self.classifier = nn.Linear(512, C)

    def forward(self, x):
        N = x.size(0)
        x = self.vote(x)  # [N 1 9 F] -> [N 1 1 F]
        x = x.view(N, -1)  # [N 1 1 F]->[N F]
        x = self.classifier(x)
        return x


class KNet(nn.Module):
    def __init__(self, F, C):
        super(KNet, self).__init__()
        self.R = resnetK(F)
        self.V = VNetA(512, C)

    def forward(self, x):
        n_img = x.size(0)
        x = self.sep(x)  # [N C W H] -> [9N C W H]
        x = self.R(x)  # [9N C W H] -> [9N F]

        n_fea = x.size(1)
        x = x.view(9, n_img, n_fea).permute(1, 0, 2).contiguous().unsqueeze(
            1)  # [9N F] -> [9 N F] -> [N 9 F] -> [N 1 9 F]
        x = self.V(x)  # [N 1 9 F] -> [N Class]
        return x

    def sep(self, img):  # input N by C by (w, h)
        # N,C,W,H = img.size()
        IMSIZE = 224
        NGSIZE = 3
        imgs = []
        # crop the center part of the image
        for nh in range(NGSIZE):
            for nw in range(NGSIZE):
                Hsta = nh * IMSIZE
                Wsta = nw * IMSIZE
                Hend = nh * IMSIZE + IMSIZE
                Wend = nw * IMSIZE + IMSIZE
                imgs.append(img[:, :, Hsta:Hend, Wsta:Wend])

        return torch.stack(imgs, 0).view(-1, 3, IMSIZE, IMSIZE)  # [9N C W H] C=3
