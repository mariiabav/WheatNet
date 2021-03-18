import torch.utils.data as data

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dict, class_to_idx):
    images = []
    idx_to_class = {}
    intervals = []
    tmp0, tmp1 = 0, 0

    for PCTHEAD in sorted(dict):
        for file in dict[PCTHEAD]:
            if is_image_file(file):
                idx_to_class[tmp1] = class_to_idx[PCTHEAD]
                images.append((file, class_to_idx[PCTHEAD]))
                tmp1 += 1
        if tmp0 != tmp1:
            intervals.append((tmp0, tmp1))
        tmp0 = tmp1

    return images, intervals, idx_to_class


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class LabeledImageReader(data.Dataset):

    def __init__(self, data_dict, transform=None, target_transform=None,
                 loader=default_loader):

        classes = [c for c in sorted(data_dict)]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        images, intervals, idx_to_class = make_dataset(data_dict, class_to_idx)

        if len(images) == 0:
            raise (RuntimeError("No images found"))

        self.images = images
        self.classes = classes
        self.class_to_idx = class_to_idx  # cat->1
        self.intervals = intervals
        self.idx_to_class = idx_to_class  # i(img idx)->2(class)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.images[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.images)
