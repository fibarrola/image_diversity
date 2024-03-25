import imghdr
import os
from torch.utils.data import Dataset
from PIL import Image

VALID_IMG_FORMATS = ["jpeg", "png", "gif", "bmp", "tiff"]


def get_img_names(path):
    filenames = os.listdir(path)
    img_names = [
        fname
        for fname in filenames
        if imghdr.what(os.path.join(path, fname)) in VALID_IMG_FORMATS
    ]
    return img_names


class ImgDataset(Dataset):
    def __init__(self, img_names, img_dir, transforms=None):
        self.img_dir = img_dir
        self.img_names = img_names
        self.transforms = transforms

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, i):
        img_name = self.img_names[i]
        img = Image.open(os.path.join(self.img_dir, img_name))
        if self.transforms is not None:
            img = self.transforms(img)
        return img
