import imghdr
import os

VALID_IMG_FORMATS = ["jpeg", "png", "gif", "bmp", "tiff"]


def get_img_names(path):
    filenames = os.listdir(path)
    img_names = [
        fname
        for fname in filenames
        if imghdr.what(os.path.join(path, fname)) in VALID_IMG_FORMATS
    ]
    return img_names
