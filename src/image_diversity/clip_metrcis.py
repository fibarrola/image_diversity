import clip
import torch
import os
from PIL import Image
from scipy import linalg

class DivClip:
    def __init__(self, device=None, n_eigs=15):
        self.mock = None

    def compute_div(self, img_dir, img_names=None):
        return "well look at that"