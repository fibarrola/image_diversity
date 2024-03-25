from .clip import clip
import torch
import warnings
from scipy import linalg
from .utils import get_img_names, ImgDataset
from torch.utils.data import DataLoader


class ClipMetrics:
    def __init__(self, device=None, n_eigs=20):
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            print(f"Device set as: {self.device}")

        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.n_eigs = n_eigs

    @torch.no_grad()
    def encode(self, img_names, img_dir, batch_size):
        dataset = ImgDataset(img_names=img_names, img_dir=img_dir, transforms=self.preprocess)
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size)
        zz = torch.empty((len(img_names), 512), device=self.device)
        k = 0
        for batch in data_loader:
            batch = batch.to(self.device)
            zz[k:k + batch.shape[0], :] = self.clip_model.encode_image(batch)
            k += batch.shape[0]

        return zz

    @torch.no_grad()
    def tce(self, img_dir, img_names=None, batch_size=16):
        if img_names is None:
            img_names = get_img_names(img_dir)
        assert self.n_eigs < len(
            img_names
        ), "The number of eigenvalues for truncation must be smaller than the number of samples"
        zz = self.encode(img_names, img_dir, batch_size=batch_size)
        sigma = torch.cov(torch.t(zz))
        eigvals, _ = torch.lobpcg(sigma, k=self.n_eigs)
        eigvals = torch.real(eigvals)
        return self.truncated_entropy(eigvals).item()

    def truncated_entropy(self, eigvals):
        output = (
            len(eigvals)
            * torch.log(torch.tensor(2 * torch.pi * torch.e, device=self.device))
            / 2
        )
        output += 0.5 * sum(torch.log(eigvals))
        return output

    @torch.no_grad()
    def fcd(self, img_dir1, img_dir2, img_names1=None, img_names2=None, batch_size=10):
        if img_names1 is None:
            img_names1 = get_img_names(img_dir1)
        if img_names2 is None:
            img_names2 = get_img_names(img_dir2)

        if len(img_names1) != len(img_names2):
            warnings.warn(
                "WARNING: to make a fair comparison, both sets should have the same number of images"
            )

        assert self.n_eigs < len(
            img_names1
        ), "The number of eigenvalues for truncation must be smaller than the number of samples"
        assert self.n_eigs < len(
            img_names2
        ), "The number of eigenvalues for truncation must be smaller than the number of samples"

        zz1 = self.encode(img_names1, img_dir1, batch_size=batch_size)
        zz2 = self.encode(img_names2, img_dir2, batch_size=batch_size)

        mu_diff = torch.mean(zz1, dim=0) - torch.mean(zz2, dim=0)
        sigma1 = torch.cov(torch.t(zz1)) + 1e-6 * torch.eye(
            zz1.shape[1], device=self.device
        )
        sigma2 = torch.cov(torch.t(zz2)) + 1e-6 * torch.eye(
            zz2.shape[1], device=self.device
        )
        approx_sqrt = linalg.sqrtm(torch.matmul(sigma1, sigma2).to("cpu")).real

        dist = torch.matmul(mu_diff, torch.t(mu_diff))
        dist += torch.trace(
            sigma1 + sigma2 - 2 * torch.tensor(approx_sqrt).to(self.device)
        )

        return dist
