import torch
import os
from PIL import Image
from torchvision import transforms
from scipy import linalg
from src.inception import InceptionV3


class DivInception:
    def __init__(self, out_dim=2048, device=None, n_eigs=15):
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            print(f"Device set as: {self.device}")

        self.preprocess = transforms.Compose(
            [
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
            ]
        )
        self.out_dim = out_dim
        net_idx = InceptionV3.BLOCK_INDEX_BY_DIM[out_dim]
        self.inception_model = InceptionV3([net_idx]).to(self.device)
        self.inception_model.eval()
        self.n_eigs = n_eigs

    @torch.no_grad()
    def encode(self, img_names, img_dir):
        # BATCH THIS
        zz = torch.empty((len(img_names), self.out_dim), device=self.device)
        for i, img_name in enumerate(img_names):
            image = (
                self.preprocess(Image.open(f"{img_dir}/{img_name}"))
                .unsqueeze(0)
                .to(self.device)
            )
            zz[i, :] = self.inception_model(image)[0].squeeze(3).squeeze(2)

        return zz

    @torch.no_grad()
    def tie(self, img_dir, img_names=None):
        if img_names is None:
            img_names = os.listdir(img_dir)
        assert self.n_eigs < len(
            img_names
        ), "The number of eigenvalues for truncation must be smaller than the number of samples"
        zz = self.encode(img_names, img_dir)
        sigma = torch.cov(torch.t(zz))
        eigvals = torch.linalg.eigvals(sigma)[: self.n_eigs]
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
    def compute_dist(self, img_dir1, img_dir2, img_names1=None, img_names2=None):
        if img_names1 is None:
            img_names1 = os.listdir(img_dir1)
        if img_names2 is None:
            img_names2 = os.listdir(img_dir2)

        zz1 = self.encode(img_names1, img_dir1)
        zz2 = self.encode(img_names2, img_dir2)

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
