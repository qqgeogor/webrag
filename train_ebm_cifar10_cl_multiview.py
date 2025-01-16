import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
from tqdm import tqdm

import torch.nn.functional as F
import copy

from karas_sampler import KarrasSampler,get_sigmas_karras

# class EnergyNet(nn.Module):
#     def __init__(self, img_channels=3, hidden_dim=64):
#         super().__init__()
        
#         self.net = nn.Sequential(
#             # Initial conv: [B, 3, 32, 32] -> [B, 64, 16, 16]
#             nn.Conv2d(img_channels, hidden_dim, 4, 2, 1),
#             nn.LeakyReLU(0.2),
            
#             # [B, 64, 16, 16] -> [B, 128, 8, 8]
#             nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1),
#             nn.LeakyReLU(0.2),
            
#             # [B, 128, 8, 8] -> [B, 256, 4, 4]
#             nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, 2, 1),
#             nn.LeakyReLU(0.2),
            
#             # [B, 256, 4, 4] -> [B, 512, 2, 2]
#             nn.Conv2d(hidden_dim * 4, hidden_dim * 8, 4, 2, 1),
#             nn.LeakyReLU(0.2),
            
#             # Final conv to scalar energy: [B, 512, 2, 2] -> [B, 1, 1, 1]
#             nn.Conv2d(hidden_dim * 8, 1, 2, 1, 0)
#         )
    
#     def forward(self, x):
#         logits = self.net(x).squeeze()
#         e = -torch.log(torch.sigmoid(logits))
#         return e

sampler = KarrasSampler()

def R(Z,eps=0.5):
    c = Z.shape[-1]
    b = Z.shape[-2]
    
    Z = F.normalize(Z, p=2, dim=-1)
    cov = Z.T @ Z
    I = torch.eye(cov.size(-1)).to(Z.device)
    alpha = c/(b*eps)
    
    cov = alpha * cov +  I

    out = 0.5*torch.logdet(cov)
    return out.mean()


def mcr(Z1,Z2):
    return R(torch.cat([Z1,Z2],dim=0))-0.5*R(Z1)-0.5*R(Z2)


# def dino_loss(Z1,Z2,scale_Z1=1e-2):
#     return -R(Z1).mean()*scale_Z1 + (1 - F.cosine_similarity(Z1,Z2,dim=-1)).mean()


def tcr_loss(Z1,Z2):
    return R(Z1).mean() - R(Z2).mean()


class SimSiamModel(nn.Module):
    def __init__(self, img_channels=3, hidden_dim=64, proj_dim=128, pred_dim=128):
        super().__init__()
        
        # Encoder network
        self.encoder = nn.Sequential(
            # Initial conv: [B, 3, 32, 32] -> [B, 64, 16, 16]
            nn.Conv2d(img_channels, hidden_dim, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            
            # [B, 64, 16, 16] -> [B, 128, 8, 8]
            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            
            # [B, 128, 8, 8] -> [B, 256, 4, 4]
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True),
            
            # [B, 256, 4, 4] -> [B, 512, 2, 2]
            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.ReLU(inplace=True),
            
            # Final conv: [B, 512, 2, 2] -> [B, 512, 1, 1]
            nn.Conv2d(hidden_dim * 8, proj_dim, 2, 1, 0)
        )
        
        # Projector network
        self.projector = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
            nn.BatchNorm1d(proj_dim)
        )
        
    
    def forward(self, x1):
        # Get representations
        z1 = self.encoder(x1).squeeze()
        
        # Get projections
        p1 = self.projector(z1)
        
        return p1
    
    def get_features(self, x):
        """Get encoder features for a single image"""
        z = self.encoder(x).squeeze()
        p = self.projector(z)
        return z, p
    


def mcr_nv_loss(ps):
    c = ps.shape[-1]
    b = ps.shape[-2]
    n_views = len(ps)
    
    joint_p = ps.reshape(-1,c)
    expd = R(joint_p)

    comp = 0
    for i in range(b):
        comp += R(ps[:,i,:])/b
    
    return expd.mean() - comp.mean()
    


def R_nonorm(Z,eps=0.5):
    c = Z.shape[-1]
    b = Z.shape[-2]
    
    cov = Z.T @ Z
    I = torch.eye(cov.size(-1)).to(Z.device)
    alpha = c/(b*eps)
    
    cov = alpha * cov +  I

    out = 0.5*torch.logdet(cov)
    return out.mean()


# Add SimSiam loss function
def simsiam_loss(p1, p2, h1, h2):

    loss_tcr = -R(p1).mean()
    loss_tcr *=1e-2

    # Negative cosine similarity
    loss_cos = (F.cosine_similarity(h1, p2.detach(), dim=-1).mean() + 
             F.cosine_similarity(h2, p1.detach(), dim=-1).mean()) * 0.5
    
    loss_cos = 1-loss_cos

    return loss_cos,loss_tcr


# Modify the training function
def train_ebm(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data preprocessing with two augmentations
    transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-10
    trainset = torchvision.datasets.CIFAR10(root=args.data_path, train=True,
                                          download=True, transform=MultiViewTransform(transform))
    trainloader = DataLoader(trainset, batch_size=args.batch_size,
                           shuffle=True, num_workers=args.num_workers)

    # Initialize model
    model = SimSiamModel(img_channels=3, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for i, (images, _) in enumerate(tqdm(trainloader)):
            # img1, img2 = images[0].to(device), images[1].to(device)  # Unpack the two views
            views = [img.to(device) for img in images]

            Ps = []
            for view in views:
                p = model(view)
                Ps.append(p)
            
            p = torch.stack(Ps,dim=0)
            p = F.normalize(p,dim=-1)

            loss_tcr = -R_nonorm(p.mean(dim=0))
            loss_cos = (1 - F.cosine_similarity(p[0], p[-1], dim=-1).mean())

            # Compute loss
            loss = loss_tcr
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            if i % args.log_freq == 0:
                print(f'Epoch [{epoch}/{args.epochs}], Step [{i}/{len(trainloader)}], '
                      f'Loss: {loss.item():.4f},  Loss_tcr: {loss_tcr.item():.4f}, Loss_cos: {loss_cos.item():.4f}')

        # Add visualization of augmentations periodically
        if epoch % args.save_freq == 0:
            
            # Save model checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss / len(trainloader),
            }, os.path.join(args.output_dir, f'simsiam_checkpoint_{epoch}.pth'))

# Add MultiViewTransform class
class MultiViewTransform:
    
    def __init__(self, base_transform,n_views=20):
        self.n_views = n_views
        self.base_transform = base_transform

    def __call__(self, x):
        views = []
        for _ in range(self.n_views):
            views.append(self.base_transform(x))
        return views

def get_args_parser():
    import argparse
    parser = argparse.ArgumentParser('EBM training for CIFAR-10')
    
    # Training parameters
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    
    # Langevin dynamics parameters
    parser.add_argument('--langevin_steps', default=60, type=int)
    parser.add_argument('--step_size', default=10.0, type=float)
    parser.add_argument('--noise_scale', default=0.005, type=float)
    
    # System parameters
    parser.add_argument('--data_path', default='c:/dataset', type=str)
    parser.add_argument('--output_dir', default='F:/output/cifar10-ebm-cl-multiview')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--save_freq', default=1, type=int)
    parser.add_argument('--resume', default=None, type=str)
    
    return parser

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_ebm(args) 