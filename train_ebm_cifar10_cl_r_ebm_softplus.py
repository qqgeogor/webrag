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

def R_nonorm(Z,eps=0.5):
    c = Z.shape[-1]
    b = Z.shape[-2]
    
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
        )
        

        self.head = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, 1)
        )
        
    
    def forward(self, x1, x2):
        # Get representations
        z1 = self.encoder(x1).squeeze()
        z2 = self.encoder(x2).squeeze()
        
        # Get projections
        p1 = self.projector(z1)
        p2 = self.projector(z2)
        
        return p1, p2, p1, p2
    
    def get_features(self, x):
        """Get encoder features for a single image"""
        z = self.encoder(x).squeeze()
        p = self.projector(z)
        h = self.predictor(p)
        return z, p, h
    
    def get_augmented_views(self, x):
        """Get augmented views and their features for visualization"""
        # Create two augmented views
        transform = TwoCropsTransform(self.transform)
        views = transform(x)
        view1, view2 = views[0].unsqueeze(0), views[1].unsqueeze(0)
        
        # Get features for both views
        with torch.no_grad():
            z1, p1, h1 = self.get_features(view1)
            z2, p2, h2 = self.get_features(view2)
        
        return {
            'views': (view1, view2),
            'features': (z1, z2),
            'projections': (p1, p2),
            'predictions': (h1, h2)
        }



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
    p1 = F.normalize(p1, p=2, dim=-1)
    loss_tcr = -R_nonorm(p1).mean()
    loss_tcr *=1e-2

    # Negative cosine similarity
    loss_cos = (F.cosine_similarity(h1, p2.detach(), dim=-1).mean() + 
             F.cosine_similarity(h2, p1.detach(), dim=-1).mean()) * 0.5
    

    
    return loss_cos,loss_tcr

# # Add SimSiam loss function
# def simsiam_loss(p1, p2, h1, h2):
#     p1 = F.normalize(p1, p=2, dim=-1)
#     p2 = F.normalize(p2, p=2, dim=-1)
#     loss_tcr = -R_nonorm(p1+p2).mean()
#     loss_tcr *=1e-2

#     # Negative cosine similarity
#     loss_cos = (F.cosine_similarity(h1, p2.detach(), dim=-1).mean() + 
#              F.cosine_similarity(h2, p1.detach(), dim=-1).mean()) * 0.5
    
#     loss_cos = 1-loss_cos

#     return loss_cos,loss_tcr

def hyperspherical_energy_loss(z1, z2, kernel='inverse', t=2):
    """
    Compute hyperspherical energy loss between two sets of features
    Args:
        z1: first set of features [B, D]
        z2: second set of features [B, D]
        kernel: type of kernel ('inverse' or 'gaussian')
        t: temperature parameter for gaussian kernel
    Returns:
        energy: hyperspherical energy loss
    """
    # Normalize features to unit hypersphere
    z1 = F.normalize(z1, p=2, dim=-1)
    z2 = F.normalize(z2, p=2, dim=-1)
    
    # Compute cross-set distances
    dot_product = torch.matmul(z1, z2.t())
    # Clip dot product to avoid numerical instability
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    
    # Compute angular distances
    angular_dist = torch.acos(dot_product)
    
    if kernel == 'inverse':
        # Avoid division by zero by adding small epsilon
        kernel_matrix = 1.0 / (angular_dist + 1e-8)
    elif kernel == 'gaussian':
        kernel_matrix = torch.exp(-angular_dist / t)
    else:
        raise ValueError(f"Unknown kernel type: {kernel}")
    
    # Compute energy (mean over all pairs)
    energy = kernel_matrix.mean()
    
    return energy

def mahalanobis_distance(z1, z2, eps=1e-6):
    """
    Compute Mahalanobis distance between two sets of vectors
    Args:
        z1: tensor of shape [b, c]
        z2: tensor of shape [b, c]
        eps: small constant for numerical stability
    Returns:
        distances: tensor of shape [b]
    """
    # Compute mean of combined features
    z_combined = torch.cat([z1, z2], dim=0)
    mean = torch.mean(z_combined, dim=0, keepdim=True)
    
    # Center the data
    z1_centered = z1 - mean
    z2_centered = z2 - mean
    
    # Compute covariance matrix
    z_centered = torch.cat([z1_centered, z2_centered], dim=0)
    cov = torch.mm(z_centered.t(), z_centered) / (z_centered.shape[0] - 1)
    
    # Add small diagonal term for numerical stability
    cov = cov + torch.eye(cov.shape[0], device=cov.device) * eps
    
    # Compute inverse of covariance matrix
    inv_cov = torch.linalg.inv(cov)
    
    
    # Compute Mahalanobis distance
    diff = z1 - z2
    distances = torch.sqrt(torch.sum(torch.mm(diff, inv_cov) * diff, dim=1))
    
    return distances


# Alternative energy function implementations
def energy_function(real_energy, fake_energy, version='mse'):
    if version == 'mse':
        # Mean squared error (current implementation)
        return ((real_energy - fake_energy)**2).sum(-1).mean()
    
    elif version == 'l1':
        # L1 distance (absolute difference)
        return (real_energy - fake_energy).abs().sum(-1).mean()
    
    elif version == 'exp':
        # Exponential form (similar to Boltzmann distribution)
        return torch.exp(-(real_energy - fake_energy)**2).sum(-1).mean()
    
    elif version == 'logsigmoid':
        # Log-based energy
        diff = (real_energy - fake_energy)**2
        return F.logsigmoid(diff).sum(-1).mean()

    elif version == 'softplus':
        # Log-based energy
        diff = (real_energy - fake_energy)**2
        return -F.softplus(-diff).sum(-1).mean()
    
    elif version == 'cosine':
        return 1-F.cosine_similarity(real_energy.detach(),fake_energy).mean()
    
def ebm(real_energy,fake_energy,teacher_temp=0.04,student_temp=0.1):
    real_energy = F.normalize(real_energy, p=2, dim=-1)
    fake_energy = F.normalize(fake_energy, p=2, dim=-1)
    # real_energy = F.softmax(real_energy/teacher_temp,dim=-1)
    # fake_energy = fake_energy/student_temp

    loss_tcr = -R(real_energy)*1e-2
    # d_loss = 1 - F.cosine_similarity(real_energy.detach(),fake_energy).mean()
    # loss_cos = (F.cosine_similarity(real_energy.detach(),fake_energy, dim=-1).mean() + 
    #          F.cosine_similarity(fake_energy.detach(),real_energy, dim=-1).mean()) * 0.5
    # loss_cos = 1-loss_cos
    # d_loss = loss_cos
    
    # d_loss = mahalanobis_distance(real_energy.detach(),fake_energy.detach()).mean()
    # d_loss = hyperspherical_energy_loss(real_energy.detach(),fake_energy.detach())
    realistic_logits = real_energy.detach() - fake_energy
    # realistic_logits = (realistic_logits**2).sum(-1).mean()
    d_loss = energy_function(real_energy.detach() ,fake_energy,version='softplus')
    
    # realistic_logits = realistic_logits.abs()
    
    # realistic_logits = F.mse_loss(real_energy.detach(),fake_energy).sum(-1)
    # # # sigma = 0.01
    # # realistic_logits = torch.exp(-((real_energy.detach()-fake_energy)**2).sum(-1) / (2 * sigma**2))
    
    # d_loss = F.softplus(realistic_logits)

    # d_loss = F.mse_loss(real_energy.sum(-1).detach(),fake_energy.sum(-1),reduction='none').mean()
    # d_loss = realistic_logits.mean()
    # d_loss = (real_energy.detach() - fake_energy).abs()
    # d_loss = d_loss.mean(-1).mean()
    # teacher_temp = 0.04
    # student_temp = 0.01
    # d_loss = torch.sum(-real_energy.detach() * F.log_softmax(fake_energy, dim=-1), dim=-1)
    d_loss = d_loss.mean()
    
    # d_loss = (F.mse_loss(real_energy, fake_energy.detach(), reduction='none').sum(dim=-1).mean() + 
    #         F.mse_loss(fake_energy, real_energy.detach(), reduction='none').sum(dim=-1).mean()) * 0.5
    
    

    return loss_tcr,d_loss

# Add a function to visualize augmented views
def visualize_augmentations(model, image, save_path=None):
    """
    Visualize augmented views and their similarity
    Args:
        model: trained SimSiam model
        image: original image tensor [C, H, W]
        save_path: path to save visualization
    """
    model.eval()
    results = model.get_augmented_views(image)
    view1, view2 = results['views']
    
    # Denormalize images for visualization
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    view1_show = view1 * std + mean
    view2_show = view2 * std + mean
    
    # Calculate similarity between features
    z1, z2 = results['features']
    similarity = F.cosine_similarity(z1, z2).item()
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    ax1.imshow(view1_show.squeeze().permute(1, 2, 0).cpu())
    ax1.set_title('View 1')
    ax1.axis('off')
    
    ax2.imshow(view2_show.squeeze().permute(1, 2, 0).cpu())
    ax2.set_title('View 2')
    ax2.axis('off')
    
    plt.suptitle(f'Feature Similarity: {similarity:.3f}')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

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
                                          download=True, transform=TwoCropsTransform(transform))
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
            img1, img2 = images[0].to(device), images[1].to(device)  # Unpack the two views
            
            # Forward pass
            p1, p2, h1, h2 = model(img1, img2)
            
            # Compute loss
            loss_cos,loss_tcr = simsiam_loss(p1, p2, h1, h2)

            
            loss_tcr1,loss_ebm1 = ebm(p1,p2)
            loss_tcr2,loss_ebm2 = ebm(p2,p1)
            loss_tcr = (loss_tcr1+loss_tcr2)/2
            d_loss = (loss_ebm1+loss_ebm2)/2

            
            loss = loss_tcr+d_loss
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            if i % args.log_freq == 0:
                print(f'Epoch [{epoch}/{args.epochs}], Step [{i}/{len(trainloader)}], '
                      f'Loss: {loss.item():.4f}, Loss_cos: {loss_cos.item():.4f}, Loss_tcr: {loss_tcr.item():.4f}, d_loss: {d_loss.item():.4f}')

        # Add visualization of augmentations periodically
        if epoch % args.save_freq == 0:
            
            # Save model checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss / len(trainloader),
            }, os.path.join(args.output_dir, f'simsiam_checkpoint_{epoch}.pth'))

# Add TwoCropsTransform class
class TwoCropsTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)  # First augmented view
        k = self.base_transform(x)  # Second augmented view
        return [q, k]  # Returns a list containing both views

def get_args_parser():
    import argparse
    parser = argparse.ArgumentParser('EBM training for CIFAR-10')
    
    # Training parameters
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    
    # Langevin dynamics parameters
    parser.add_argument('--langevin_steps', default=60, type=int)
    parser.add_argument('--step_size', default=10.0, type=float)
    parser.add_argument('--noise_scale', default=0.005, type=float)
    
    # System parameters
    parser.add_argument('--data_path', default='c:/dataset', type=str)
    parser.add_argument('--output_dir', default='F:/output/cifar10-ebm-cl-ebm-softplus')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--save_freq', default=10, type=int)
    parser.add_argument('--resume', default=None, type=str)
    
    return parser

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_ebm(args) 