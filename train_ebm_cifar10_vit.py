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
from train_mae_cifar10_edm import MaskedAutoencoderViT  # Import the MAE model

class EnergyNet(nn.Module):
    def __init__(self, img_channels=3, hidden_dim=64):
        super().__init__()
        
        self.model = MaskedAutoencoderViT(
            img_size=32,
            patch_size=4,
            in_chans=3,
            embed_dim=192,
            depth=4,
            num_heads=3,
            decoder_embed_dim=96,
            decoder_depth=4,
            decoder_num_heads=3,
            mlp_ratio=4.,
        )
        self.head = nn.Linear(192,1)    
    def forward(self, x):
        x = self.model.forward_feature(x)
        x = x[:,0]
        x = self.head(x)
        e = -F.logsigmoid(x)
        return e


class LangevinSampler:
    def __init__(self, n_steps=60, step_size=10.0, noise_scale=0.005):
        self.n_steps = n_steps
        self.step_size = step_size
        self.noise_scale = noise_scale
    
    def sample(self, model, x_init, return_trajectory=False):
        model.eval()
        # Ensure x requires gradients
        x = x_init.clone().detach().requires_grad_(True)
        trajectory = [x.clone().detach()] if return_trajectory else None
        
        for _ in range(self.n_steps):
            # Ensure x requires gradients at each step
            if not x.requires_grad:
                x.requires_grad_(True)
                
            # Compute energy gradient
            energy = model(x)
            if isinstance(energy, torch.Tensor):
                energy = energy.sum()
            
            # Compute gradients
            if x.grad is not None:
                x.grad.zero_()
            grad = torch.autograd.grad(energy, x, create_graph=False, retain_graph=True)[0]
            
            # Langevin dynamics update
            noise = torch.randn_like(x) * self.noise_scale
            x = x.detach()  # Detach from computation graph
            x = x - self.step_size * grad + noise  # Update x
            x.requires_grad_(True)  # Re-enable gradients
            x = torch.clamp(x, -1, 1)  # Keep samples in valid range
            
            if return_trajectory:
                trajectory.append(x.clone().detach())
        
        return (x.detach(), trajectory) if return_trajectory else x.detach()

def train_ebm(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data preprocessing
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-10
    trainset = torchvision.datasets.CIFAR10(root=args.data_path, train=True,
                                          download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=args.batch_size,
                           shuffle=True, num_workers=args.num_workers)

    # Initialize model and sampler
    model = EnergyNet(img_channels=3, hidden_dim=64).to(device)
    sampler = LangevinSampler(
        n_steps=args.langevin_steps,
        step_size=args.step_size,
        noise_scale=args.noise_scale
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler() if args.use_amp else None

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for i, (real_samples, _) in enumerate(tqdm(trainloader)):
            real_samples = real_samples.to(device)
            batch_size = real_samples.size(0)
            
            # Generate negative samples
            init_samples = torch.randn_like(real_samples)
            if args.use_amp:
                with autocast():
                    neg_samples = sampler.sample(model, init_samples)
                    
                    # Compute energies
                    pos_energy = model(real_samples)
                    neg_energy = model(neg_samples.detach())
                    
                    # Contrastive divergence loss
                    loss = pos_energy.mean() - neg_energy.mean()
                    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                neg_samples = sampler.sample(model, init_samples)
                
                # Compute energies
                pos_energy = model(real_samples)
                neg_energy = model(neg_samples.detach())
                
                # Contrastive divergence loss
                loss = pos_energy.mean() - neg_energy.mean()
                
                loss.backward()
                optimizer.step()
            
            optimizer.zero_grad()
            total_loss += loss.item()

            if i % args.log_freq == 0:
                print(f'Epoch [{epoch}/{args.epochs}], Step [{i}/{len(trainloader)}], '
                      f'Loss: {loss.item():.4f}, '
                      f'Pos Energy: {pos_energy.mean().item():.4f}, '
                      f'Neg Energy: {neg_energy.mean().item():.4f}')

        # Save samples and model checkpoint
        if epoch % args.save_freq == 0:
            save_samples(model, sampler, epoch, args.output_dir, device)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss / len(trainloader),
            }, os.path.join(args.output_dir, f'ebm_checkpoint_{epoch}.pth'))

def save_samples(model, sampler, epoch, output_dir, device, n_samples=36):
    model.eval()
    # Generate samples - remove no_grad context since we need gradients for Langevin dynamics
    init_noise = torch.randn(n_samples, 3, 32, 32, device=device)
    samples, trajectory = sampler.sample(model, init_noise, return_trajectory=True)
    
    # Use no_grad only for visualization
    with torch.no_grad():
        # Save final samples
        grid = make_grid(samples, nrow=6, normalize=True, range=(-1, 1))
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.cpu().permute(1, 2, 0))
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'samples_epoch_{epoch}.png'))
        plt.close()
        
        # Save sampling trajectory
        if epoch % 10 == 0 and trajectory:  # Save trajectory less frequently
            # Make sure we have enough trajectory steps
            step = max(1, len(trajectory) // 8)
            selected_trajectories = trajectory[::step][:8]  # Take up to 8 steps
            
            if selected_trajectories:
                trajectory_samples = torch.cat([traj[0:6] for traj in selected_trajectories])
                trajectory_grid = make_grid(
                    trajectory_samples,
                    nrow=6, normalize=True, range=(-1, 1)
                )
                plt.figure(figsize=(15, 10))
                plt.imshow(trajectory_grid.cpu().permute(1, 2, 0))
                plt.axis('off')
                plt.savefig(os.path.join(output_dir, f'trajectory_epoch_{epoch}.png'))
                plt.close()

def get_args_parser():
    import argparse
    parser = argparse.ArgumentParser('EBM training for CIFAR-10')
    
    # Training parameters
    parser.add_argument('--epochs', default=1200, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    
    # Langevin dynamics parameters
    parser.add_argument('--langevin_steps', default=60, type=int)
    parser.add_argument('--step_size', default=10.0, type=float)
    parser.add_argument('--noise_scale', default=0.005, type=float)
    
    # System parameters
    parser.add_argument('--data_path', default='c:/dataset', type=str)
    parser.add_argument('--output_dir', default='F:/output/cifar10-ebm-vit')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--save_freq', default=1, type=int)
    
    return parser

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_ebm(args) 