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

from timm.scheduler.cosine_lr import CosineLRScheduler
import torch.nn.functional as F

class EnergyNet(nn.Module):
    def __init__(self, img_channels=3, hidden_dim=64):
        super().__init__()
        
        self.net = nn.Sequential(
            # Initial conv: [B, 3, 32, 32] -> [B, 64, 16, 16]
            nn.Conv2d(img_channels, hidden_dim, 4, 2, 1),
            # nn.GroupNorm(8, hidden_dim),  # Add normalization
            nn.LeakyReLU(0.2),
            
            # [B, 64, 16, 16] -> [B, 128, 8, 8]
            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1),
            # nn.GroupNorm(8, hidden_dim * 2),  # Add normalization
            nn.LeakyReLU(0.2),
            
            # [B, 128, 8, 8] -> [B, 256, 4, 4]
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, 2, 1),
            # nn.GroupNorm(8, hidden_dim * 4),  # Add normalization
            nn.LeakyReLU(0.2),
            
            # [B, 256, 4, 4] -> [B, 512, 2, 2]
            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, 4, 2, 1),
            # nn.GroupNorm(8, hidden_dim * 8),  # Add normalization
            nn.LeakyReLU(0.2),
            
            # Final conv to scalar energy: [B, 512, 2, 2] -> [B, 1, 1, 1]
            nn.Conv2d(hidden_dim * 8, 1, 2, 1, 0)
        )
        
        # Initialize weights properly
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
    
    def forward(self, x):
        logits = self.net(x).squeeze()
        # print(x.shape)
        # logits = self.head(logits)
        # Add regularization term to prevent collapse
        # reg_term = 0.1 * (logits ** 2).mean()
        # logits = logits# + reg_term
        # logits = -F.logsigmoid(logits)
        return logits


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.gn1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.gn2 = nn.GroupNorm(8, out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.GroupNorm(8, out_channels)
            )
    
    def forward(self, x):
        out = F.leaky_relu(self.gn1(self.conv1(x)), 0.2)
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.leaky_relu(out, 0.2)
        return out

class ResNetEnergyNet(nn.Module):
    def __init__(self, img_channels=3, hidden_dim=64):
        super().__init__()
        
        # Initial conv layer
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, hidden_dim, 3, 1, 1),
            nn.GroupNorm(8, hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # ResNet blocks with downsampling
        self.layer1 = ResBlock(hidden_dim, hidden_dim * 2, stride=2)
        self.layer2 = ResBlock(hidden_dim * 2, hidden_dim * 4, stride=2)
        self.layer3 = ResBlock(hidden_dim * 4, hidden_dim * 8, stride=2)
        self.layer4 = ResBlock(hidden_dim * 8, hidden_dim * 8, stride=2)
        
        # Final energy output
        self.energy_head = nn.Sequential(
            nn.Conv2d(hidden_dim * 8, hidden_dim * 4, 2, 1, 0),
            nn.GroupNorm(8, hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dim * 4, 1, 1, 1, 0)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
    
    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        logits = self.energy_head(x).squeeze()
        energy = -F.logsigmoid(logits)
        return energy
    
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



def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    """Compute gradient penalty for improved training stability"""
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    
    d_interpolates = discriminator(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty



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
    # model = ResNetEnergyNet(img_channels=3, hidden_dim=64).to(device)
    model = EnergyNet(img_channels=3, hidden_dim=64).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Use timm's CosineLRScheduler
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=args.epochs,  # Total number of epochs
        lr_min=args.min_lr,
        warmup_t=args.warmup_epochs,  # Warmup epochs
        warmup_lr_init=1e-6,  # Initial warmup learning rate
        cycle_limit=args.num_cycles,  # Number of cycles
        t_in_epochs=True,  # Use epochs for scheduling
        warmup_prefix=True,  # Don't count warmup in cycle
    )

    scaler = GradScaler() 
    
    # Add checkpoint loading logic
    start_epoch = 0
    if args.resume:
        checkpoint_path = args.resume
        if os.path.isfile(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at {checkpoint_path}")

    sampler = LangevinSampler(
        n_steps=args.langevin_steps,
        step_size=args.step_size,
        noise_scale=args.noise_scale
    )

    # Modify training loop to start from loaded epoch
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        
        for i, (real_samples, _) in enumerate(tqdm(trainloader)):
            real_samples = real_samples.to(device)
            batch_size = real_samples.size(0)
            
            # Generate negative samples
            init_samples = torch.randn_like(real_samples)

            with autocast():
                neg_samples = sampler.sample(model, init_samples)
                
                # Compute energies
                pos_energy = model(real_samples)
                neg_energy = model(neg_samples.detach())
                # loss_energy = model(neg_samples)*0.1
                
                loss = ( pos_energy.mean() - neg_energy.mean())
                
                
                # # Contrastive divergence loss
                # loss = ( pos_energy.mean().detach() - neg_energy.mean())
                # loss += ( pos_energy.mean() - neg_energy.mean().detach())
                if args.gp_weight > 0:  
                    gp = compute_gradient_penalty(model, real_samples, neg_samples, device)
                    # Contrastive divergence loss
                    # loss += loss_energy.mean()
                    loss += gp*args.gp_weight
                # loss = ( pos_energy.mean() - neg_energy.mean())
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
            
            optimizer.zero_grad()
            total_loss += loss.item()
            
            if i % args.log_freq == 0:
                print(f'Epoch [{epoch}/{args.epochs}], Step [{i}/{len(trainloader)}], '
                      f'Loss: {loss.item():.4f}, '
                      f'Pos Energy: {pos_energy.mean().item():.4f}, '
                      f'Neg Energy: {neg_energy.mean().item():.4f}')
        scheduler.step(epoch)
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
    parser.add_argument('--min_lr', default=1e-5, type=float)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--num_cycles', default=1, type=int)

    # Langevin dynamics parameters
    parser.add_argument('--langevin_steps', default=60, type=int)
    parser.add_argument('--step_size', default=10.0, type=float)
    parser.add_argument('--noise_scale', default=0.005, type=float)
    parser.add_argument('--gp_weight', default=10.0, type=float)
    
    # System parameters
    parser.add_argument('--data_path', default='c:/dataset', type=str)
    parser.add_argument('--output_dir', default='F:/output/cifar10-ebm')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--save_freq', default=1, type=int)
    parser.add_argument('--resume', default='', type=str,
                       help='path to latest checkpoint (default: none)')
    
    return parser

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_ebm(args) 