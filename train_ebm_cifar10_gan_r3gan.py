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
import argparse
import torch.nn.functional as F

def zero_centered_gradient_penalty(samples, critics):
    grad, = torch.autograd.grad(outputs=critics.sum(), inputs=samples, create_graph=True)
    return grad.square().sum([1, 2, 3])

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
        #logits = -F.logsigmoid(logits)
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


class TCREnergyNet(nn.Module):
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
            nn.Conv2d(hidden_dim * 4, hidden_dim * 4, 1, 1, 0)
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
        Z = self.energy_head(x).squeeze()
        Z = R(Z)
        return Z


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

# Add a proper reshape layer
class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        
    def forward(self, x):
        return x.view(x.size(0), *self.shape)

# Add Generator class
class Generator(nn.Module):
    def __init__(self, latent_dim=100, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            # Initial projection
            nn.Linear(latent_dim, hidden_dim * 8 * 4 * 4),
            nn.LeakyReLU(0.2),
            
            # Reshape layer instead of lambda
            Reshape((hidden_dim * 8, 4, 4)),
            
            # [4x4] -> [8x8]
            nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2),
            
            # [8x8] -> [16x16]
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            
            # [16x16] -> [32x32]
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2),
            
            # Final layer
            nn.ConvTranspose2d(hidden_dim, 3, 3, 1, 1),
            nn.Tanh()
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, z):
        return self.net(z)

# Modify training function
def train_ebm_gan(args):
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
    
    # Filter the dataset to only include class 1
    class_1_indices = [i for i, label in enumerate(trainset.targets) if label == 1]
    trainset.data = trainset.data[class_1_indices]
    trainset.targets = [trainset.targets[i] for i in class_1_indices]
    

    trainloader = DataLoader(trainset, batch_size=args.batch_size,
                           shuffle=True, num_workers=args.num_workers)

    # Initialize models
    generator = Generator(latent_dim=args.latent_dim, hidden_dim=64).to(device)
    # discriminator = ResNetEnergyNet(img_channels=3, hidden_dim=64).to(device)
    discriminator = EnergyNet(img_channels=3, hidden_dim=64).to(device)
    
    # Optimizers
    g_optimizer = torch.optim.Adam(
        generator.parameters(), 
        lr=args.g_lr, 
        betas=(args.g_beta1, args.g_beta2)
    )
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(), 
        lr=args.d_lr, 
        betas=(0.5, 0.999)
    )
    
    # Add Cosine Annealing schedulers
    g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        g_optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr
    )
    d_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        d_optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr
    )
    
    # Add checkpoint loading logic
    if args.resume:
        checkpoint_path = args.resume
        if os.path.isfile(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            generator.load_state_dict(checkpoint['generator_state_dict'])
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
            d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
            g_scheduler.load_state_dict(checkpoint['g_scheduler_state_dict'])
            d_scheduler.load_state_dict(checkpoint['d_scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")
    
    # Training loop
    for epoch in range(args.epochs):
        generator.train()
        discriminator.train()
        
        for i, (real_samples, _) in enumerate(tqdm(trainloader)):
            batch_size = real_samples.size(0)
            real_samples = real_samples.to(device)
            
            # Train Discriminator
            for _ in range(args.n_critic):  # Train discriminator more frequently
                d_optimizer.zero_grad()
                
                # Generate fake samples
                z = torch.randn(batch_size, args.latent_dim, device=device)
                real_samples = real_samples.detach().requires_grad_(True)
                fake_samples = generator(z).detach().requires_grad_(True)
                
                # Compute energies
                real_energy = discriminator(real_samples)
                fake_energy = discriminator(fake_samples)
                
                realistic_logits = real_energy - fake_energy
                d_loss = F.softplus(-realistic_logits)
                # Improved EBM-GAN discriminator loss
                # d_loss = (F.softplus(real_energy) + (-fake_energy))
                
                r1 = zero_centered_gradient_penalty(real_samples, real_energy)
                r2 = zero_centered_gradient_penalty(fake_samples, fake_energy)

                d_loss = d_loss + args.gp_weight/2 * (r1 + r2)
                d_loss = d_loss.mean()

                # # Add gradient penalty
                # gp = compute_gradient_penalty(discriminator, real_samples, fake_samples, device)
                # d_loss = d_loss + args.gp_weight * gp
                
                d_loss.backward()
                d_optimizer.step()
            
            # Train Generator
            g_optimizer.zero_grad()
            
            # Generate new fake samples
            z = torch.randn(batch_size, args.latent_dim, device=device)
            fake_samples = generator(z)
            fake_energy = discriminator(fake_samples)
            real_energy = discriminator(real_samples)

            realistic_logits = fake_energy - real_energy
            g_loss = F.softplus(-realistic_logits)
            g_loss = g_loss.mean()
            
            # Improved generator loss
            # g_loss = (fake_energy).mean()
            
            g_loss.backward()
            g_optimizer.step()
            
            if i % args.log_freq == 0:
                current_g_lr = g_optimizer.param_groups[0]['lr']
                current_d_lr = d_optimizer.param_groups[0]['lr']
                print(f'Epoch [{epoch}/{args.epochs}], Step [{i}/{len(trainloader)}], '
                      f'D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}, '
                      f'Real Energy: {real_energy.mean().item():.4f}, '
                      f'Fake Energy: {fake_energy.mean().item():.4f}, '
                      f'G_LR: {current_g_lr:.6f}, D_LR: {current_d_lr:.6f}'
                      )
        
        # Step the schedulers at the end of each epoch
        g_scheduler.step()
        d_scheduler.step()
        
        # Save samples and model checkpoints
        if epoch % args.save_freq == 0:
            save_gan_samples(generator, discriminator, epoch, args.output_dir, device)
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
                'g_scheduler_state_dict': g_scheduler.state_dict(),
                'd_scheduler_state_dict': d_scheduler.state_dict(),
            }, os.path.join(args.output_dir, f'ebm_gan_checkpoint_{epoch}.pth'))

def save_gan_samples(generator, discriminator, epoch, output_dir, device, n_samples=36):
    generator.eval()
    discriminator.eval()
    
    with torch.no_grad():
        z = torch.randn(n_samples, args.latent_dim, device=device)
        fake_samples = generator(z)
        
        # Changed 'range' to 'value_range'
        grid = make_grid(fake_samples, nrow=6, normalize=True, value_range=(-1, 1))
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.cpu().permute(1, 2, 0))
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'gan_samples_epoch_{epoch}.png'))
        plt.close()

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

def get_args_parser():
    parser = argparse.ArgumentParser('EBM-GAN training for CIFAR-10')
    
    # Add GAN-specific parameters
    parser.add_argument('--latent_dim', default=100, type=int)
    parser.add_argument('--g_lr', default=1e-4, type=float)
    parser.add_argument('--d_lr', default=3e-4, type=float)
    parser.add_argument('--n_critic', default=5, type=int,
                        help='Number of discriminator updates per generator update')
    parser.add_argument('--gp_weight', default=0.05, type=float,
                        help='Weight of gradient penalty')
    
    # Modify learning rates
    parser.add_argument('--g_beta1', default=0.0, type=float,
                        help='Beta1 for generator optimizer')
    parser.add_argument('--g_beta2', default=0.9, type=float,
                        help='Beta2 for generator optimizer')
    
    # Existing parameters
    parser.add_argument('--epochs', default=1200, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--data_path', default='/home/qianqian/repo/cnn_cl/data', type=str)
    parser.add_argument('--output_dir', default='./output/cifar10-ebm-gan-r3gan')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--save_freq', default=1, type=int)
    
    # Add learning rate scheduling parameters
    parser.add_argument('--min_lr', default=1e-6, type=float,
                        help='Minimum learning rate for cosine annealing')
    
    # Add checkpoint loading parameter
    parser.add_argument('--resume', default=None, type=str,
                        help='Path to checkpoint to resume training from')
    
    return parser

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_ebm_gan(args) 