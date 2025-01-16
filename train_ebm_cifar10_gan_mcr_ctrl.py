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
from ibot_ctrl import utils_ibot as utils



class EnergyNet(nn.Module):
    def __init__(self, img_channels=3, hidden_dim=64):
        super().__init__()
        
        self.net = nn.Sequential(
            # Initial conv: [B, 3, 32, 32] -> [B, 64, 16, 16]
            nn.Conv2d(img_channels, hidden_dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            
            # [B, 64, 16, 16] -> [B, 128, 8, 8]
            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1),
            nn.LeakyReLU(0.2),
            
            # [B, 128, 8, 8] -> [B, 256, 4, 4]
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, 2, 1),
            nn.LeakyReLU(0.2),
            
            # [B, 256, 4, 4] -> [B, 512, 2, 2]
            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, 4, 2, 1),
            nn.LeakyReLU(0.2),
            
            # Final conv to scalar energy: [B, 512, 2, 2] -> [B, 128, 1, 1]
            nn.Conv2d(hidden_dim * 8, 128, 2, 1, 0)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)   
    
    def forward(self, x):
        Z = self.net(x).squeeze()
        #Z = R(Z)
        return Z

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


def dino_loss(Z1,Z2,scale_Z1=1e-2):
    return -R(Z1).mean()*scale_Z1 + (1 - F.cosine_similarity(Z1,Z2,dim=-1)).mean()

def inv_dino_loss(Z1,Z2,scale_Z1=1e-2):
    return -R(Z1).mean()*scale_Z1 + (F.cosine_similarity(Z1,Z2,dim=-1).abs()).mean()




def mmcr_loss(Z1,Z2,scale_Z1=1e-2):
    return -R(Z1).mean()*scale_Z1 + (1 - F.cosine_similarity(Z1,Z2,dim=-1)).mean()

def inv_dino_loss(Z1,Z2,scale_Z1=1e-2):
    return -R(Z1).mean()*scale_Z1 + (F.cosine_similarity(Z1,Z2,dim=-1).abs()).mean()



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
    

    if args.cls != -1:
        # Filter the dataset to only include class 1
        class_1_indices = [i for i, label in enumerate(trainset.targets) if label == args.cls]
        trainset.data = trainset.data[class_1_indices]
        trainset.targets = [trainset.targets[i] for i in class_1_indices]
    

    trainloader = DataLoader(trainset, batch_size=args.batch_size,
                           shuffle=True, num_workers=args.num_workers)

    # Initialize models
    generator = Generator(latent_dim=args.latent_dim, hidden_dim=64).to(device)
    discriminator = ResNetEnergyNet(img_channels=3, hidden_dim=64).to(device)
    discriminator = EnergyNet(img_channels=3, hidden_dim=64).to(device)
    
    # Optimizers
    g_optimizer = torch.optim.Adam(
        generator.parameters(), 
        lr=args.g_lr, 
        betas=(0.5, 0.999)
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
    

    g_tcr_schedule = utils.cosine_scheduler(
        0.2,
        0.15,
        args.epochs, len(trainloader),
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
        
        
        
        # Initialize generator accumulation
        accumulated_real_energy_d = []
        accumulated_fake_energy_d = []


    
        # Initialize generator accumulation
        accumulated_real_energy_g = []
        accumulated_fake_energy_g = []
    
    
        for i, (real_samples, _) in enumerate(tqdm(trainloader)):
            
            it = len(trainloader) * epoch + i  # global training iteration
            g_tcr_scale = g_tcr_schedule[it]
            
            batch_size = real_samples.size(0)
            real_samples = real_samples.to(device)
            
            
            # Train Discriminator
            for _ in range(args.n_critic):  # Train discriminator more frequently
                d_optimizer.zero_grad()
                
                # Generate fake samples
                # z = torch.randn(batch_size, args.latent_dim, device=device)
                
                z = discriminator(real_samples)

                fake_samples = generator(z).detach()
                
                # Compute energies
                real_energy = discriminator(real_samples)
                fake_energy = discriminator(fake_samples)
                accumulated_real_energy_d.append(real_energy)
                accumulated_fake_energy_d.append(fake_energy)
                
                if len(accumulated_real_energy_d) == args.accumulation_steps:
                    # Concatenate accumulated energies
                    real_energy_cat = torch.cat(accumulated_real_energy_d, dim=0)
                    fake_energy_cat = torch.cat(accumulated_fake_energy_d, dim=0)
                    
                    # Compute MCR loss on accumulated energies
                    # d_loss = -mcr(real_energy_cat, fake_energy_cat)
                    d_loss = inv_dino_loss(real_energy_cat,fake_energy_cat,scale_Z1=1e-2)
                    
                    # Backward and optimize
                    d_loss.backward()
                    d_optimizer.step()
                    
                    # Clear accumulation lists
                    accumulated_real_energy_d = []
                    accumulated_fake_energy_d = []


                # Improved EBM-GAN discriminator loss
                # d_loss = ((real_energy) + (-fake_energy)).mean()

                # d_loss = -mcr(real_energy,fake_energy)
                
                # # # Add gradient penalty
                # # gp = compute_gradient_penalty(discriminator, real_samples, fake_samples, device)
                # # d_loss = d_loss + args.gp_weight * gp
                
                # d_loss.backward()
                # d_optimizer.step()
            
            # Train Generator
            g_optimizer.zero_grad()
            

            # Generate new fake samples
            # z = torch.randn(batch_size, args.latent_dim, device=device)
            
            z = discriminator(real_samples)
            fake_samples = generator(z)
            
            
            real_energy = discriminator(real_samples)
            fake_energy = discriminator(fake_samples)
            

            accumulated_real_energy_g.append(real_energy)
            accumulated_fake_energy_g.append(fake_energy)

            if len(accumulated_real_energy_g) == args.accumulation_steps:
                # Concatenate accumulated energies
                real_energy_cat = torch.cat(accumulated_real_energy_g, dim=0)
                fake_energy_cat = torch.cat(accumulated_fake_energy_g, dim=0)
                
                # g_loss = mcr(real_energy_cat,fake_energy_cat)
                # g_loss += -R(fake_energy_cat)*g_tcr_scale
                
                g_loss = dino_loss(real_energy_cat,fake_energy_cat,scale_Z1=1e-2)
                
                # Backward and optimize
                g_loss.backward()
                g_optimizer.step()
                
                # Clear accumulation lists
                accumulated_real_energy_g = []
                accumulated_fake_energy_g = []

            
            # # Improved generator loss
            # # g_loss = (fake_energy).mean()
            # g_loss = mcr(real_energy,fake_energy)
            # g_loss += -R(fake_energy)*g_tcr_scale

            # # g_loss = dino_loss(real_energy,fake_energy,scale_Z1=0)*200
            
            # g_loss.backward()
            # g_optimizer.step()
            
            try:
                if i % args.log_freq == 0 :
                    current_g_lr = g_optimizer.param_groups[0]['lr']
                    current_d_lr = d_optimizer.param_groups[0]['lr']
                    print(f'Epoch [{epoch}/{args.epochs}], Step [{i}/{len(trainloader)}], '
                        f'D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}, '
                        f'Real Energy: {R(real_energy).item():.4f}, '
                        f'Fake Energy: {R(fake_energy).mean().item():.4f}, '
                        f'G_LR: {current_g_lr:.6f}, D_LR: {current_d_lr:.6f}, G_TCR_SCALE: {g_tcr_scale:.6f}'
                        )
            except:
                pass
        
        # Step the schedulers at the end of each epoch
        g_scheduler.step()
        d_scheduler.step()
        real_samples = next(iter(trainloader))[0].to(device)
        # Save samples and model checkpoints
        if epoch % args.save_freq == 0:
            save_gan_samples(generator, discriminator, epoch, args.output_dir, device, n_samples=36,real_samples=real_samples)
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
                'g_scheduler_state_dict': g_scheduler.state_dict(),
                'd_scheduler_state_dict': d_scheduler.state_dict(),
            }, os.path.join(args.output_dir, f'ebm_gan_checkpoint_{epoch}.pth'))

def save_gan_samples(generator, discriminator, epoch, output_dir, device, n_samples=36,real_samples=None):
    generator.eval()
    discriminator.eval()
    
    with torch.no_grad():
        # z = torch.randn(n_samples, args.latent_dim, device=device)
        
        z = discriminator.net(real_samples[:n_samples]).squeeze()
        fake_samples = generator(z)
        
        # Changed 'range' to 'value_range'
        grid = make_grid(fake_samples, nrow=6, normalize=True, value_range=(-1, 1))
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.cpu().permute(1, 2, 0))
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'gan_samples_epoch_{epoch}.png'))

        grid = make_grid(real_samples[:n_samples], nrow=6, normalize=True, value_range=(-1, 1))
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.cpu().permute(1, 2, 0))
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'gan_samples_epoch_{epoch}_real.png'))
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
    parser.add_argument('--latent_dim', default=128, type=int)
    parser.add_argument('--g_lr', default=1e-4, type=float)
    parser.add_argument('--d_lr', default=1e-4, type=float)
    parser.add_argument('--n_critic', default=2, type=int,
                        help='Number of discriminator updates per generator update')
    parser.add_argument('--gp_weight', default=10.0, type=float,
                        help='Weight of gradient penalty')
    
    # Modify learning rates
    parser.add_argument('--g_beta1', default=0.5, type=float,
                        help='Beta1 for generator optimizer')
    parser.add_argument('--g_beta2', default=0.9, type=float,
                        help='Beta2 for generator optimizer')
    
    # Existing parameters
    parser.add_argument('--epochs', default=1200, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--data_path', default='c:/dataset', type=str)
    parser.add_argument('--output_dir', default='F:/output/cifar10-ebm-gan-mcr-ctrl')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--save_freq', default=1, type=int)
    parser.add_argument('--cls', default=-1, type=int)
    parser.add_argument('--accumulation_steps', default=1, type=int)
    
    # Add learning rate scheduling parameters
    parser.add_argument('--min_lr', default=1e-6, type=float,
                        help='Minimum learning rate for cosine annealing')
    
    # Add checkpoint loading parameter
    parser.add_argument('--resume', default=None, type=str,
                        help='Path to checkpoint to resume training from')
    
    return parser

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depth-wise conv
        self.norm = LayerNorm(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = input + self.drop_path(x)
        return x

class ConvNeXtEnergyNet(nn.Module):
    def __init__(self, img_channels=3, hidden_dim=96):
        super().__init__()
        
        # Stem stage: patchify using conv
        self.stem = nn.Sequential(
            nn.Conv2d(img_channels, hidden_dim, kernel_size=4, stride=4),
            LayerNorm(hidden_dim)
        )
        
        # Stage 1
        self.stage1 = nn.Sequential(
            ConvNeXtBlock(hidden_dim),
            ConvNeXtBlock(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=2, stride=2),
            LayerNorm(hidden_dim * 2)
        )
        
        # Stage 2
        self.stage2 = nn.Sequential(
            ConvNeXtBlock(hidden_dim * 2),
            ConvNeXtBlock(hidden_dim * 2),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=2, stride=2),
            LayerNorm(hidden_dim * 4)
        )
        
        # Final stage
        self.stage3 = nn.Sequential(
            ConvNeXtBlock(hidden_dim * 4),
            ConvNeXtBlock(hidden_dim * 4),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim * 4, 128, kernel_size=1)
        )
        
        # Optional: add a head for energy output
        self.head = nn.Linear(128, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = x.squeeze()
        
        # Optional: use head for energy output
        # x = self.head(x)
        return x

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_ebm_gan(args) 