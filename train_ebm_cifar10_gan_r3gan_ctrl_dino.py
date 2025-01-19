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
    def __init__(self, img_channels=3, hidden_dim=64, img_size=32):
        super().__init__()
        
        # Calculate number of downsampling steps needed
        self.n_downsample = int(np.log2(img_size) - 2)  # Final feature map should be 4x4
        
        layers = [
            # Initial conv: [B, 3, img_size, img_size] -> [B, 64, img_size/2, img_size/2]
            nn.Conv2d(img_channels, hidden_dim, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2),
            ResBlock(hidden_dim,hidden_dim,1)
        ]
        
        # Add downsampling layers
        current_dim = hidden_dim
        for i in range(self.n_downsample - 1):
            next_dim = min(current_dim * 2, 512)
            layers.extend([
                nn.Conv2d(current_dim, next_dim, 4, 2, 1),
                nn.BatchNorm2d(next_dim),
                nn.LeakyReLU(0.2),
                ResBlock(next_dim,next_dim,1)
            ])
            current_dim = next_dim
        
        # Final conv to scalar energy: [B, current_dim, 4, 4] -> [B, 128, 1, 1]
        layers.append(nn.Conv2d(current_dim, 128, 4, 1, 0))
        
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(128, 1)
        
        # Initialize weights properly
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
    
    def forward(self, x):
        logits = self.net(x).squeeze()
        logits = self.head(logits)
        # print(x.shape)
        # logits = self.head(logits)
        # Add regularization term to prevent collapse
        # reg_term = 0.1 * (logits ** 2).mean()
        # logits = logits# + reg_term
        #logits = -F.logsigmoid(logits)
        return logits




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
    
    Z = Z
    cov = Z.T @ Z
    I = torch.eye(cov.size(-1)).to(Z.device)
    alpha = c/(b*eps)
    
    cov = alpha * cov +  I

    out = 0.5*torch.logdet(cov)
    return out.mean()

def mcr(Z1,Z2):
    return R(torch.cat([Z1,Z2],dim=0))-0.5*R(Z1)-0.5*R(Z2)



# Add SimSiam loss function
def simsiam_loss(p1, p2, h1, h2):

    loss_tcr = -R(p1).mean()
    loss_tcr *=1e-2

    # Negative cosine similarity
    loss_cos = (F.cosine_similarity(h1, p2.detach(), dim=-1).mean() + 
             F.cosine_similarity(h2, p1.detach(), dim=-1).mean()) * 0.5
    
    loss_cos = 1-loss_cos

    return loss_cos,loss_tcr

def tcr_loss(Z1,Z2):
    Z1 = F.normalize(Z1,p=2,dim=-1)
    Z2 = F.normalize(Z2,p=2,dim=-1)
    Z = (Z1+Z2)/2
    return R_nonorm(Z)


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
            nn.BatchNorm2d(hidden_dim * 4),
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

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        # self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                # nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.2)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.leaky_relu(out, 0.2)
        return out
    

    # def forward(self, x):
    #     out = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
    #     out = self.bn2(self.conv2(out))
    #     out += self.shortcut(x)
    #     out = F.leaky_relu(out, 0.2)
    #     return out




# Add Generator class
class Generator(nn.Module):
    def __init__(self, latent_dim=100, hidden_dim=64, img_size=32):
        super().__init__()
        
        # Calculate number of upsampling steps needed
        self.n_upsample = int(np.log2(img_size) - 2)  # Start from 4x4
        
        # Initial projection and reshape
        self.project = nn.Sequential(
            nn.Linear(latent_dim+128, hidden_dim * 8 * 4 * 4),
            nn.LeakyReLU(0.2),
        )
        
        # Build upsampling layers
        layers = []
        current_dim = hidden_dim * 8
        
        for i in range(self.n_upsample):
            next_dim = current_dim // 2
            layers.extend([
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(current_dim, next_dim, 3, 1, 1),
                nn.BatchNorm2d(next_dim),
                nn.LeakyReLU(0.2),
                ResBlock(next_dim,next_dim,1)
            ])
            current_dim = next_dim
        
        # Final layer
        layers.extend([
            nn.ConvTranspose2d(current_dim, 3, 3, 1, 1),
            nn.Tanh()
        ])
        
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, z, c=None):
        if c is not None:
            z = torch.cat([z, c], dim=-1)
        x = self.project(z)
        x = x.view(x.size(0), -1, 4, 4)
        return self.net(x)

# Add TwoCropsTransform class
class TwoCropsTransform:
    def __init__(self, base_transform,base_transform2):
        self.base_transform = base_transform
        self.base_transform2 = base_transform2

    def __call__(self, x):
        q = self.base_transform(x)  # First augmented view
        k = self.base_transform2(x)  # Second augmented view
        return [q, k]  # Returns a list containing both views


# Modify training function
def train_ebm_gan(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Modified data preprocessing for configurable image size
    transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform2 = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


    
    # Load dataset using ImageFolder
    trainset = torchvision.datasets.ImageFolder(
        root=os.path.join(args.data_path, ''),
        transform=TwoCropsTransform(transform,transform2)
    )
    
    # Filter the dataset to only include class 1
    if args.cls!=-1:
        class_1_indices = [i for i, label in enumerate(trainset.targets) if label == args.cls]
        trainset.data = trainset.data[class_1_indices]
        trainset.targets = [trainset.targets[i] for i in class_1_indices]
    

    trainloader = DataLoader(trainset, batch_size=args.batch_size,
                           shuffle=True, num_workers=args.num_workers)

    # Initialize models with configurable image size
    generator = Generator(latent_dim=args.latent_dim, hidden_dim=64, img_size=args.img_size).to(device)
    discriminator = EnergyNet(img_channels=3, hidden_dim=64, img_size=args.img_size).to(device)
    
    # Optimizers
    g_optimizer = torch.optim.AdamW(
        generator.parameters(), 
        lr=args.g_lr, 
        betas=(args.g_beta1, args.g_beta2)
    )
    d_optimizer = torch.optim.AdamW(
        discriminator.parameters(), 
        lr=args.d_lr, 
        betas=(0.5, 0.999)
    )
    
    import utils_ibot as utils
    g_lr_schedule = utils.cosine_scheduler(
        args.g_lr,  # linear scaling rule
        args.min_lr,
        args.epochs, len(trainloader),
        warmup_epochs=args.warmup_epochs,
    )
    d_lr_schedule = utils.cosine_scheduler(
        args.d_lr,  # linear scaling rule
        args.min_lr,
        args.epochs, len(trainloader),
        warmup_epochs=args.warmup_epochs,
    )
    start_epoch = 0
    
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
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")
    
    # Training loop
    for epoch in range(start_epoch,args.epochs):
        generator.train()
        discriminator.train()
        
        for i, (real_samples, _) in enumerate(tqdm(trainloader)):
            real_samples,aug_samples = real_samples

            g_optimizer.param_groups[0]['lr'] = g_lr_schedule[epoch*len(trainloader)+i]
            d_optimizer.param_groups[0]['lr'] = d_lr_schedule[epoch*len(trainloader)+i]

            batch_size = real_samples.size(0)
            real_samples = real_samples.to(device)
            aug_samples = aug_samples.to(device)

            # Train Discriminator
            for _ in range(args.n_critic):  # Train discriminator more frequently
                d_optimizer.zero_grad()
                
                # Generate fake samples
                z = torch.randn(batch_size, args.latent_dim, device=device)
                c_real = discriminator.net(real_samples.detach()).squeeze()

                real_samples = real_samples.detach().requires_grad_(True)
                fake_samples = generator(z,c_real.detach()).detach().requires_grad_(True)
                
                # c_fake = discriminator.net(aug_samples.detach()).squeeze()
                c_fake = discriminator.net(fake_samples.detach()).squeeze()
                
                loss_cos,loss_tcr = simsiam_loss(c_real,c_fake,c_real,c_fake)
                cl_loss = loss_tcr+loss_cos
                # Compute energies
                real_energy = discriminator(real_samples)
                fake_energy = discriminator(fake_samples)
                
                realistic_logits = real_energy - fake_energy
                d_loss = F.softplus(-realistic_logits)
                # Improved EBM-GAN discriminator loss
                # d_loss = (F.softplus(real_energy) + (-fake_energy))
                
                r1 = zero_centered_gradient_penalty(real_samples, real_energy)
                r2 = zero_centered_gradient_penalty(fake_samples, fake_energy)

                d_loss = d_loss + args.gp_weight/2 * (r1 + r2)+cl_loss 
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
            c_real = discriminator.net(real_samples.detach()).squeeze()

            fake_samples = generator(z,c_real.detach())
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
                      f'cl_loss: {cl_loss.item():.4f}, '
                      f'tcr_loss: {loss_tcr.item():.4f}, '
                      f'cos_loss: {loss_cos.item():.4f}, '
                      f'r1: {r1.mean().item():.4f}, r2: {r2.mean().item():.4f}, '
                      f'Real Energy: {real_energy.mean().item():.4f}, '
                      f'Fake Energy: {fake_energy.mean().item():.4f}, '
                      f'G_LR: {current_g_lr:.6f}, D_LR: {current_d_lr:.6f}'
                      )
        
        # Step the schedulers at the end of each epoch

        
        real_samples = next(iter(trainloader))[0][0].to(device)
        
        # Save samples and model checkpoints
        if epoch % args.save_freq == 0:
            save_gan_samples(generator, discriminator, epoch, args.output_dir, device,real_samples=real_samples)
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
            }, os.path.join(args.output_dir, f'ebm_gan_checkpoint_{epoch}.pth'))

def save_gan_samples(generator, discriminator, epoch, output_dir, device, n_samples=36,real_samples=None):
    generator.eval()
    discriminator.eval()
    real_samples = real_samples[:n_samples]
    batch_size = real_samples.size(0)
    with torch.no_grad():
        z = torch.randn(batch_size, args.latent_dim, device=device)
        c_real = discriminator.net(real_samples.detach()).squeeze()

        fake_samples = generator(z,c_real.detach())
        
        # Changed 'range' to 'value_range'
        grid = make_grid(fake_samples, nrow=6, normalize=True, value_range=(-1, 1))
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.cpu().permute(1, 2, 0))
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'gan_samples_epoch_{epoch}.png'))


        # Changed 'range' to 'value_range'
        grid = make_grid(real_samples, nrow=6, normalize=True, value_range=(-1, 1))
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
    parser = argparse.ArgumentParser('EBM-GAN training for custom image datasets')
    
    # Add GAN-specific parameters
    parser.add_argument('--latent_dim', default=128, type=int)
    parser.add_argument('--g_lr', default=2e-4, type=float)
    parser.add_argument('--d_lr', default=2e-4, type=float)
    parser.add_argument('--n_critic', default=1, type=int,
                        help='Number of discriminator updates per generator update')
    parser.add_argument('--gp_weight', default=0.05, type=float,
                        help='Weight of gradient penalty')
    
    parser.add_argument('--warmup_epochs', default=10, type=int,
                        help='Number of warmup epochs')
    # Modify learning rates
    parser.add_argument('--g_beta1', default=0.5, type=float,
                        help='Beta1 for generator optimizer')
    parser.add_argument('--g_beta2', default=0.999, type=float,
                        help='Beta2 for generator optimizer')
    
    parser.add_argument('--cls', default=-1, type=int,
                        help='Class to train on')
    
    # Existing parameters
    parser.add_argument('--epochs', default=1200, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    
    parser.add_argument('--data_path', default='c:/dataset/tiny-imagenet/train/', type=str,
                        help='Path to the data directory containing train folder')
    parser.add_argument('--output_dir', default='F:/output/tiny_imagenet_ctrl_dino')
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
    
    # Add image size parameter
    parser.add_argument('--img_size', default=32, type=int,
                        help='Size of input images (assumes square images)')
    
    return parser

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_ebm_gan(args) 