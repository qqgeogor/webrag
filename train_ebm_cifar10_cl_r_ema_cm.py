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
from train_mae_cifar10_edm import MaskedAutoencoderViT
import numpy as np
from tqdm import tqdm

import torch.nn.functional as F
import copy

from karas_sampler import KarrasSampler,get_sigmas_karras
from ibot_ctrl import utils_ibot as utils
import math


# Masked Autoencoder approach
def add_mask(img, mask_ratio=0.75,patch_size=4):
    # Randomly mask patches
    B, C, H, W = img.shape
    n_patches = (H//patch_size) * (W//patch_size)  # assuming 16x16 patches
    n_mask = int(mask_ratio * n_patches)
    
    # Create random mask
    noise = torch.rand(B, n_patches)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    mask = torch.ones([B, n_patches])
    mask[:, :n_mask] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    mask = 1-mask

    # Reshape mask to match image patches
    n_patches_h = H//patch_size
    n_patches_w = W//patch_size
    mask = mask.reshape(B, n_patches_h, n_patches_w)
    mask = mask.repeat_interleave(patch_size, 1).repeat_interleave(patch_size, 2)
    mask = mask.unsqueeze(1).repeat(1, C, 1, 1).to(img.device)
    
    
    return (1-mask)*img+mask*torch.randn_like(img),mask

def pgd_attack(model, images, eps=8. / 255., alpha=2. / 255., iters=20, advFlag=None, forceEval=True, randomInit=True):
    # images = images.to(device)
    # labels = labels.to(device)
    loss = nn.CrossEntropyLoss()

    # init
    if randomInit:
        delta = torch.rand_like(images) * eps * 2 - eps
    else:
        delta = torch.zeros_like(images)
    delta = torch.nn.Parameter(delta, requires_grad=True)

    for i in range(iters):

        model.zero_grad()
        with torch.no_grad():
            original_features = model.get_features(images)
        adversarial_features = model.get_features(images+delta)
        
        # AIR loss: combination of feature distance and regularization
        cost = -F.cosine_similarity(original_features, adversarial_features, dim=-1).mean() 

        # cost.backward()
        delta_grad = torch.autograd.grad(cost, [delta])[0]

        delta.data = delta.data + alpha * delta_grad.sign()
        delta.grad = None
        delta.data = torch.clamp(delta.data, min=-eps, max=eps)
        delta.data = torch.clamp(images + delta.data, min=0, max=1) - images

    model.zero_grad()

    return (images + delta).detach()



def generate_air_samples(model, images,  epsilon=8. / 255., alpha=2. / 255., num_steps=20, reg_weight=1.0):
    """
    Generate samples with Adversarial Invariant Regularization
    
    Args:
        model: neural network model
        images: original images
        epsilon: maximum perturbation amount
        alpha: step size for each iteration
        num_steps: number of PGD steps
        reg_weight: weight for invariance regularization
    """
    # Initialize adversarial samples
    adv_images = images.clone().detach()
    adv_images = adv_images + torch.empty_like(adv_images).uniform_(-epsilon, epsilon)
    adv_images = torch.clamp(adv_images, -1, 1)

    for _ in range(num_steps):
        adv_images.requires_grad = True
        
        # Get features for both original and adversarial images
        with torch.no_grad():
            original_features = model.get_features(images)
        adversarial_features = model.get_features(adv_images)
        
        # AIR loss: combination of feature distance and regularization
        feature_loss = -F.cosine_similarity(original_features, adversarial_features, dim=-1).mean()
        
        # Add invariance regularization
        reg_loss = R(adversarial_features) - R(original_features)
        
        # Combined loss with regularization
        loss = feature_loss + reg_weight * reg_loss
        
        # Update adversarial samples
        grad = torch.autograd.grad(loss, adv_images)[0]
        adv_images = adv_images.detach() + alpha * grad.sign()
        
        # Project back to epsilon ball
        delta = torch.clamp(adv_images - images, -epsilon, epsilon)
        adv_images = torch.clamp(images + delta, -1, 1)
    
    return adv_images.detach()


def generate_adversarial_samples(model, images, epsilon=0.03, alpha=0.007, num_steps=10):
    """
    Generate adversarial samples using PGD (Projected Gradient Descent)
    
    Args:
        model: neural network model
        images: original images
        epsilon: maximum perturbation amount
        alpha: step size for each iteration
        num_steps: number of PGD steps
    """
    # Initialize adversarial samples with original images
    adv_images = images.clone().detach()
    
    # Add small random noise to start with
    adv_images = adv_images + torch.empty_like(adv_images).uniform_(-epsilon, epsilon)
    adv_images = torch.clamp(adv_images, -1, 1)

    for _ in range(num_steps):
        adv_images.requires_grad = True
        
        # Get features for both original and adversarial images
        with torch.no_grad():
            original_features = model.get_features(images)
        adversarial_features = model.get_features(adv_images)
        
        # Maximize feature similarity (negative of cosine similarity)
        # loss = -F.cosine_similarity(original_features, adversarial_features, dim=-1).mean()
        loss = simsiam_loss(adversarial_features,original_features,adversarial_features,original_features)
        
        # Compute gradients
        grad = torch.autograd.grad(loss, adv_images)[0]
        
        # Update adversarial images
        adv_images = adv_images.detach() + alpha * grad.sign()
        
        # Project back to epsilon ball
        delta = torch.clamp(adv_images - images, -epsilon, epsilon)
        adv_images = torch.clamp(images + delta, -1, 1)
    
    return adv_images.detach()

class LangevinSampler:
    def __init__(self, n_steps=10, step_size=10, noise_scale=0.005):
        self.n_steps = n_steps
        self.step_size = step_size
        self.noise_scale = noise_scale
    
    def sample(self, model,teacher_model, x_init, return_trajectory=False):
        model.eval()
        # Ensure x requires gradients
        x = x_init.clone().detach().requires_grad_(True)
        # Get a shuffled version of x_init
        x_shuffled = x_init[torch.randperm(x_init.size(0))].clone().detach()
        
        

        trajectory = [x.clone().detach()] if return_trajectory else None
        
        for _ in range(self.n_steps):
            # Ensure x requires gradients at each step
            if not x.requires_grad:
                x.requires_grad_(True)
                
            # Compute energy gradient
            Zs = model.get_features(x)
            with torch.no_grad():
                Zt = teacher_model.get_features(x_init.detach())
            
            # loss_cos = -F.cosine_similarity(Zs,Zt,dim=-1).mean()

            # loss_cos = -R(Zs).mean()
            
            loss_cos = F.cosine_similarity(Zs,Zt,dim=-1).abs().mean()-R(Zs).mean()/100
            
            # Compute gradients
            if x.grad is not None:
                x.grad.zero_()
            grad = torch.autograd.grad(loss_cos, x, create_graph=False, retain_graph=True)[0]
            grad_norm = grad.norm(2, dim=(1, 2, 3), keepdim=True)
            grad = grad / (grad_norm + 1e-8)

            # Langevin dynamics update
            noise = torch.randn_like(x) * self.noise_scale

            x = x.detach()  # Detach from computation graph
            x = x - self.step_size * grad + noise  # Update x
            x.requires_grad_(True)  # Re-enable gradients
            x = torch.clamp(x, -1, 1)  # Keep samples in valid range
            
            if return_trajectory:
                trajectory.append(x.clone().detach())
        
        return (x.detach(), trajectory) if return_trajectory else x.detach()
    

def add_noise(img,sigma_min=0.01,sigma_max=0.3,noise_type='log'):
    """Add DDPM-style noise to images"""
    u = torch.rand(img.shape[0]).to(img.device)
    if noise_type == 'log':
        log_sigma_min = math.log(sigma_min)
        log_sigma_max = math.log(sigma_max)
        
        log_sigma = log_sigma_min + u * (log_sigma_max - log_sigma_min)
        sigma = torch.exp(log_sigma).view(-1, 1, 1, 1)
    elif noise_type == 'linear':
        sigma = sigma_min + u * (sigma_max - sigma_min)
        sigma = sigma.view(-1, 1, 1, 1)
    else:
        raise ValueError(f"Invalid noise type: {noise_type}")

    noise = torch.randn_like(img)
    img = img * (1 - sigma) + noise * sigma
    return img, sigma



def add_adaptive_grad_noise(model,teacher_model, images,sigma_min=0.01, sigma_max=0.5, grad_weight=1,n_iter=1,noise_type='linear'):
    """
    Add noise based on gradient direction and random noise
    Args:
        model: neural network
        images: input images
        base_sigma: base noise level
        grad_weight: weight for gradient component
    """
    noisy_images = images
    # Randomly select noise level for each image in batch
    batch_size = images.shape[0]
    # Generate random sigma values for each image using log sampling
    if noise_type == 'log':
        log_sigma_min = math.log(sigma_min)
        log_sigma_max = math.log(sigma_max)
        log_sigmas = torch.rand(batch_size, device=images.device) * (log_sigma_max - log_sigma_min) + log_sigma_min
        base_sigma = torch.exp(log_sigmas).view(-1, 1, 1, 1)
    elif noise_type == 'linear':
        u =  torch.rand(batch_size, device=images.device)
        base_sigma = sigma_min + u * (sigma_max - sigma_min)

    base_sigma = base_sigma.view(-1, 1, 1, 1)
    
    for i in range(n_iter):
        # Enable gradients for input
        noisy_images.requires_grad_(True)
        
        with torch.no_grad():
            p2 = teacher_model.get_features(noisy_images)
            p2 = p2.detach()
        
        # Get model output and compute loss
        with torch.enable_grad():
            p1 = model.get_features(noisy_images)
            loss_cos, loss_tcr = simsiam_loss(p1, p2, p1, p2)
            loss = loss_tcr  # Using same loss as training
            
            # Get gradient w.r.t input
            grad = torch.autograd.grad(loss, images)[0]
            
        # Normalize gradient
        grad_norm = grad.norm(2, dim=1,keepdim=True)
        grad = grad / (grad_norm + 1e-8)

        # Random noise component
        random_noise = torch.randn_like(images)
        random_noise = random_noise / random_noise.norm(dim=1, keepdim=True)
        
        # Combine both types of noise
        effective_noise = (1 - grad_weight) * random_noise + grad_weight * grad
        effective_noise = effective_noise / effective_noise.norm(dim=1, keepdim=True)
        
        # Add scaled noise
        noisy_images = images + base_sigma * effective_noise
        noisy_images = torch.clamp(noisy_images, -1, 1)
    
    return noisy_images.detach()


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
            nn.BatchNorm1d(proj_dim)
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
        return p
    
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


# Add SimSiam loss function
def simsiam_loss(p1, p2, h1, h2):
    p1 = F.normalize(p1, p=2, dim=-1)
    p2 = F.normalize(p2, p=2, dim=-1)
    loss_tcr = -R_nonorm(p1).mean()
    loss_tcr *=1e-2

    # Negative cosine similarity
    loss_cos =  F.cosine_similarity(h1, p2, dim=-1).mean()
    
    loss_cos = 1-loss_cos

    return loss_cos,loss_tcr


@torch.no_grad()
def visualize_augmentations(model,view1,view2, image, save_path=None,epoch=0):
    """
    Visualize augmented views and their similarity
    Args:
        model: trained SimSiam model
        image: original image tensor [C, H, W]
        save_path: path to save visualization
    """
    model.eval()
    view1,view2 = view1[0],view2[0]
    # Denormalize images for visualization
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(view1.device)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(view1.device)
    view1_show = view1 * std + mean
    view2_show = view2 * std + mean
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    ax1.imshow(view1_show.squeeze().permute(1, 2, 0).cpu())
    ax1.set_title('View 1')
    ax1.axis('off')
    
    ax2.imshow(view2_show.squeeze().permute(1, 2, 0).cpu())
    ax2.set_title('View 2')
    ax2.axis('off')
    
    plt.suptitle(f'Feature Similarity: {1:.3f}')
    
    if save_path:
        plt.savefig(os.path.join(save_path,f'epoch_{epoch}.png'))
    plt.close()


# Modify the training function
def train_ebm(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    scaler = GradScaler(enabled=args.use_amp)
    # Data preprocessing with two augmentations
    transform = transforms.Compose([
        # transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        # transforms.RandomGrayscale(p=0.2),
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
    teacher_model = SimSiamModel(img_channels=3, hidden_dim=64).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 0

    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        teacher_model.load_state_dict(checkpoint['teacher_model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

    lr_schedule = utils.cosine_scheduler(
        args.lr,  # linear scaling rule
        args.min_lr,
        args.epochs, len(trainloader),
        warmup_epochs=args.warmup_epochs,
    )

    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(trainloader),
    )

    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                            args.epochs, len(trainloader))

    names_q, params_q, names_k, params_k = [], [], [], []
    for name_q, param_q in model.named_parameters():
        names_q.append(name_q)
        params_q.append(param_q)
    for name_k, param_k in teacher_model.named_parameters():
        names_k.append(name_k)
        params_k.append(param_k)
    names_common = list(set(names_q) & set(names_k))
    params_q = [param_q for name_q, param_q in zip(names_q, params_q) if name_q in names_common]
    params_k = [param_k for name_k, param_k in zip(names_k, params_k) if name_k in names_common]

    sampler = KarrasSampler()
    lan_sampler = LangevinSampler()
    # Training loop
    for epoch in range(start_epoch,args.epochs):
        model.train()
        total_loss = 0
        
        for i, (images, _) in enumerate(tqdm(trainloader)):
            it = len(trainloader) * epoch + i  # global training iteration

            img1, img2 = images[0].to(device), images[1].to(device)  # Unpack the two views
            images = img1
            img1, sigma1 = add_noise(images)
            img2, sigma2 = add_noise(images)
            # img2 = images
            # log_steps = torch.linspace(0, torch.log(torch.tensor(sampler.num_steps - 1)), sampler.num_steps - 1)
            # indices = torch.exp(log_steps[torch.randint(0, sampler.num_steps - 1, (images.shape[0],))]).long()
            # indices = indices.to(images.device)
            # sigma = sampler.sigmas.to(images.device)[indices].view(-1,1,1,1).to(images.device)
            # sigma_next = sampler.sigmas.to(images.device)[indices+1].view(-1,1,1,1).to(images.device)
            # # Compute SNR (Signal-to-Noise Ratio) for current and next noise levels
            # snr_sigma = 1

            # noise = torch.randn_like(images)
            # img1 = images + noise*sigma
            # img2 = images + noise*sigma_next
            
            # img1,img2 = generate_adversarial_samples(model,images),images
            # img1 = images#,mask1 = add_mask(images,mask_ratio=0.4,patch_size=4)
            # img2 = images#,mask2 = add_mask(images,mask_ratio=0.4,patch_size=4)
            
            
            # img1 = add_adaptive_grad_noise(model,teacher_model,images)
            # img2 = add_adaptive_grad_noise(model,teacher_model,images) #+ noise*sigma_next
            
            
            
            # Forward pass
            args.use_amp = True

            with autocast(enabled=args.use_amp):
                p1, p2, h1, h2 = model(img1, img2)
                with torch.no_grad():
                    p1_t, p2_t, h1_t, h2_t = teacher_model(img1, img2)
                # Compute loss

                loss_cos1,loss_tcr1 = simsiam_loss(p1, p2_t.detach(), h1, h2_t)
                loss_cos2,loss_tcr2 = simsiam_loss(p2, p1_t.detach(), h2, h1_t)
                
                loss_cos = (loss_cos1 + loss_cos2)/2
                loss_tcr = (loss_tcr1 + loss_tcr2)/2
                
                
                # loss_cos,loss_tcr = simsiam_loss(p1, p2, h1, h2)
                loss = loss_cos+loss_tcr
            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            
            # EMA update for the teacher
            with torch.no_grad():
                m = momentum_schedule[it]  # momentum parameter
                for param_q, param_k in zip(model.parameters(), teacher_model.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            
            if i % args.log_freq == 0:
                visualize_augmentations(model,img1,img2,images,args.output_dir,epoch)
                print(f'Epoch [{epoch}/{args.epochs}], Step [{i}/{len(trainloader)}], '
                      f'Loss: {loss.item():.4f}, Loss_cos: {loss_cos.item():.4f}, Loss_tcr: {loss_tcr.item():.4f}')

        # Add visualization of augmentations periodically
        if epoch % args.save_freq == 0:
            
            # Save model checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'teacher_model_state_dict': teacher_model.state_dict(),
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
    parser.add_argument('--min_lr', default=1e-5, type=float)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--weight_decay', default=0.04, type=float)
    parser.add_argument('--weight_decay_end', default=0.4, type=float)
    parser.add_argument('--momentum_teacher', default=0.9994, type=float)
    
    # Langevin dynamics parameters
    parser.add_argument('--langevin_steps', default=60, type=int)
    parser.add_argument('--step_size', default=10.0, type=float)
    parser.add_argument('--noise_scale', default=0.005, type=float)
    
    # System parameters
    parser.add_argument('--data_path', default='c:/dataset', type=str)
    parser.add_argument('--output_dir', default='F:/output/cifar10-ebm-cl-r-ema-cm')
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