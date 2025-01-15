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
import math
from ibot_ctrl import utils_ibot as utils

def add_noise(img,sigma_min=0.01,sigma_max=0.3,noise_type='linear'):
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
    
    
    return (1-mask)*img,mask


sampler = KarrasSampler()
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
        
    
    def forward(self, x):
        # Get representations
        x = self.encoder(x).squeeze()
        
        # Get projections
        z = self.projector(x)
        
        return z
    
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
# def simsiam_loss(p1, p1_teacher,snr=1):

#     loss_tcr = -R(p1).mean()
#     # loss_tcr_next = -R(p2).mean()
    
#     # loss_tcr = loss_tcr + loss_tcr_next
#     # loss_tcr = loss_tcr/2
#     loss_tcr /=100
    
#     # # Negative cosine similarity
#     # loss_cos = (F.cosine_similarity(h1, p2.detach(), dim=-1)*snr + 
#     #          F.cosine_similarity(h2, p1.detach(), dim=-1)*snr) * 0.5

    
#     # Negative cosine similarity
#     loss_cos = F.cosine_similarity(p1, p1_teacher.detach(), dim=-1)
    
#     loss_cos = 1-loss_cos
    
#     loss_cos = loss_cos.mean()
    
    

#     return loss_cos,loss_tcr



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
def simsiam_loss(p1, p2,snr=1):
    p1 = F.normalize(p1, p=2, dim=-1)
    p2 = F.normalize(p2, p=2, dim=-1)
    loss_tcr = -R_nonorm((p1+p2)*0.5).mean()
    loss_tcr *=1e-2

    # Negative cosine similarity
    loss_cos = (F.cosine_similarity(p1, p2.detach(), dim=-1).mean() + 
             F.cosine_similarity(p2, p1.detach(), dim=-1).mean()) * 0.5
    
    loss_cos = 1-loss_cos

    return loss_cos,loss_tcr



# Add SimSiam loss function
# def simsiam_loss(p1, p1_teacher,snr=1):
#     p1 = F.normalize(p1, p=2, dim=-1)
#     p1_teacher = F.normalize(p1_teacher, p=2, dim=-1)

#     loss_tcr = -R_nonorm((p1+p1_teacher)/2).mean()
#     # loss_tcr_next = -R(p2).mean()
    
#     # loss_tcr = loss_tcr + loss_tcr_next
#     # loss_tcr = loss_tcr/2
#     loss_tcr /=100
    
#     # # Negative cosine similarity
#     # loss_cos = (F.cosine_similarity(h1, p2.detach(), dim=-1)*snr + 
#     #          F.cosine_similarity(h2, p1.detach(), dim=-1)*snr) * 0.5

    
#     # Negative cosine similarity
#     loss_cos = F.cosine_similarity(p1, p1_teacher.detach(), dim=-1)
    
#     loss_cos = 1-loss_cos
    
#     loss_cos = loss_cos.mean()
    
    

#     return loss_cos,loss_tcr


# Add a function to visualize augmented views
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

    # Data preprocessing with two augmentations
    # transform = transforms.Compose([
    #     transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
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
    teacher_model = SimSiamModel(img_channels=3, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    teacher_model.eval()

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


    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for i, (images, _) in enumerate(tqdm(trainloader)):
            # images = images.to(device)
            img1 = images[0].to(device)
            img2 = images[1].to(device)
            it = len(trainloader) * epoch + i  # global training iteration
            
            # for idx, param_group in enumerate(optimizer.param_groups):
            #     param_group["lr"] = lr_schedule[it]
                # if idx == 0:  # only the first group is regularized
                #     param_group["weight_decay"] = wd_schedule[it]

            # img1,sigma1 = sampler.add_noise(images)
            # img2,sigma2 = sampler.add_noise(images)

            # noise = torch.randn_like(images)
            # # Sample random sigma if not provided
            # indices1 = torch.randint(
            #     0, (sampler.num_steps - 1)//2, (images.shape[0],), device=images.device
            # ).to(images.device)

            # indices2 = torch.randint(
            #     (sampler.num_steps - 1)//2, sampler.num_steps - 1, (images.shape[0],), device=images.device
            # ).to(images.device)
            # #Sample indices logarithmically to focus more on smaller noise levels

            
            # log_steps = torch.linspace(0, torch.log(torch.tensor(sampler.num_steps - 1)), sampler.num_steps - 1)
            # indices = torch.exp(log_steps[torch.randint(0, sampler.num_steps - 1, (images.shape[0],))]).long()
            # indices = indices.to(images.device)
            # # print('sampler.sigmas',sampler.sigmas)
            
            # # exit()
            # sigma = sampler.sigmas.to(images.device)[indices].view(-1,1,1,1).to(images.device)
            # sigma_next = sampler.sigmas.to(images.device)[indices+1].view(-1,1,1,1).to(images.device)
            # # Compute SNR (Signal-to-Noise Ratio) for current and next noise levels
            # snr_sigma = 1
            
            # noise = torch.randn_like(images)
            # img1 = images + noise*sigma
            
            # img2 = images + noise*sigma_next
            

            # ##==============================##
            # log_steps = torch.linspace(0, torch.log(torch.tensor(sampler.num_steps - 1)), sampler.num_steps - 1)
            # indices = torch.exp(log_steps[torch.randint(0, sampler.num_steps - 1, (images.shape[0],))]).long()
            # indices = indices.to(images.device)
            # # print('sampler.sigmas',sampler.sigmas)
            
            # # exit()
            # sigma = sampler.sigmas.to(images.device)[indices].view(-1,1,1,1).to(images.device)
            
            # # Compute SNR (Signal-to-Noise Ratio) for current and next noise levels
            # snr_sigma = 1
            
            # noise = torch.randn_like(images)
            # img1 = images 
            
            # img2 = images + noise*sigma
            
            ##==============================##
            loss = 0

            # # Forward pass      
            # img1,mask = add_mask(images,mask_ratio=0.3)
            # img2,_ = add_noise(images)
            
            # img2 = torch.flip(img2,dims=[2])

            snr_sigma = 1
            # # Forward pass
            # Z_s = model(img2)
            # Z_t = model(img1)
            # loss_cos,loss_tcr = simsiam_loss(Z_s, Z_t.detach(),snr_sigma)
            
            # loss += loss_tcr
            # loss_cos,loss_tcr = simsiam_loss(Z_t, Z_s.detach(),snr_sigma)
            
            # loss += loss_tcr
            # loss /=2
            
            Z_s = model(img2)
            Z_t = model(img1)

            # with torch.no_grad():
            #     Z_t = teacher_model(img1)
            # Compute loss
            loss_cos,loss_tcr = simsiam_loss(Z_s, Z_t,snr_sigma)
            loss = loss_tcr
            # loss += loss_tcr
            
            # Z_s = model(img1)
            
            # with torch.no_grad():
            #     Z_t = teacher_model(img2)
            # # Compute loss
            # loss_cos,loss_tcr = simsiam_loss(Z_s, Z_t.detach(),snr_sigma)
            
            # loss += loss_tcr
            # loss/=2
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            # EMA update for the teacher
            with torch.no_grad():
                m = momentum_schedule[it]  # momentum parameter
                for param_q, param_k in zip(model.parameters(), teacher_model.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
    
            if i % args.log_freq == 0:
                visualize_augmentations(model,img1,img2,images,save_path=args.output_dir,epoch=epoch)
                print(f'Epoch [{epoch}/{args.epochs}], Step [{i}/{len(trainloader)}], '
                      f'lr: {lr_schedule[it]:.4f}, '
                      f'momentum: {momentum_schedule[it]:.4f}, '
                      f'wd: {optimizer.param_groups[0]["weight_decay"]:.4f}, '
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
    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--weight_decay', default=0.001, type=float)
    parser.add_argument('--weight_decay_end', default=0.01, type=float)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--momentum_teacher', default=0.996, type=float)
    
    # Langevin dynamics parameters
    parser.add_argument('--langevin_steps', default=60, type=int)
    parser.add_argument('--step_size', default=10.0, type=float)
    parser.add_argument('--noise_scale', default=0.005, type=float)
    
    # System parameters
    parser.add_argument('--data_path', default='c:/dataset', type=str)
    parser.add_argument('--output_dir', default='F:/output/cifar10-ebm-rcl')
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