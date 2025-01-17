import torch
import torchvision
import torchvision.transforms as transforms
from train_ebm_cifar10_gan_r3gan_vit_cls_self_att import MaskedAutoencoderViT
import matplotlib.pyplot as plt
import seaborn as sn
from pathlib import Path
import argparse
import numpy as np
import cv2
from matplotlib.patches import Polygon
from skimage.measure import find_contours
import colorsys
import random
from PIL import Image
import torch.nn as nn
import os

company_colors = [
    (0/255, 160/255, 215/255),  # blue
    (220/255, 55/255, 60/255),  # red
    (245/255, 180/255, 0/255),  # yellow
    (10/255, 120/255, 190/255), # navy
    (40/255, 150/255, 100/255), # green
    (135/255, 75/255, 145/255), # purple
]

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image."""
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image

def random_colors(N, bright=True):
    """Generate random colors."""
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def apply_mask2(image, mask, color, alpha=0.5):
    """Apply the given mask to the image with improved visualization."""
    t = 0.2
    mi = np.min(mask)
    ma = np.max(mask)
    mask = (mask - mi) / (ma - mi)
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * np.sqrt(mask) * (mask>t)) + alpha * np.sqrt(mask) * (mask>t) * color[c] * 255
    return image

def show_attn(img, index=None, model=None):
    """Get attention maps from model and prepare visualizations."""
    w_featmap = img.shape[-2] // args.patch_size
    h_featmap = img.shape[-1] // args.patch_size

    # Get attention maps from model
    attentions = model.get_attention_maps(img.cuda(), layer_idx=args.layer_idx)
    nh = attentions.shape[1]  # number of heads


    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
    print(attentions.shape)

    if args.threshold is not None:
        # we keep only a certain percentage of the mass
        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - args.threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        # interpolate
        th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()

    # save attentions heatmaps
    prefix = f'id{index}_' if index is not None else ''
    os.makedirs(args.output_dir, exist_ok=True)
    torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), os.path.join(args.output_dir, "img" + ".png"))
    img = Image.open(os.path.join(args.output_dir, "img" + ".png"))

    attns = Image.new('RGB', (attentions.shape[2] * nh, attentions.shape[1]))
    for j in range(nh):
        #fname = os.path.join(args.output_dir, prefix + "attn-head" + str(j) + ".png")
        fname = os.path.join(args.output_dir, "attn-head" + str(j) + ".png")
        plt.imsave(fname=fname, arr=attentions[j], format='png')
        attns.paste(Image.open(fname), (j * attentions.shape[2], 0))

    return attentions, th_attn, img, attns

def show_attn_color(image, attentions, th_attn, index=None, head=[0,1,2]):
    """Create colored attention visualization."""
    # Prepare image
    M = image.max()
    m = image.min()
    span = 64
    image = ((image - m) / (M-m)) * span + (256 - span)
    image = image.mean(axis=2)
    image = np.repeat(image[:, :, np.newaxis], 3, axis=2)

    # Process attention maps
    for j in head:
        m = attentions[j]
        m *= th_attn[j]
        attentions[j] = m
    mask = np.stack([attentions[j] for j in head])

    # Setup figure
    figsize = tuple([i / 100 for i in args.image_size])
    fig = plt.figure(figsize=figsize, frameon=False, dpi=100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    # Process mask
    N = mask.shape[0] if len(mask.shape) == 3 else 1
    if N == 1:
        mask = mask[None, :, :]

    # Apply mask processing
    for i in range(N):
        mask[i] = mask[i] * (mask[i] == np.amax(mask, axis=0))
    a = np.cumsum(mask, axis=0)
    for i in range(N):
        mask[i] = mask[i] * (mask[i] == a[i])

    # Setup visualization
    height, width = image.shape[:2]
    ax.set_ylim(height, 0)
    ax.set_xlim(0, width)
    ax.axis('off')

    # Create masked image
    masked_image = 0.1 * image.astype(np.uint32).copy()
    colors = company_colors[:N]
    
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        masked_image = apply_mask2(masked_image, _mask, color, alpha=1)

    # Save visualization
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    ax.axis('image')
    prefix = f'id{index}_' if index is not None else ''
    fname = os.path.join(args.output_dir, "attn_color.png")
    fig.savefig(fname)
    return Image.open(fname)

def visualize_attention(args):
    """Main visualization function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-10
    testset = torchvision.datasets.CIFAR10(root=args.data_path, train=False,
                                         download=True, transform=transform)
    
    # Create dataloader
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                           shuffle=True, num_workers=2)

    # Initialize model
    model = MaskedAutoencoderViT(
        img_size=32, 
        patch_size=4, 
        in_chans=3,
        embed_dim=192,
        decoder_embed_dim=192,
        depth=6,
        num_heads=3
    ).to(device)

    # Load checkpoint
    if os.path.isfile(args.checkpoint_path):
        print(f"Loading checkpoint from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        if 'discriminator_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['discriminator_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        raise FileNotFoundError(f"No checkpoint found at {args.checkpoint_path}")

    model.eval()

    # Process images and visualize attention
    print("Generating attention visualizations...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(testloader):
            if i >= args.num_samples:
                break

            # Process single image
            img = images[0:1]
            attentions, th_attn, pic_i, pic_attn = show_attn(img, index=i,model=model)
            pic_attn_color = show_attn_color(
                img[0].permute(1, 2, 0).cpu().numpy(),
                attentions,
                th_attn,
                index=i
            )

            # Create final visualization
            final_pic = Image.new('RGB', (pic_i.size[1] * 2 + pic_attn.size[0], pic_i.size[1]))
            final_pic.paste(pic_i, (0, 0))
            final_pic.paste(pic_attn_color, (pic_i.size[1], 0))
            final_pic.paste(pic_attn, (pic_i.size[1] * 2, 0))
            final_pic.save(os.path.join(args.output_dir, f"sample_{i}_attn.png"))

            print(f"Processed sample {i+1}/{args.num_samples}")
    
    print(f"Visualizations saved to {args.output_dir}")

def get_args_parser():
    parser = argparse.ArgumentParser('MAE ViT Attention Visualization')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_path', default='./data', type=str,
                        help='Path to CIFAR-10 dataset')
    parser.add_argument('--output_dir', default='./attention_vis',
                        help='Path to save visualizations')
    parser.add_argument('--num_samples', default=5, type=int,
                        help='Number of samples to visualize')
    parser.add_argument('--layer_idx', default=-1, type=int,
                        help='Index of transformer layer to visualize (-1 for last layer)')
    parser.add_argument('--patch_size', default=4, type=int,
                        help='Patch size of the model')
    parser.add_argument('--image_size', default=(32, 32), type=int, nargs='+',
                        help='Image size for visualization')
    parser.add_argument('--threshold', default=0.6, type=float,
                        help='Threshold for attention visualization')
    return parser

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    visualize_attention(args)