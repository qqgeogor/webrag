import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from train_mae_cifar10_edm import MaskedAutoencoderViT  # Import the MAE model
import os
from torch.cuda.amp import autocast, GradScaler
from timm.optim import create_optimizer_v2
from timm.scheduler.cosine_lr import CosineLRScheduler

class ViTForClassification(nn.Module):
    def __init__(self, pretrained_mae, num_classes=10):
        super().__init__()
        # Load pretrained encoder from MAE
        self.encoder = pretrained_mae
        # Replace classification head
        self.head = nn.Linear(pretrained_mae.embed_dim, num_classes)
        
    def forward(self, x):
        # Use encoder without masking
        x = self.encoder.forward_feature(x)
        
        # Classification from CLS token
        x = x[:, 0]
        x = self.head(x)
        return x

def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data preprocessing
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load datasets
    trainset = torchvision.datasets.CIFAR10(root='c:/dataset', train=True,
                                          download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, 
                           shuffle=True, num_workers=args.num_workers)

    testset = torchvision.datasets.CIFAR10(root='c:/dataset', train=False,
                                         download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=args.batch_size, 
                          shuffle=False, num_workers=args.num_workers)

    # Load pretrained MAE model
    mae_model = MaskedAutoencoderViT(
        img_size=32,
        patch_size=4,
        embed_dim=192,
        depth=12,
        num_heads=3,
        decoder_embed_dim=96,
        decoder_depth=4,
        decoder_num_heads=3,
        mlp_ratio=4,
        use_checkpoint=True
    ).to(device)
    
    # Load pretrained weights
    checkpoint = torch.load(args.mae_checkpoint, map_location=device)
    mae_model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded pretrained MAE model")

    # Create classification model
    model = ViTForClassification(mae_model, num_classes=10).to(device)

    # Optimizer and scheduler
    optimizer = create_optimizer_v2(
        model,
        opt='adamw',
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=args.epochs,
        lr_min=args.min_lr,
        warmup_t=args.warmup_epochs,
        warmup_lr_init=args.warmup_lr_init,
    )

    # Loss function and AMP scaler
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler() if args.use_amp else None

    # Training loop
    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            if args.use_amp:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if i % args.log_freq == args.log_freq - 1:
                print(f'Epoch: {epoch + 1}, Batch: {i + 1}, '
                      f'Loss: {running_loss / args.log_freq:.3f}, '
                      f'Acc: {100. * correct / total:.2f}%')
                running_loss = 0.0
        
        scheduler.step(epoch)
        
        # Evaluate
        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            acc = 100. * correct / total
            print(f'Epoch {epoch + 1} Test Accuracy: {acc:.2f}%')
            
            # Save best model
            if acc > best_acc:
                best_acc = acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_acc': best_acc,
                }, os.path.join(args.output_dir, 'model_best.pth'))

def get_args_parser():
    import argparse
    parser = argparse.ArgumentParser('MAE fine-tuning for CIFAR-10')
    
    # Model parameters
    parser.add_argument('--mae_checkpoint', default='./output/model_best.pth',
                        help='Path to pretrained MAE checkpoint')
    
    # Training parameters
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--warmup_lr_init', default=1e-6, type=float)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    
    # System parameters
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--output_dir', default='./finetune_output')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--eval_freq', default=1, type=int)
    
    return parser

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train_model(args) 