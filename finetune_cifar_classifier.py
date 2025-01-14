import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from pathlib import Path
from tqdm import tqdm
from train_ebm_cifar10_cl import SimSiamModel

class CifarClassifier(nn.Module):
    def __init__(self, backbone, num_classes=10):
        super().__init__()
        self.backbone = backbone
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.classifier = nn.Sequential(
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # Get features from the backbone
        with torch.no_grad():
            features = self.backbone.encoder(x).squeeze()
        
        # Pass through classifier
        return self.classifier(features)

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total

def finetune(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data preprocessing for training
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Data preprocessing for testing
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-10
    trainset = torchvision.datasets.CIFAR10(root=args.data_path, train=True,
                                          download=True, transform=train_transform)
    trainloader = DataLoader(trainset, batch_size=args.batch_size,
                           shuffle=True, num_workers=args.num_workers)
    
    testset = torchvision.datasets.CIFAR10(root=args.data_path, train=False,
                                         download=True, transform=test_transform)
    testloader = DataLoader(testset, batch_size=args.batch_size,
                          shuffle=False, num_workers=args.num_workers)

    # Load pretrained SimSiam model
    backbone = SimSiamModel(img_channels=3, hidden_dim=64).to(device)
    checkpoint = torch.load(args.pretrained_path)
    backbone.load_state_dict(checkpoint['model_state_dict'])
    
    # Create classifier
    model = CifarClassifier(backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr)
    
    best_acc = 0
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(tqdm(trainloader)):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % args.log_freq == 0:
                print(f'Epoch [{epoch}/{args.epochs}], Step [{i}/{len(trainloader)}], '
                      f'Loss: {loss.item():.4f}')
        
        # Evaluate
        test_acc = evaluate(model, testloader, device)
        print(f'Epoch [{epoch}/{args.epochs}], Test Accuracy: {test_acc:.2f}%')
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_acc,
            }, os.path.join(args.output_dir, 'best_classifier.pth'))
        
        # Save checkpoint
        if epoch % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_acc,
            }, os.path.join(args.output_dir, f'classifier_checkpoint_{epoch}.pth'))

def get_args_parser():
    import argparse
    parser = argparse.ArgumentParser('CIFAR-10 Classifier Finetuning')
    
    # Training parameters
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    
    # System parameters
    parser.add_argument('--data_path', default='c:/dataset', type=str)
    parser.add_argument('--output_dir', default='F:/output/cifar10-classifier')
    parser.add_argument('--pretrained_path', default='F:/output/cifar10-ebm-tcr/simsiam_checkpoint_30.pth', 
                       type=str, help='Path to pretrained SimSiam model')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--save_freq', default=10, type=int)
    
    return parser

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    finetune(args) 