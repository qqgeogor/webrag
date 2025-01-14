import torch
from rag_old.reptile_meta_learning import ConvNet, ReptileLearner, create_task_dataloaders
import torchvision
import torchvision.transforms as transforms

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters
    n_way = 5
    k_shot = 1
    q_query = 15
    meta_epochs = 50000
    inner_steps = 5
    eval_interval = 1000
    
    # Load CIFAR-100 dataset
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = torchvision.datasets.CIFAR100(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = torchvision.datasets.CIFAR100(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # Initialize model and learner
    model = ConvNet(num_classes=n_way)
    reptile = ReptileLearner(model, device)
    
    # Training loop
    for epoch in range(meta_epochs):
        # Create task dataloaders
        support_loader, query_loader = create_task_dataloaders(
            train_dataset, n_way, k_shot, q_query
        )
        
        # Inner loop training
        task_model = reptile.inner_loop(support_loader, inner_steps)
        
        # Meta update
        reptile.meta_update(task_model)
        
        # Evaluation
        if (epoch + 1) % eval_interval == 0:
            support_loader, query_loader = create_task_dataloaders(
                test_dataset, n_way, k_shot, q_query
            )
            
            eval_model = reptile.inner_loop(support_loader, inner_steps)
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in query_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = eval_model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            print(f'Epoch [{epoch+1}/{meta_epochs}], Accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    main() 