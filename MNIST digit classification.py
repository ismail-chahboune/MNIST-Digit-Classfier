
"""

@author: Ismail
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")



class OptimalMNISTNet(nn.Module):
    def __init__(self):
        super(OptimalMNISTNet, self).__init__()
        # Optimized for 28x28 images
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout(0.25)  # Fixed: Use Dropout instead of Dropout2d
        self.dropout2 = nn.Dropout(0.5)   # Fixed: Use Dropout instead of Dropout2d
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Input: 1x28x28
        x = F.relu(self.conv1(x)) 
        x = F.max_pool2d(x, 2)     
        x = F.relu(self.conv2(x))  
        x = F.max_pool2d(x, 2)     
        x = F.relu(self.conv3(x))  
        x = F.max_pool2d(x, 2)     
        x = x.view(-1, 64 * 3 * 3) 
        x = self.dropout1(x)      
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)       
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)



def train_optimal_mnist():
    print(" Loading MNIST dataset...")
    
    
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=test_transform)
    
   
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    print(f" Dataset loaded: {len(train_dataset)} training, {len(test_dataset)} test images")
    
    
    model = OptimalMNISTNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=4, gamma=0.5)
    
    print(f" Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    
    best_accuracy = 0
    print(f"\n Starting training for 10 epochs...")
    print("=" * 60)
    
    for epoch in range(1, 11):  
        
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        
        model.eval()
        test_loss = 0
        test_correct = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(test_loader.dataset)
        test_accuracy = 100. * test_correct / len(test_loader.dataset)
        train_accuracy = 100. * correct / total
        
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'Epoch {epoch:2d}/10:')
        print(f'  Train Acc: {train_accuracy:6.2f}% | Test Acc: {test_accuracy:6.2f}%')
        print(f'  Test Loss: {test_loss:.4f} | LR: {current_lr:.6f}')
        
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), 'best_mnist_model.pth')
            print(f'  New best model saved! ({test_accuracy:.2f}%)')
        
        print('-' * 50)
    
    return best_accuracy



def predict_digit(image_tensor, model):
    
    model.eval()
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0).to(device))
        pred = output.argmax(dim=1, keepdim=True)
        confidence = torch.exp(output).max().item()
        return pred.item(), confidence



def test_saved_model():
    
    model = OptimalMNISTNet().to(device)
    model.load_state_dict(torch.load('best_mnist_model.pth', map_location=device))
    model.eval()
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f" Loaded model test accuracy: {accuracy:.2f}%")
    return accuracy



if __name__ == '__main__':
    print(" OPTIMAL MNIST DIGIT CLASSIFIER - 10 EPOCHS (FIXED)")
    print("=" * 50)
    
    
    best_accuracy = train_optimal_mnist()
    
    print("\n" + "=" * 50)
    print(" TRAINING COMPLETED!")
    print("=" * 50)
    print(f" Best Test Accuracy: {best_accuracy:.2f}%")
    print(f" Best model saved as: best_mnist_model.pth")
    
    print("\n Testing saved model...")
    final_accuracy = test_saved_model()
    
    print("\n Model is ready for digit recognition!")
    print("Usage: digit, confidence = predict_digit(image_tensor, model)")
    print("=" * 50)
