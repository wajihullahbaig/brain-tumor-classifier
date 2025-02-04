import os
import random
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def set_seed(seed: Optional[int] = 42) -> None:
    """Set all random seeds for reproducibility"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)

class Expert(nn.Module):
    """Enhanced expert with residual connections and batch norm"""
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, hidden_dim, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, in_dim, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_dim)
        
        # Skip connection if dimensions change
        self.skip = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1),
            nn.BatchNorm2d(in_dim)
        ) if in_dim != hidden_dim else nn.Identity()
        
    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)

class Gate(nn.Module):
    """Enhanced gating network with attention and regularization"""
    def __init__(self, in_dim, num_experts, temperature=0.1):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1)*temperature)
        
        # Spatial attention
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 4, 1),
            nn.BatchNorm2d(in_dim // 4),
            nn.ReLU(),
            nn.Conv2d(in_dim // 4, 1, 1),
            nn.Sigmoid()
        )
        
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Expert selection
        self.expert_gate = nn.Sequential(
            nn.Linear(in_dim * 2, in_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_dim // 2, num_experts)
        )
        
    def forward(self, x, training=True):
        # Spatial attention
        spatial_weights = self.spatial_gate(x)
        x = x * spatial_weights
        
        # Channel attention with both avg and max pooling
        avg_feat = self.avg_pool(x).flatten(1)
        max_feat = self.max_pool(x).flatten(1)
        combined = torch.cat([avg_feat, max_feat], dim=1)
        
        # Expert selection with temperature scaling
        gates = self.expert_gate(combined)
        if training:
            gates = gates / self.temperature
        
        return F.softmax(gates, dim=1)

class MoE(nn.Module):
    """Improved Mixture of Experts with enhanced components and regularization"""
    def __init__(self, num_experts=8, expert_hidden_dim=256, temp=0.1, num_classes=4):
        super().__init__()
        # Backbone with larger feature maps
        self.backbone = timm.create_model(
            'efficientnet_b2',
            pretrained=True,
            features_only=True,
            out_indices=(1,)  # Use Stage 1 for 8x8 features
        )
        
        # Get feature dimensions
        dummy = torch.randn(2, 3, 224, 224)
        features = self.backbone(dummy)
        self.feature_dim = features[0].shape[1]
        
        # Enhanced gating network
        self.gate = Gate(self.feature_dim, num_experts, temp)
        
        # Enhanced experts
        self.experts = nn.ModuleList([
            Expert(self.feature_dim, expert_hidden_dim)
            for _ in range(num_experts)
        ])
        
        # Enhanced classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(self.feature_dim, self.feature_dim, 3, padding=1),
            nn.BatchNorm2d(self.feature_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, num_classes)
        )
    
    def forward(self, x, labels=None, n_classes=None, return_losses=False):
        features = self.backbone(x)[0]
        
        # Get expert weights
        expert_weights = self.gate(features, self.training)
        
        # Process through experts
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(features))
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # Combine expert outputs
        combined = torch.sum(
            expert_outputs * expert_weights.view(-1, expert_weights.size(1), 1, 1, 1),
            dim=1
        )
        
        # Classification
        output = self.classifier(combined)
        
        if return_losses and self.training:
            # Load balancing loss
            balance_loss = self.calculate_balance_loss(expert_weights, labels, n_classes)
            specialization_loss = self.calculate_specialization_loss(expert_weights, labels, n_classes)
            
            # Auxiliary losses dictionary
            losses = {
                'balance_loss': balance_loss,
                'specialization_loss': specialization_loss
            }
            return output, losses
            
        return output
    
    def calculate_balance_loss(self, expert_weights, labels, num_classes):
        num_experts = expert_weights.size(1)
        
        # Ideal usage should be uniform across all experts
        ideal_usage = torch.ones(num_experts, device=expert_weights.device) / num_experts  # [num_experts]
                                    
        # Get actual expert usage
        expert_usage = expert_weights.mean(0)  # [num_experts]
        
        # Use KL divergence to measure the difference between ideal and actual usage
        balance_loss = F.kl_div(torch.log(expert_usage), ideal_usage, reduction='batchmean')
        
        return balance_loss
        
    def calculate_specialization_loss(self, expert_weights, labels, num_classes):
        batch_size = expert_weights.size(0)
        
        # Get class-expert correlations
        labels_onehot = F.one_hot(labels, num_classes).float()
        expert_class_correlation = torch.matmul(
            expert_weights.t(),  # [num_experts, batch_size]
            labels_onehot        # [batch_size, num_classes]
        ) / batch_size          # [num_experts, num_classes]
        
        # Encourage each expert to specialize by maximizing the correlation for one class
        # and minimizing for others
        specialization_loss = -torch.mean(
            torch.max(expert_class_correlation, dim=1)[0]  # Take the max correlation for each expert
        )
        
        return specialization_loss

def train_epoch(model, loader, optimizer, scheduler, num_classes, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for inputs, targets in tqdm(loader, desc='Training'):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs, aux_losses = model(inputs, targets, num_classes, return_losses=True)
        
        # Main classification loss
        task_loss = F.cross_entropy(outputs, targets)
        
        # Combine all losses
        loss = task_loss + aux_losses['balance_loss'] + aux_losses['specialization_loss']
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        torch.cuda.empty_cache()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(targets.cpu().numpy())
    
    metrics = calculate_metrics(all_labels, all_preds)
    metrics['loss'] = total_loss / len(loader)
    metrics['accuracy'] = 100. * correct / total
    return metrics

def evaluate(model, loader, device, num_classes):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc='Evaluating'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    
    metrics = calculate_metrics(all_labels, all_preds)
    metrics['accuracy'] = 100. * correct / total
    return metrics

def calculate_metrics(labels, preds):
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    return {
        'accuracy': accuracy * 100,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def plot_confusion_matrix(labels, preds, classes, save_path='confusion_matrix.csv'):
    cm = confusion_matrix(labels, preds)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    
    # Save to CSV
    cm_df.to_csv(save_path)
    
    # Plotting
    plt.figure(figsize=(10,7))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig(save_path.replace('.csv', '.png'))
    plt.close()
    return cm_df

class MRI_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_mapping = {'pituitary': 0, 'notumor': 1, 'meningioma': 2, 'glioma': 3}
        self._load_dataset()

    def _load_dataset(self):
        for class_name, label in self.class_mapping.items():
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = torchvision.io.read_image(img_path, torchvision.io.ImageReadMode.GRAY)
        # Convert grayscale to RGB by duplicating channels
        image = image.repeat(3, 1, 1)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def main():
    set_seed(42)
    batch_size = 16  # Adjust based on your GPU memory
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data augmentation and normalization for training
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Only normalization for testing
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Data loaders
    train_dataset = MRI_Dataset(root_dir=R'C:\Users\Precision\Downloads\MRIs\Training', transform=transform_train)
    test_dataset = MRI_Dataset(root_dir=R'C:\Users\Precision\Downloads\MRIs\Testing', transform=transform_test)
    
    trainloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True
    )
    
    testloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=True
    )
    
    # Create model
    model = MoE(
        num_experts=2,
        expert_hidden_dim=128,
        temp=2.0,
        num_classes=4
    ).to(device)
    
    # Separate learning rates for different components
    params = [
        {'params': model.backbone.parameters(), 'lr': 1e-4},  # Lower LR for pretrained
        {'params': model.gate.parameters(), 'lr': 3e-4},      # Medium for gate
        {'params': model.experts.parameters(), 'lr': 3e-4},   # Medium for experts
        {'params': model.classifier.parameters(), 'lr': 3e-4} # Medium for classifier
    ]
    
    optimizer = torch.optim.AdamW(params, weight_decay=0.01)
    
    # Calculate exact number of steps
    total_epochs = 100  # Adjust based on your dataset size and resources
    total_steps = total_epochs * len(trainloader)  
    
    # OneCycle scheduler with exact steps
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[1e-4, 2e-4, 2e-4, 2e-4] ,  # Corresponding to param groups
        total_steps=total_steps,
        pct_start=0.2,
        anneal_strategy='cos',
        final_div_factor=1e4
    )
    
    # Training loop
    best_accuracy = 0
    num_classes = 4
    for epoch in range(total_epochs):  
        print(f'\nEpoch: {epoch+1}')
        train_metrics = train_epoch(
            model, trainloader, optimizer, scheduler, num_classes, device
        )
        print(f'Train Loss: {train_metrics['loss']:.3f} | Train Accuracy: {train_metrics['accuracy']:.3f}%')
        print(f'Train Precision: {train_metrics['precision']:.3f} | Train Recall: {train_metrics['recall']:.3f} | Train F1-Score: {train_metrics['f1_score']:.3f}')
        
        if epoch % 5 == 0:  
            test_metrics = evaluate(model, testloader, device, num_classes)
            print(f'Test Accuracy: {test_metrics['accuracy']:.3f}%')
            print(f'Test Precision: {test_metrics['precision']:.3f} | Test Recall: {test_metrics['recall']:.3f} | Test F1-Score: {test_metrics['f1_score']:.3f}')
            
            # Plot and save confusion matrix
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for inputs, targets in tqdm(testloader, desc='Evaluating for Confusion Matrix'):
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(targets.numpy())
            
            classes = list(train_dataset.class_mapping.keys())
            cm_df = plot_confusion_matrix(all_labels, all_preds, classes, save_path='confusion_matrix.csv')
            
            if test_metrics['accuracy'] > best_accuracy:
                best_accuracy = test_metrics['accuracy']
                torch.save(model.state_dict(), 'best_model.pth')
    
    print(f'Best Test Accuracy: {best_accuracy:.3f}%')


if __name__ == '__main__':
    main()
    