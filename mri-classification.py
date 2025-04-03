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
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import torchvision.transforms.functional as TF

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
class AutomaticBrightnessAndContrast(torch.nn.Module):
    """Transform to automatically adjust brightness and contrast of images."""
    
    def __init__(self, clip_hist_percent=1):
        super().__init__()
        self.clip_hist_percent = clip_hist_percent
    
    def forward(self, image_tensor):
        """
        Apply automatic brightness and contrast adjustment to an image tensor.
        
        Args:
            image_tensor (torch.Tensor): Image tensor of shape [C, H, W] in [0,1] range
            
        Returns:
            torch.Tensor: Adjusted image tensor in same range as input
        """
        
        image = image_tensor.clone()
        
        # Check if image is in [0,1] range (standard for ToTensor output)
        is_normalized = image.max() <= 1.0
        
        # Scale to [0,255] range for histogram calculation
        if is_normalized:
            image_for_hist = image * 255.0
        else:
            image_for_hist = image
        
        # Convert image tensor to grayscale if it's RGB
        if image_for_hist.shape[0] == 3:  # If RGB (C, H, W format)
            # Convert to grayscale using RGB weights
            gray = 0.299 * image_for_hist[0] + 0.587 * image_for_hist[1] + 0.114 * image_for_hist[2]
        else:
            gray = image_for_hist.squeeze()  # If already grayscale
        
        # Calculate histogram
        hist = torch.histc(gray.flatten(), bins=256, min=0, max=255)
        hist_size = len(hist)
        
        # Calculate cumulative distribution
        accumulator = torch.cumsum(hist, dim=0)
        
        # Locate points to clip
        maximum = accumulator[-1].item()
        clip_value = self.clip_hist_percent * (maximum / 100.0) / 2.0
        
        # Locate left cut
        minimum_gray = 0
        while minimum_gray < hist_size and accumulator[minimum_gray].item() < clip_value:
            minimum_gray += 1
        
        # Locate right cut
        maximum_gray = hist_size - 1
        while maximum_gray >= 0 and accumulator[maximum_gray].item() >= (maximum - clip_value):
            maximum_gray -= 1
        
        # Prevent division by zero
        if maximum_gray <= minimum_gray:
            # No adjustment needed or possible
            return image_tensor
        
        # Calculate alpha and beta values
        alpha = 255.0 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha
        
        # Apply contrast stretching based on original image range
        if is_normalized:
            # For [0,1] range images (from ToTensor)
            alpha_norm = alpha / 255.0
            beta_norm = beta / 255.0
            auto_result = image * alpha_norm + beta_norm
            # Clamp values to [0, 1] range
            auto_result = torch.clamp(auto_result, 0, 1)
        else:
            # For [0,255] range images
            auto_result = image * alpha + beta
            # Clamp values to [0, 255] range
            auto_result = torch.clamp(auto_result, 0, 255)
        
        return auto_result

class IntensityClamp(torch.nn.Module):
    """Clamp intensity values to handle outliers in MRI images"""
    def __init__(self, percentile_low=1, percentile_high=99):
        super().__init__()
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        
    def forward(self, x):
        # Process first channel for MRI (as they're duplicated)
        if x.shape[0] == 3 and torch.all(x[0] == x[1]) and torch.all(x[1] == x[2]):
            x_channel = x[0]
            values = x_channel[x_channel > 0]  # Consider only non-zero values
            if len(values) > 0:
                p_low = torch.quantile(values, self.percentile_low/100)
                p_high = torch.quantile(values, self.percentile_high/100)
                
                # Apply to all channels (they're identical)
                for c in range(x.shape[0]):
                    x[c] = torch.clamp(x[c], p_low, p_high)
        return x
class VisualizationTracker:
    """Tracks and visualizes training metrics, confusion matrices, and expert activations."""
    
    def __init__(self, class_names, num_experts=2):
        self.class_names = class_names
        self.num_experts = num_experts
        self.train_losses = []
        self.train_accs = []
        self.test_losses = []
        self.test_accs = []
        self.epochs = []
        self.test_epochs = []  # Separate tracking for test epochs
        self.expert_activations = []
        
        # Create output directory
        os.makedirs('visualizations', exist_ok=True)
    
    def update_metrics(self, epoch, train_metrics, test_metrics=None):
        """Update training and testing metrics."""
        self.epochs.append(epoch)
        self.train_losses.append(train_metrics['loss'])
        self.train_accs.append(train_metrics['accuracy'])
        
        if test_metrics is not None:
            self.test_epochs.append(epoch)  # Track which epochs have test metrics
            self.test_losses.append(test_metrics.get('loss', 0))
            self.test_accs.append(test_metrics['accuracy'])
            
        # Plot updated metrics
        self.plot_metrics()
    
    def plot_metrics(self):
        """Plot training and testing metrics."""
        plt.figure(figsize=(12, 5))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.epochs, self.train_losses, 'b-', label='Training Loss')
        if len(self.test_losses) > 0:
            plt.plot(self.test_epochs, self.test_losses, 'r-', label='Testing Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        current_epoch = self.epochs[-1]
        plt.title(f'Loss Curves (Epoch {current_epoch}, Train: {self.train_losses[-1]:.4f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(self.epochs, self.train_accs, 'b-', label='Training Accuracy')
        if len(self.test_accs) > 0:
            plt.plot(self.test_epochs, self.test_accs, 'r-', label='Testing Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title(f'Accuracy Curves (Epoch {current_epoch}, Train: {self.train_accs[-1]:.2f}%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualizations/train_test_metrics.png')
        plt.close()
    
    def plot_confusion_matrix(self, labels, preds, epoch):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(labels, preds)
        cm_df = pd.DataFrame(cm, index=self.class_names, columns=self.class_names)
        
        # Save to CSV
        cm_df.to_csv(f'visualizations/confusion_matrix.csv')
        
        # Plotting with epoch and accuracy info
        accuracy = np.sum(np.diag(cm)) / np.sum(cm) * 100
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title(f'Confusion Matrix - Epoch {epoch} (Accuracy: {accuracy:.2f}%)')
        plt.savefig(f'visualizations/confusion_matrix.png')
        plt.close()
        
        return cm_df
    
    def record_expert_activations(self, model, dataloader, device, epoch):
        """Record and visualize expert activations across the dataset."""
        model.eval()
        expert_weights_by_class = {class_idx: [] for class_idx in range(len(self.class_names))}
        
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, desc='Recording expert activations'):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Extract backbone features
                features = model.backbone(inputs)[0]
                
                # Get expert weights
                expert_weights = model.gate(features, False)  # False for evaluation mode
                
                # Record weights by class
                for i, target in enumerate(targets):
                    class_idx = target.item()
                    expert_weights_by_class[class_idx].append(expert_weights[i].cpu().numpy())
        
        # Average expert weights per class
        avg_expert_weights = np.zeros((len(self.class_names), self.num_experts))
        for class_idx, weights in expert_weights_by_class.items():
            if weights:  # Check if the list is not empty
                avg_expert_weights[class_idx] = np.mean(weights, axis=0)
        
        # Store for tracking over time
        self.expert_activations.append((epoch, avg_expert_weights))
        
        # Plot expert activation heatmap
        self.plot_expert_activations(epoch, avg_expert_weights)
    
    def plot_expert_activations(self, epoch, avg_expert_weights):
        """Plot expert activation heatmap."""
        plt.figure(figsize=(8, 6))
        
        expert_names = [f'Expert {i+1}' for i in range(self.num_experts)]
        df = pd.DataFrame(avg_expert_weights, index=self.class_names, columns=expert_names)
        
        sns.heatmap(df, annot=True, fmt='.2f', cmap='viridis')
        plt.title(f'Expert Activation by Class - Epoch {epoch}')
        plt.ylabel('Class')
        plt.xlabel('Expert')
        plt.tight_layout()
        plt.savefig(f'visualizations/expert_activations.png')
        plt.close()

def generate_activation_maps(model, sample_images, device, class_names, mean,std, save_dir='visualizations'):
    """Generate activation maps showing what each expert focuses on for MRI images."""
    os.makedirs(save_dir, exist_ok=True)
    
    
    model.eval()
    with torch.no_grad():
        for idx, (image, label) in enumerate(sample_images):
            
            # For MRI images: Since all 3 channels are identical (replicated grayscale),
            # we can just take one channel and denormalize it
            img_np = image[0].cpu().numpy()  # Take first channel
            
            # Undo the normalization for this channel
            denorm_img = img_np * std[0] + mean[0]
            
            # Ensure values are in valid range for display
            denorm_img = np.clip(denorm_img, 0, 1)
            
            # Ensure image is on the right device and has batch dimension for model
            model_input = image.unsqueeze(0).to(device)
            
            # Get backbone features
            features = model.backbone(model_input)[0]
            
            # Get expert weights
            expert_weights = model.gate(features, False)
            expert_weights = expert_weights.cpu().numpy()[0]
            
            # Process through experts to get attention
            expert_outputs = []
            for i, expert in enumerate(model.experts):
                # Forward through expert
                expert_output = expert(features)
                
                # For visualization: get average activation across channels
                activation = expert_output.mean(dim=1).cpu().numpy()[0]
                
                # Normalize for visualization
                activation = (activation - activation.min()) / (activation.max() - activation.min() + 1e-8)
                
                expert_outputs.append(activation)
            
            # Plot
            class_name = class_names[label]
            plt.figure(figsize=(12, 4))
            
            # Original MRI image - display as grayscale
            plt.subplot(1, 3, 1)
            plt.imshow(denorm_img, cmap='gray')
            plt.title(f'Original MRI: {class_name}')
            plt.axis('off')
            
            # Expert activations
            for i in range(min(2, len(expert_outputs))):
                plt.subplot(1, 3, i+2)
                plt.imshow(expert_outputs[i], cmap='viridis')
                plt.title(f'Expert {i+1} (Weight: {expert_weights[i]:.2f})')
                plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/activation_map_sample_{idx}_{class_name}.png')
            plt.close()


class Expert(nn.Module):
    """Enhanced expert with residual connections and Swish activation"""
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
        if in_dim != hidden_dim else nn.Identity()
        )
        
    def forward(self, x):
        identity = self.skip(x)
        out = F.silu(self.bn1(self.conv1(x)))  # Swish activation
        out = self.bn2(self.conv2(out))
        return F.silu(out + identity)  # Swish activation

class Gate(nn.Module):
    """Enhanced gating network with attention and regularization"""
    def __init__(self, in_dim, num_experts, temperature=0.1):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1)*temperature)
        
        # Spatial attention
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 4, 1),
            nn.BatchNorm2d(in_dim // 4),
            nn.SiLU(),  # Swish activation
            nn.Conv2d(in_dim // 4, 1, 1),
            nn.Sigmoid()
        )
        
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Expert selection
        self.expert_gate = nn.Sequential(
            nn.Linear(in_dim * 2, in_dim // 2),
            nn.SiLU(),  # Swish activation
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
    def __init__(self, num_experts=4, expert_hidden_dim=256, temp=0.1, num_classes=4):
        super().__init__()
        # Backbone with ImageNet normalization
        self.backbone = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=True,
            features_only=True,
            out_indices=(3,)  # Higher-level features
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
            nn.SiLU(),  # Swish activation
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(self.feature_dim // 2, num_classes)
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
            # Auxiliary losses
            balance_loss = self.calculate_balance_loss(expert_weights)
            specialization_loss = self.calculate_specialization_loss(expert_weights, labels, n_classes)
            
            losses = {
                'balance_loss': balance_loss * 0.5,
                'specialization_loss': specialization_loss * 0.5
            }
            return output, losses
            
        return output
    
    def calculate_balance_loss(self, expert_weights):
        num_experts = expert_weights.size(1)
        ideal_usage = torch.ones(num_experts, device=expert_weights.device) / num_experts
        expert_usage = expert_weights.mean(0)
        balance_loss = torch.sum((expert_usage - ideal_usage)**2)  # L2 loss
        return balance_loss
        
    def calculate_specialization_loss(self, expert_weights, labels, num_classes):
        batch_size = expert_weights.size(0)
        labels_onehot = F.one_hot(labels, num_classes).float()
        expert_class_correlation = torch.matmul(expert_weights.t(), labels_onehot) / batch_size
        # Encourage max correlation with margin
        max_values = torch.max(expert_class_correlation, dim=1)[0]
        second_max = torch.topk(expert_class_correlation, 2, dim=1).values[:, 1]
        margin_loss = F.relu(second_max - max_values + 0.1).mean()  # Margin of 0.1
        return margin_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, reduction='mean', weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight  # Class weights

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, 
                                 reduction='none', 
                                 weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def train_epoch(model, loader, optimizer, scheduler, num_classes,class_weights, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    if class_weights is not None:
        weights = torch.FloatTensor(class_weights).to(device)
        main_loss = FocalLoss(weight=weights)
    else:
        main_loss = FocalLoss()
        
    for inputs, targets in tqdm(loader, desc='Training'):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs, aux_losses = model(inputs, targets, num_classes, return_losses=True)
        
        # Focal loss with label smoothing - Good for imbalanced classes.
        task_loss = main_loss(outputs, targets)
        
        # Combine losses
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
    total_loss = 0
    all_preds = []
    all_labels = []
    focal_loss = FocalLoss()
    
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc='Evaluating'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Calculate loss for tracking
            loss = focal_loss(outputs, targets)
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    
    metrics = calculate_metrics(all_labels, all_preds)
    metrics['accuracy'] = 100. * correct / total
    # Add loss to metrics
    metrics['loss'] = total_loss / len(loader)
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

def plot_confusion_matrix(labels, preds, classes, save_path='visualizations/confusion_matrix.csv'):
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

def calculate_normalization_params(dataset, batch_size=32):
    """Calculate the mean and std of the dataset for normalization"""
    loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        num_workers=4,
        shuffle=False
    )
    
    # For MRI images converted to 3-channel (but identical channels)
    mean = 0.0
    std = 0.0
    total_samples = 0
    
    for images, _ in tqdm(loader, desc="Calculating normalization parameters"):
        batch_samples = images.size(0)
        # Since channels are identical for MRI (replicated grayscale), just use one channel
        images = images[:, 0, :, :].view(batch_samples, -1)
        mean += images.mean(1).sum().item()
        std += images.std(1).sum().item()
        total_samples += batch_samples
            
    mean /= total_samples
    std /= total_samples
    
    # For 3-channel representation
    return [mean, mean, mean], [std, std, std]

def calculate_class_weights(dataset):
    """Calculate class weights inversely proportional to class frequencies"""
    class_counts = np.zeros(len(dataset.class_mapping))
    for _, label in dataset:
        class_counts[label] += 1
    
    # Handle potential zero counts
    class_counts = np.maximum(class_counts, 1)
    
    # Inverse frequency weighting
    weights = 1.0 / class_counts
    
    # Normalize weights to sum to number of classes
    weights = weights * len(dataset.class_mapping) / weights.sum()
    
    print(f"Class counts: {class_counts}")
    print(f"Class weights: {weights}")
    
    return weights

def get_class_samples(dataset, num_per_class=1):
    """Select representative samples from each class in the dataset"""
    class_samples = {}
    class_counts = {}
    
    # Initialize counters for each class
    for class_name in dataset.class_mapping.keys():
        class_samples[class_name] = []
        class_counts[class_name] = 0
    
    # Collect samples until we have enough from each class
    for idx in range(len(dataset)):
        image, label = dataset[idx]
        
        # Get class name from label
        class_name = list(dataset.class_mapping.keys())[list(dataset.class_mapping.values()).index(label)]
        
        # If we need more samples from this class
        if class_counts[class_name] < num_per_class:
            class_samples[class_name].append((image, label))
            class_counts[class_name] += 1
        
        # Exit if we have enough samples from all classes
        if all(count >= num_per_class for count in class_counts.values()):
            break
    
    # Flatten the samples list
    samples = []
    for class_name in class_samples:
        samples.extend(class_samples[class_name])
    
    return samples

def main():
    set_seed(42)
    batch_size = 32  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create a dataset without normalization first
    clip_hist_percent = 0.5
    percentile_low = 1.0
    percentile_high = 100.0 - percentile_low
    temp_dataset = MRI_Dataset(
        root_dir=R'C:\Users\Precision\Onus\Data\Brain-MRIs\Training', 
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            #AutomaticBrightnessAndContrast(clip_hist_percent=clip_hist_percent),
            #IntensityClamp(percentile_low=percentile_low, percentile_high=percentile_high),
        ])
    )

    mri_mean, mri_std = calculate_normalization_params(temp_dataset)
    print(f"MRI Dataset - Mean: {mri_mean[0]:.4f}, Std: {mri_std[0]:.4f}")
    
    class_weights = calculate_class_weights(temp_dataset)
    # Data augmentation and normalization for training
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #AutomaticBrightnessAndContrast(clip_hist_percent=clip_hist_percent),
        #IntensityClamp(percentile_low=percentile_low, percentile_high=percentile_high),
        transforms.Normalize(mean=mri_mean, std=mri_std)
    ])
    # No augumentation, only Normalize for testing
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        #AutomaticBrightnessAndContrast(clip_hist_percent=clip_hist_percent),
        #IntensityClamp(percentile_low=percentile_low, percentile_high=percentile_high),
        transforms.Normalize(mean=mri_mean, std=mri_std)
    ])

    
    train_dataset = MRI_Dataset(root_dir=R'C:\Users\Precision\Onus\Data\Brain-MRIs\Training', transform=transform_train)
    test_dataset = MRI_Dataset(root_dir=R'C:\Users\Precision\Onus\Data\Brain-MRIs\Testing', transform=transform_test)
    
    trainloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True
    )
    
    testloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=True
    )
    
    num_experts = 4
    num_classes = 4
    model = MoE(
        num_experts=num_experts,
        expert_hidden_dim=128,
        temp=2.0,
        num_classes=num_classes
    ).to(device)
    
    # Separate learning rates for different components
    params = [
        {'params': model.backbone.parameters(), 'lr': 1e-4},  # Lower LR for pretrained
        {'params': model.gate.parameters(), 'lr': 3e-4},      # Medium for gate
        {'params': model.experts.parameters(), 'lr': 3e-4},   # Medium for experts
        {'params': model.classifier.parameters(), 'lr': 3e-4} # Medium for classifier
    ]
    
    optimizer = torch.optim.AdamW(params, weight_decay=0.01)
    
    total_epochs = 100  
    total_steps = total_epochs * len(trainloader)  
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[1e-4, 2e-4, 2e-4, 2e-4],  # Corresponding to param groups
        total_steps=total_steps,
        pct_start=0.2,
        anneal_strategy='cos',
        final_div_factor=1e4
    )
    
    tracker = VisualizationTracker(
        class_names=list(train_dataset.class_mapping.keys()), 
        num_experts=num_experts
    )
    
    os.makedirs('visualizations', exist_ok=True)
    best_accuracy = 0
    
    # Select a few sample images for activation maps
    sample_images = get_class_samples(test_dataset)
    print(f"Selected {len(sample_images)} samples for visualization, one from each class")
    
    for epoch in range(total_epochs):
        print(f'\nEpoch: {epoch+1}/{total_epochs}')
        
        # Train for one epoch
        train_metrics = train_epoch(
            model, trainloader, optimizer, scheduler, num_classes,class_weights, device
        )
        print(f'Train Loss: {train_metrics["loss"]:.3f} | Train Accuracy: {train_metrics["accuracy"]:.3f}%')
        print(f'Train Precision: {train_metrics["precision"]:.3f} | Train Recall: {train_metrics["recall"]:.3f} | Train F1-Score: {train_metrics["f1_score"]:.3f}')
        
        # Update metrics tracking (without test metrics)
        tracker.update_metrics(epoch+1, train_metrics)
        
        # Every 5 epochs, run evaluation
        if epoch % 5 == 0 or epoch == total_epochs - 1:  
            test_metrics = evaluate(model, testloader, device, num_classes)
            print(f'Test Accuracy: {test_metrics["accuracy"]:.3f}%')
            print(f'Test Precision: {test_metrics["precision"]:.3f} | Test Recall: {test_metrics["recall"]:.3f} | Test F1-Score: {test_metrics["f1_score"]:.3f}')
            
            tracker.update_metrics(epoch+1, train_metrics, test_metrics)
            
            # Get predictions for confusion matrix
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for inputs, targets in tqdm(testloader, desc='Generating confusion matrix'):
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(targets.numpy())
            
            tracker.plot_confusion_matrix(all_labels, all_preds, epoch+1)            
            tracker.record_expert_activations(model, testloader, device, epoch+1)
            
            # Generate activation maps for sample images
            generate_activation_maps(
                model, 
                sample_images, 
                device, 
                list(test_dataset.class_mapping.keys()),
                mri_mean,
                mri_std,
                save_dir=f'visualizations'
            )
            
            if test_metrics['accuracy'] > best_accuracy:
                best_accuracy = test_metrics['accuracy']
                torch.save(model.state_dict(), 'best_model.pth')
                print(f'New best model saved with accuracy: {best_accuracy:.3f}%')
            
    print(f'Training completed. Best Test Accuracy: {best_accuracy:.3f}%')


if __name__ == '__main__':
    main()
    