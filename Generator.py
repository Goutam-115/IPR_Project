import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import h5py 
from glob import glob
import scipy.io as sio
from scipy.ndimage import gaussian_filter
import scipy
from scipy.io import loadmat
import cv2
import torchvision.transforms.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models

## check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class CrowdDataset(Dataset):
    def __init__(self, image_dir, ground_truth_dir, output_size=(512, 512)):
        self.image_dir = image_dir
        self.ground_truth_dir = ground_truth_dir
        self.output_size = output_size
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.output_size), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)  

        gt_path = os.path.join(self.ground_truth_dir, self.image_files[idx].replace('.jpg', '.h5'))
        with h5py.File(gt_path, 'r') as gt_file:
            gt_density_map = np.asarray(gt_file['density'])

        gt_density_map = cv2.resize(gt_density_map, self.output_size) 
        gt_density_map = torch.tensor(gt_density_map, dtype=torch.float32)

        gt_density_map = torch.unsqueeze(gt_density_map, 0)

        return image, gt_density_map

class CSRNet(nn.Module):
    def __init__(self, local_vgg_path=None):
        
        super(CSRNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        
        if local_vgg_path is not None:
            # Load VGG-16 weights from a local path (e.g., if internet is not available)
            print(f"Loading VGG-16 weights from local path: {local_vgg_path}")
            mod = models.vgg16()  # Initialize VGG-16 without pre-trained weights
            mod.load_state_dict(torch.load(local_vgg_path))  # Load the local VGG-16 weights
        else:
            print("Loading pretrained VGG-16 weights from the internet")
            mod = models.vgg16(pretrained=True)

        self._initialize_weights()  # Initialize backend weights
        self.initialize_weights(mod)  # Initialize frontend weights with VGG-16

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def initialize_weights(self, vgg16):
        frontend_state_dict_items = list(self.frontend.state_dict().items())
        vgg16_state_dict_items = list(vgg16.state_dict().items())
        
        for i in range(len(frontend_state_dict_items)):
            frontend_state_dict_items[i][1].data[:] = vgg16_state_dict_items[i][1].data[:]

def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    d_rate = 2 if dilation else 1
    layers = []
    
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class Refiner(nn.Module):
    def __init__(self):
        super(Refiner, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 512, kernel_size=3, padding=1),  # Input channel = 1
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)   
        )

    def forward(self, density_map):
        refined_density_map = self.layers(density_map)
        return refined_density_map

def joint_loss_function(counter_output, refined_density, gt_density_map, alpha=1.0):
    upsampled_counter_output = nn.functional.interpolate(counter_output, size=gt_density_map.shape[2:], mode='bilinear', align_corners=False)
    counting_loss = torch.mean((upsampled_counter_output - refined_density) ** 2)
    refinement_loss = torch.mean((refined_density - gt_density_map) ** 2)
    total_loss = counting_loss + alpha * refinement_loss
    return total_loss

def train_model(counter, refiner, optimizer_counter, optimizer_refiner,scheduler_counter,scheduler_refiner, train_loader, epochs):
    counter.train()
    refiner.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        print(f'\nEpoch [{epoch + 1}/{epochs}] started...')

        for i, (images, gt_density_maps) in enumerate(train_loader):
            images = images.to(device)
            gt_density_maps = gt_density_maps.to(device)

            optimizer_counter.zero_grad()
            optimizer_refiner.zero_grad()

            counter_output = counter(images)
            refined_density = refiner(gt_density_maps)

            loss = joint_loss_function(counter_output, refined_density, gt_density_maps)

            loss.backward()
        
            optimizer_counter.step()
            optimizer_refiner.step()

            running_loss += loss.item()
            torch.cuda.empty_cache()

        avg_epoch_loss = running_loss / len(train_loader)
        scheduler_counter.step(avg_epoch_loss)
        scheduler_refiner.step(avg_epoch_loss)
        print(f'Epoch [{epoch + 1}/{epochs}] completed. Average Loss: {avg_epoch_loss:.7f}', flush = True)
       
        torch.cuda.empty_cache()

def normalize_counter_output(output):
    min_value = torch.min(output)
    max_value = torch.max(output)
    
    if max_value == min_value:
        return torch.zeros_like(output)
    
    normalized_output = (output - min_value) / (max_value - min_value)

    rescaled_output = normalized_output * (max_value - min_value)

    return rescaled_output

def evaluate_model(model, image_dir, ground_truth_dir):
    model.eval() 
    mae = 0
    rmse = 0
    img_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.jpg')]

    for i, img_path in enumerate(img_paths):
        # Load and preprocess the image
        img = 255.0 * F.to_tensor(Image.open(img_paths[i]).convert('RGB'))
    
        # Normalize each channel separately (as per CSRNet)
        img[0, :, :] = img[0, :, :] - 92.8207477031
        img[1, :, :] = img[1, :, :] - 95.2757037428
        img[2, :, :] = img[2, :, :] - 104.877445883

        img = img.cuda()

        gt_file = h5py.File(os.path.join(ground_truth_dir, os.path.basename(img_path).replace('.jpg', '.h5')), 'r')
        groundtruth = np.asarray(gt_file['density'])

        output = model(img.unsqueeze(0))
        normalized_output = normalize_counter_output(output)
        pred_count = torch.sum(normalized_output).item()
        gt_count = np.sum(groundtruth)  # Ground truth count
        mae += abs(abs(pred_count) - gt_count)
        rmse += ((abs(pred_count)-gt_count)**2)

        print(f"Image {i + 1}/{len(img_paths)} - Predicted: {pred_count:.2f}, Ground Truth: {gt_count:.2f}")

    print(f"Final MAE: {mae / len(img_paths)}")
    print(f"Final RMSE: {(rmse / len(img_paths))**0.5}")




###############
###############
## REPLACE THE PATH WITH YOUR DATSET PATH (images AND groundtruth density maps (density maps in .h5 files))

train_image_dir = '/kaggle/input/shanghaitech-with-people-density-map/ShanghaiTech/part_A/train_data/images'
train_gt_dir = '/kaggle/input/shanghaitech-with-people-density-map/ShanghaiTech/part_A/train_data/ground-truth-h5'

test_image_dir = '/kaggle/input/shanghaitech-with-people-density-map/ShanghaiTech/part_A/test_data/images'
test_gt_dir = '/kaggle/input/shanghaitech-with-people-density-map/ShanghaiTech/part_A/test_data/ground-truth-h5'

###############
###############

# Dataset and Dataloaders
train_dataset = CrowdDataset(train_image_dir, train_gt_dir)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)

# Example Usage:
# If internet is unavailable
# you can pass the local path to load them:
vgg16_path = '/kaggle/input/vgg-16/vgg16-397923af.pth' ## replace with the actual path of vgg in your directory

### uncomment this part if internet not available
counter1 = CSRNet(local_vgg_path=vgg16_path).to(device) 

# counter1 = CSRNet().to(device) ####comment if previous line is uncommented 
refiner = Refiner().to(device)

optimizer_counter = torch.optim.SGD(counter1.parameters(), lr=5e-7)
optimizer_refiner = torch.optim.Adam(refiner.parameters(), lr=1e-5)
scheduler_counter = ReduceLROnPlateau(optimizer_counter, mode='min', factor=0.1, patience=5)
scheduler_refiner = ReduceLROnPlateau(optimizer_refiner, mode='min', factor=0.1, patience=5)

train_model(counter1, refiner, optimizer_counter, optimizer_refiner, scheduler_counter,scheduler_refiner, train_loader, epochs=1)

evaluate_model(counter1, test_image_dir, test_gt_dir)

