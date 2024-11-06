import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.io import loadmat
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import h5py
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Counter Network ###

class CoprimeDilationConvGroup(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilations):
        super(CoprimeDilationConvGroup, self).__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=((kernel_size - 1) * d) // 2, dilation=d),
                nn.ReLU(inplace=True)  
            )
            for d in dilations
        ])

    def forward(self, x):
        conv_outputs = [conv(x) for conv in self.convs]
        out = torch.stack(conv_outputs, dim=0).sum(dim=0)
        return out

class CountingNet(nn.Module):
    def __init__(self, in_channels):
        super(CountingNet, self).__init__()
        self.branch_3x3 = CoprimeDilationConvGroup(in_channels, in_channels, kernel_size=3, dilations=[1, 2, 3])
        self.branch_5x5 = CoprimeDilationConvGroup(in_channels, in_channels, kernel_size=5, dilations=[1, 2])
        
        self.final_conv = nn.Conv2d(in_channels * 2, 1, kernel_size=1)

    def forward(self, x):
        branch_3x3_out = self.branch_3x3(x)
        branch_5x5_out = self.branch_5x5(x)
        combined = torch.cat([branch_3x3_out, branch_5x5_out], dim=1)
        density_map = self.final_conv(combined)
        return density_map

class CounterNetwork(nn.Module):
    def __init__(self):
        super(CounterNetwork, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.front_end = nn.Sequential(*list(vgg16.features)[:10])
        
        self.counting_net = CountingNet(in_channels=128)

    def forward(self, x):
        features = self.front_end(x)
        density_map = self.counting_net(features)
        return density_map

### Data Loading ###

def load_image(image_path):
    image = Image.open(image_path).convert('L')
    transform = transforms.ToTensor()
    image = transform(image).unsqueeze(0).to(device)
    image = image.repeat(1, 3, 1, 1)
    return image

def load_dotmap(dotmap_path):
    with h5py.File(dotmap_path, 'r') as f:
        dot_map = torch.tensor(f['density'][:]).float()
    if dot_map.dim() == 2:
        dot_map = dot_map.unsqueeze(0).unsqueeze(0)
    downscale_factor = 4
    dot_map_downscaled = F.avg_pool2d(dot_map, kernel_size=downscale_factor, stride=downscale_factor) * (downscale_factor ** 2)
    return dot_map_downscaled.to(device)

### Training Function ###

def train(train_images, train_dots, num_epochs=40, alpha=0.5, lr=1e-5): ## num_epochs can be adjusted (for best results : 200-300)
    counter_network = CounterNetwork().to(device)

    optimizer_counter = optim.Adam(counter_network.parameters(), lr=lr, weight_decay=1e-4)
    mse_loss = nn.MSELoss()

    for epoch in range(num_epochs):
        counter_network.train()
        epoch_loss_refine = 0.0
        epoch_loss_predict = 0.0
        for image_path, dotmap_path in tqdm(zip(train_images, train_dots), total=len(train_images)):
            image = load_image(image_path)
            dot_map = load_dotmap(dotmap_path)
            predicted_density_map = counter_network(image)
            with torch.autograd.set_detect_anomaly(True):
                
                loss_predict = mse_loss(predicted_density_map, dot_map)
                
                optimizer_counter.zero_grad()
                loss_predict.backward()
                optimizer_counter.step()

            epoch_loss_predict = epoch_loss_predict + loss_predict.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Predict Loss: {epoch_loss_predict:.4f}')
    
    return counter_network



### Testing ###

def test(counter_network, test_images, test_dots):
    counter_network.eval()
    mae, rmse = 0.0, 0.0

    with torch.no_grad():
        for image_path, dotmap_path in zip(test_images, test_dots):
            image = load_image(image_path)
            gt_density_map = load_dotmap(dotmap_path)
            gt_count = gt_density_map.sum().item()

            predicted_density_map = counter_network(image)
            predicted_count = predicted_density_map.sum().item()

            print(f"GT Count: {gt_count:.2f}, Predicted Count: {predicted_count:.2f}")

            mae += abs(predicted_count - gt_count)
            rmse += (predicted_count - gt_count) ** 2
            
            # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))
            # ax1.imshow(gt_density_map.squeeze().cpu().numpy(), cmap='jet', aspect='auto')
            # ax1.set_title(f"Ground Truth Density Map\nCount: {gt_count:.2f}")
            # ax1.axis('off')
            # plt.colorbar(ax1.imshow(gt_density_map.squeeze().cpu().numpy(), cmap='jet'), ax=ax1, fraction=0.046, pad=0.04)

            # ax2.imshow(predicted_density_map.squeeze().cpu().numpy(), cmap='jet', aspect='auto')
            # ax2.set_title(f"Predicted Density Map\nCount: {predicted_count:.2f}")
            # ax2.axis('off')
            # plt.colorbar(ax2.imshow(predicted_density_map.squeeze().cpu().numpy(), cmap='jet'), ax=ax2, fraction=0.046, pad=0.04)

            # plt.show()

    mae /= len(test_images)
    rmse = (rmse / len(test_images)) ** 0.5
    print(f'\nOverall MAE: {mae:.2f}, RMSE: {rmse:.2f}')

# Example Usage
train_images = [f"/kaggle/input/shanghaitech-with-people-density-map/ShanghaiTech/part_B/train_data/images/IMG_{i}.jpg" for i in range(1, 301)]
train_dots = [f"/kaggle/input/shanghaitech-with-people-density-map/ShanghaiTech/part_B/train_data/ground-truth-h5/IMG_{i}.h5" for i in range(1, 301)]
test_images = [f"/kaggle/input/shanghaitech-with-people-density-map/ShanghaiTech/part_B/test_data/images/IMG_{i}.jpg" for i in range(1, 183)]
test_dots = [f"/kaggle/input/shanghaitech-with-people-density-map/ShanghaiTech/part_B/test_data/ground-truth-h5/IMG_{i}.h5" for i in range(1, 183)]

counter_network = train(train_images, train_dots)
test(counter_network, test_images, test_dots)
