import os
import time
import sys
import numpy as np
import pandas as pd
import logging
import cv2

import matplotlib.pyplot  as plt

from lpips import LPIPS
from tqdm import tqdm

import torch
import torch.nn            as nn
import torch.nn.functional as F

from torch.nn.utils           import clip_grad_norm_
from torch.utils.data         import Dataset, DataLoader, TensorDataset
from torch.utils.tensorboard  import SummaryWriter
from torch.optim              import Adam, SGD
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp           import GradScaler

from torchvision            import transforms
from torchvision.transforms import Grayscale

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

from PIL import Image

from sklearn.model_selection import train_test_split
from skimage.morphology import disk
from math import log10

# Add project path to sys.path
sys.path.append('./')

# Import image processing functions
import core.networks as dh_networks

# Set logging level
logging.getLogger('lpips').setLevel(logging.WARNING)

# Set constants
PROJECT_PATH            = './'
TEST_DATASET_PATH       = PROJECT_PATH + 'data/real_images'
TRAINING_DATASET_PATH   = PROJECT_PATH + 'data/small_training_dataset'
X_TRAINING_DATASET_PATH = TRAINING_DATASET_PATH + '/X'
Y_TRAINING_DATASET_PATH = TRAINING_DATASET_PATH + '/Y'
MODEL_PATH              = PROJECT_PATH + 'models/'
IMAGE_EXTENSIONS        = [".png", ".jpg", ".jpeg", ".tif'", ".tiff", ".bmp"]

# Set hyperparameters
generator_lr        = 5e-5
discriminator_lr    = 5e-5
batch_size          = 32
num_epochs          = 100
checkpoint_interval = 10
max_grad_norm       = 1.0 # Define the maximum gradient norm for clipping

########################################################################################
#Adjustments:
# If preserving fine details and textures is more important than exact structural matching, 
# consider slightly decreasing alpha (MSE Loss) and increasing beta (L1 Loss).
#
# If depth and geometric details are crucial, 
# consider slightly increasing epsilon (Depth Consistency Loss) and zeta (Geometric Consistency Loss).
#
# If the images are too smooth or lacking in detail, reduce eta (TV Loss).
#
# Adjust gamma (SSIM) based on the perceptual quality of the outputs.
# If the outputs are structurally accurate but lack perceptual quality, consider increasing it.
# 
########################################################################################
initial_weights = {
    'alpha':   0.1,   # Weight for MSE Loss
    'beta':    0.55,   # Weight for Reconstruction Loss (L1)
    'gamma':   0.4,   # Weight for SSIM in Perceptual Loss
    'delta':   0.1,   # Weight for Adversarial Loss
    'epsilon': 0.2,   # Weight for Depth Consistency Loss
    'zeta':    0.25,   # Weight for Geometric Consistency Loss
    'eta':     0.01   # Weight for TV Loss
}

########################################################################################
# Initialize Optimizers
# RMSProp or Adagrad
########################################################################################
def init_optimizer(generator, discriminator, generator_lr, discriminator_lr):
    # Initialize optimizers
    gen_optim = Adam(generator.parameters(),     lr=generator_lr,     betas=(0.5, 0.999))
    dis_optim = Adam(discriminator.parameters(), lr=discriminator_lr, betas=(0.5, 0.999))

    # gen_optim = SGD(generator.parameters(),     lr=generator_lr)
    # dis_optim = SGD(discriminator.parameters(), lr=discriminator_lr)

    return gen_optim, dis_optim


# Create a SummaryWriter object
writer = SummaryWriter(PROJECT_PATH + '/runs/experiment_1')

# PyTorch version
print(torch.__version__)

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    device = torch.device("cuda")

    print("CUDA device found.")
else:
    device = torch.device("cpu")

    print("No CUDA device found, using CPU.")

# Check if MPS (Multi-Process Service) is available
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")

    print("MPS device found.")
else:
    print("MPS device not found.")


########################################################################################
# Function to get image from paths
########################################################################################
def get_image_paths(directory):
    return [
        os.path.join(directory, fname)

        for fname in sorted(os.listdir(directory))
        
        if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS
    ]

class DisplacementMapDataset(Dataset):
    def __init__(self, intact_image_paths, damaged_image_paths, transform=None):
        """
        Args:
            intact_image_paths  (list): List of paths to intact images
            damaged_image_paths (list): List of paths to damaged images
            transform           (callable, optional): Optional transform to be applied on a sample.
        """
        self.intact_image_paths  = intact_image_paths
        self.damaged_image_paths = damaged_image_paths
        self.transform           = transform
        self.counter             = 0 

    def filter_invalid_images(self, image_paths):
        valid_image_paths = []

        for image_path in image_paths:
            try:
                Image.open(image_path)
                valid_image_paths.append(image_path)
            except IOError:
                print(f"Error opening image file: {image_path}")
        
        return valid_image_paths
  
    def __len__(self):
        return len(self.intact_image_paths)

    def __getitem__(self, idx):
        while True:
            try:
                intact_image_path  = self.intact_image_paths[idx]
                damaged_image_path = self.damaged_image_paths[idx]

                intact_image  = Image.open(intact_image_path)
                damaged_image = Image.open(damaged_image_path)

                if self.transform:
                    intact_image  = self.transform(intact_image)
                    damaged_image = self.transform(damaged_image)

                # Increment the counter and log every 100 pairs
                self.counter += 1

                if self.counter % 100 == 0:
                    print(f"Processed {self.counter} pairs. Current pair: {intact_image_path}, {damaged_image_path}")
                
                return intact_image, damaged_image
            except IOError as e:
                print(f"Error opening image files: {intact_image_path}, {damaged_image_path}. Error: {e}")
                # Increment the index and try the next pair of images
                idx = (idx + 1) % len(self.intact_image_paths)


########################################################################################
# Load the dataset
########################################################################################
def load_dataset():
    # Get images from path
    intact_image_paths  = get_image_paths(X_TRAINING_DATASET_PATH)
    damaged_image_paths = get_image_paths(Y_TRAINING_DATASET_PATH)

    print(f"Number of Paired Inscriptions: {len(intact_image_paths)}")
    print(f"Number of Paired Inscriptions: {len(damaged_image_paths)}")

    #assert len(intact_image_paths) == len(damaged_image_paths), "Number of intact and damaged images must be the same"

    print(f"Number of Paired Inscriptions: {len(intact_image_paths)}")


    # Calculate mean and std for the dataset
    mean = 0.
    std = 0.

    # Common Data Augmentation
    common_transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.Lambda(lambda x: x.convert('L')),
        transforms.ToTensor()
    ])

     # Use a subset to calculate mean and std for efficiency
    subset_intact_image_paths = intact_image_paths[:150]
    subset_damaged_image_paths = damaged_image_paths[:150]
    min_val, max_val = float('inf'), -float('inf')

    for file_path in subset_damaged_image_paths:
        image     = Image.open(file_path)
        tensor    = common_transforms(image)

        min_val = min(min_val, tensor.min().item())
        max_val = max(max_val, tensor.max().item())
    
    print(f"Minimum pixel value in the dataset: {min_val}")
    print(f"Maximum pixel value in the dataset: {max_val}")

    for images, _ in DataLoader(DisplacementMapDataset(subset_intact_image_paths, subset_damaged_image_paths, transform=common_transforms), batch_size=batch_size):
        images       = images.to(device)  # Move images to GPU
        batch_samples = len(images)  # Use len() to get the number of images in the batch
        images       = images.view(batch_samples, images.size(1), -1)
        mean      += images.mean(2).sum(0)
        std        += images.std(2).sum(0)

    mean /= len(intact_image_paths)
    std   /= len(intact_image_paths)

    # Now mean and std should be tensors of size 1
    print(mean.size())  # Should print torch.Size([1])
    print(std.size())  # Should print torch.Size([1])

    # Data Augmentation for the Real Images
    image_transforms = transforms.Compose([
        transforms.RandomResizedCrop(512, scale=(0.8, 1.0)),
        transforms.RandomRotation(10),
        #transforms.RandomCrop(224),
        common_transforms,
        transforms.Normalize(mean=mean.tolist(), std=std.tolist())
    ])

    # Split the data into training and validation sets
    train_intact, val_intact, train_damaged, val_damaged = train_test_split(
        intact_image_paths, damaged_image_paths, test_size=0.2, random_state=42)

    # Create training and validation datasets and dataloaders
    train_dataset = DisplacementMapDataset(train_intact, train_damaged, transform = image_transforms)
    val_dataset   = DisplacementMapDataset(val_intact, val_damaged, transform = image_transforms)

    # Create dataset and dataloader
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_dataloader   = DataLoader(val_dataset, batch_size = batch_size, shuffle=False)

    return train_dataloader, val_dataloader


########################################################################################
# Instantiate the generator and discriminator
########################################################################################
def instantiate_networks():
    # grayscale images, 3 for RGB images
    gen_in_channels  = 1
    # to generate grayscale restored images 
    gen_out_channels = 1 

    # Instantiate the generator with the specified channel configurations
    generator = dh_networks.DHadadGenerator(gen_in_channels, gen_out_channels).to(device)

    # Specify the input channel configurations (it typically takes two inputs)
    # we assume we're providing pairs of images (intact and restored) as input
    disc_in_channels = 1

    discriminator = dh_networks.DHadadDiscriminator(disc_in_channels).to(device)

    return generator, discriminator


########################################################################################
# Mean Squared Error(MSE) Loss
# It ensures the overall structure of the reconstructed image
# is similar to the intact image.
# MSE could encourage blurred details, which can be detrimental for text recovery 
# and for sharp engravings.
########################################################################################
mean_sq_error_loss = nn.MSELoss().to(device)

########################################################################################
# L1 Loss
# It helps in recovering finer details without overly penalizing slight deviations that 
# aren't perceptually significant.
########################################################################################
l1_loss = nn.L1Loss().to(device)

########################################################################################
# Perceptual Loss
# Captures textural and stylistic features.
# luminance, and contrast in an image.
# A higher weight is crucial as it emphasizes on the perceptual similarity, which is key for letter reconstruction.
########################################################################################
class WeightedSSIMLoss(nn.Module):
    def __init__(self, data_range=255, size_average=True, channel=1, weight=1.0):
        super(WeightedSSIMLoss, self).__init__()

        self.ssim   = SSIM(data_range=data_range, size_average=size_average, channel=channel)
        self.weight = weight

    def forward(self, img1, img2):

        ssim_loss          = self.ssim(img1, img2)
        weighted_ssim_loss = self.weight * (1 - ssim_loss)  # Weighted SSIM loss

        return weighted_ssim_loss

weighted_ssim_loss = WeightedSSIMLoss(data_range=255, size_average=True, channel=1, weight=1.0)

########################################################################################
# LPIPS Loss
# LPIPS is a perceptual loss that uses a pretrained VGG19 network to calculate
########################################################################################
lpips_loss = LPIPS(net='vgg').to(device)  # Use the AlexNet layer

########################################################################################
# Adversarial Loss
# TO encourage the generator to create images that are indistinguishable
# from the intact displacement maps.
# Encourages realism in the generated maps.
# Keeping this weight lower ensures that the focus remains on structural and textural accuracy rather than just realism.
########################################################################################
bce_with_logits_loss = nn.BCEWithLogitsLoss().to(device)

def adversarial_loss(discriminator_preds, is_real=True):
    if is_real:
        labels = torch.ones_like(discriminator_preds).to(device)
    else:
        labels = torch.zeros_like(discriminator_preds).to(device)

    return bce_with_logits_loss(discriminator_preds, labels)

########################################################################################
# DepthConsistencyLoss
# This is a robust loss that combines the benefits of L1 and L2 losses.
# It can be particularly useful if there's a lot of noise in the damaged maps.
# Depth data is uncertain in some places
# This loss can help in smoothing the depth map without losing essential details.
########################################################################################
class DepthConsistencyLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(DepthConsistencyLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, generated_depth, target_depth):
        # Assuming generated_depth and target_depth are tensors representing depth maps
        # Charbonnier Loss: sqrt((x - y)^2 + epsilon)
        loss = torch.mean(torch.sqrt((generated_depth - target_depth) ** 2 + self.epsilon))

        return loss

depth_consistency_loss = DepthConsistencyLoss().to(device)

########################################################################################
# Geometric Consistency Loss
# Maintains geometric integrity of depth information.
# This will help in preserving the contours and shapes of letters in the displacement maps.
########################################################################################
class GeometricConsistencyLoss(nn.Module):
    def __init__(self):
        super(GeometricConsistencyLoss, self).__init__()

    def forward(self, predicted_map, target_map):
        # Calculate gradients in x and y direction
        # These gradients represent the change in depth (or displacement) across pixels
        grad_x_pred, grad_y_pred = self.compute_gradients(predicted_map)
        grad_x_target, grad_y_target = self.compute_gradients(target_map)

        # Calculate the loss as the mean squared error between the gradients of the predicted and target maps
        loss_x = F.mse_loss(grad_x_pred, grad_x_target)
        loss_y = F.mse_loss(grad_y_pred, grad_y_target)

        # Combine the losses
        loss = loss_x + loss_y

        return loss

    def compute_gradients(self, map):
        # Function to compute gradients in the x and y direction
        grad_x = map[:, :, :, :-1] - map[:, :, :, 1:]
        grad_y = map[:, :, :-1, :] - map[:, :, 1:, :]

        return grad_x, grad_y

geometric_consistency_loss = GeometricConsistencyLoss().to(device)


########################################################################################
# Total Variation Loss
# Encourages spatial smoothness in the generated images.
# This can help in reducing noise and artifacts in the generated images.
########################################################################################
def tv_loss(img):
    batch_size, _, height, width = img.size()
    
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()

    return (tv_h + tv_w) / (batch_size * height * width)

########################################################################################
# Dynamic Loss Weights
# Adjusting the loss weights for DeepHadad's displacement map inscription restoration
########################################################################################
class DynamicLossWeights:
    def __init__(self, initial_weights, max_weight=1.0, min_weight=0.01, decay_factor=0.9):
        self.weights = initial_weights
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.decay_factor = decay_factor

    def update_weights(self, performance_metrics):
        # Adjust weights based on the specific metric and its improvement
        for metric, info in performance_metrics.items():
            improvement = info['improved']
            magnitude = info['magnitude']

            if metric == 'esi':
                # ESI is a measure of edge similarity.
                # Higher ESI indicates better structural integrity.
                self.adjust_weight_for_metric('zeta', improvement, magnitude) # Geometric Consistency Loss adjustment

                # Adversarial loss might be tuned down if ESI is high, focusing more on realism
                self.adjust_weight_for_metric('delta', improvement, magnitude) # Adversarial Loss adjustment

            if metric == 'psnr':
                # PSNR focuses on reconstruction quality.
                # Improvement in PSNR indicates better structural integrity.
                self.adjust_weight_for_metric('alpha', improvement, magnitude) # MSE Loss adjustment

                # Improvement in PSNR indicates better structural integrity.
                self.adjust_weight_for_metric('beta', improvement, magnitude) # L1 Loss adjustment
            
            if metric == 'ssim':
                # SSIM focuses on perceptual similarity.
                # Improvement in SSIM indicates better structural integrity.
                self.adjust_weight_for_metric('gamma', improvement, magnitude) # SSIM Loss adjustment

        # Normalize weights after adjustment
        #self.normalize_weights()

        return self.weights

    def adjust_weight_for_metric(self, weight_key, improvement, magnitude):
        factor = self.decay_factor if improvement else 1.1 * (1 + magnitude)

        self.weights[weight_key] *= factor

        self.weights[weight_key] = min(max(self.weights[weight_key], self.min_weight), self.max_weight)

    def normalize_weights(self):
        total_weight = sum(self.weights.values())

        for key in self.weights:
            self.weights[key] = (self.weights[key] / total_weight) * self.max_weight

loss_weights = DynamicLossWeights(initial_weights)

########################################################################################
# Combined generator loss function.
########################################################################################
def combined_gen_loss(gen_imgs, real_imgs, discriminator_preds, loss_weights):
    # Mean Squared Error Loss
    mse_loss = mean_sq_error_loss(gen_imgs, real_imgs)

    # Reconstruction Loss (L1)
    recon_loss = l1_loss(gen_imgs, real_imgs)

    # Perceptual Loss
    ssim_loss = weighted_ssim_loss(gen_imgs, real_imgs).to(device)
    #lpips_loss_value = lpips_loss(gen_imgs, real_imgs)

    # Adversarial Loss for the generator
    adv_loss = adversarial_loss(discriminator_preds, is_real=False)

    # Charbonnier Loss (robust L1/L2 loss)
    depth_loss = depth_consistency_loss(gen_imgs, real_imgs)

    # predicted_map and target_map are your generated and ground truth displacement maps, respectively
    geom_loss = geometric_consistency_loss(gen_imgs, real_imgs)

    #TV loss function
    tv_loss_value = tv_loss(gen_imgs)

    combined_loss = torch.mean(
        loss_weights['alpha'] * mse_loss + \
        loss_weights['beta'] * recon_loss + \
        loss_weights['gamma'] * ssim_loss + \
        loss_weights['delta']* adv_loss + \
        loss_weights['epsilon'] * depth_loss + \
        loss_weights['zeta'] * geom_loss + \
        loss_weights['eta'] * tv_loss_value
    )

    return combined_loss

########################################################################################
# Peak Signal to Noise Ratio (PSNR)
# PSNR is a measure of reconstruction quality.
########################################################################################
def compute_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)

    if mse == 0:
        return float('inf')

    # Add a small positive number inside the square root to ensure the input is always non-negative
    return 20 * log10(1.0 / torch.sqrt(mse + 1e-10))

########################################################################################
# Structural Similarity Index (SSIM)
########################################################################################
def compute_ssim(img1, img2, C1=0.01**2, C2=0.03**2):
    mu1 = torch.mean(img1, dim=[1, 2, 3])
    mu2 = torch.mean(img2, dim=[1, 2, 3])

    sigma1_sq = torch.mean((img1 - mu1.reshape(-1, 1, 1, 1)) ** 2, dim=[1, 2, 3])
    sigma2_sq = torch.mean((img2 - mu2.reshape(-1, 1, 1, 1)) ** 2, dim=[1, 2, 3])
    sigma12   = torch.mean((img1 - mu1.reshape(-1, 1, 1, 1)) * (img2 - mu2.reshape(-1, 1, 1, 1)), dim=[1, 2, 3])

    SSIM_n = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    SSIM_d = (mu1 * mu1 + mu2 * mu2 + C1) * (sigma1_sq + sigma2_sq + C2)
    SSIM   = SSIM_n / SSIM_d

    return torch.mean(SSIM)

########################################################################################
# Edge Similarity Index (ESI)
########################################################################################
def compute_edge_similarity(img1, img2, device=device):
    # Ensure the images are single-channel and convert to NumPy arrays if they are tensors
    if torch.is_tensor(img1):
        img1 = img1.squeeze().cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.squeeze().cpu().numpy()

    # Calculate Sobel edges for image 1
    sobelx = cv2.Sobel(img1, cv2.CV_64F, dx=1, dy=0, ksize=5)
    sobely = cv2.Sobel(img1, cv2.CV_64F, dx=0, dy=1, ksize=5)
    edge1 = cv2.magnitude(sobelx, sobely)

    # Calculate Sobel edges for image 2
    sobelx = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=5)
    edge2 = cv2.magnitude(sobelx, sobely)

    # Convert edges back to PyTorch tensors for SSIM calculation
    edge1_tensor = torch.tensor(edge1, dtype=torch.float32).unsqueeze(0).to(device)
    edge2_tensor = torch.tensor(edge2, dtype=torch.float32).unsqueeze(0).to(device)

    return ssim(edge1_tensor, edge2_tensor)

########################################################################################
# Combined Score
# A combined score that takes into account PSNR, SSIM, and ESI
########################################################################################
def combined_score(psnr, ssim, edge_similarity, weights = [0.4, 0.3, 0.3]):
    weighted_psnr = 0.4 * psnr
    weighted_ssim = 0.3 * ssim
    weighted_edge = 0.3 * edge_similarity

    total_score = weighted_psnr + weighted_ssim + weighted_edge

    return total_score

# ----------------------------------------------------
# Define Evaluation Functions
# Gradient Penalty
# The gradient penalty is typically used in the context of Wasserstein GANs
# with Gradient Penalty (WGAN-GP).
# It enforces the Lipschitz constraint by penalizing the gradient norm
# of the discriminator's output with respect to its input.
# -------------------------------------------------
def compute_gradient_penalty(D, real_samples, fake_samples):
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=real_samples.device)

    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

    d_interpolates = D(interpolates)
    
    fake = torch.ones(d_interpolates.size(), requires_grad=False, device=real_samples.device)

    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

########################################################################################
# Training Loop
########################################################################################
def network_training(train_dataloader, generator, discriminator, gen_optim, dis_optim, loss_weights, num_epochs):
    #Learning Rate Scheduling
    gen_scheduler = StepLR(gen_optim, step_size=30, gamma=0.1)
    dis_scheduler = StepLR(dis_optim, step_size=30, gamma=0.1)

    # Initialize some variables for averaging
    best_psnr           = -float('inf')
    best_ssim           = -float('inf')
    best_esi            = -float('inf')
    best_combined_score = -float('inf')

    patience = 10  # Number of epochs to wait for improvement
    epochs_no_improve = 0  # Counter for epochs without improvement
    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_esi  = 0.0
    num_batches = 0
    min_psnr, max_psnr = float('inf'), float('-inf')
    min_ssim, max_ssim = float('inf'), float('-inf')
    min_esi,  max_esi  = float('inf'), float('-inf')
    std_psnr, std_ssim, std_esi = 0, 0, 0  # Standard deviation
    gen_loss, dis_loss = 0, 0  # Generator and Discriminator losse
    psnrs, ssims, esis = [], [], [] # Lists to store all PSNR and SSIM values for each epoch
    epoch_times        = []  # List to store time taken for each epoch

    # The gradient penalty coefficient
    lambda_gp = 10
    
    # Set the number of critic updates per generator update
    critic_updates_per_gen_update = 3

    torch.autograd.set_detect_anomaly(True)

    # Initialize GradScaler
    #scaler = GradScaler()

    # TRAINING LOOP
    for epoch in range(num_epochs):
        # Start time for this epoch
        start_time = time.time()
        epoch_metrics = {'psnr': {'total': 0, 'count': 0}, 'ssim': {'total': 0, 'count': 0}, 'esi': {'total': 0, 'count': 0}}

        for i, (intact, damaged) in enumerate(train_dataloader):
            # Move images to GPU
            intact, damaged = intact.to(device), damaged.to(device)

            if epoch == 0 and i == 0:
                writer.add_graph(generator, damaged)

            # Generate restored images from the damaged images
            restored = generator(damaged)

            # ---------------------------------------------------
            # Update Discriminator
            # ---------------------------------------------------
            for _ in range(critic_updates_per_gen_update):
                dis_optim.zero_grad()

                # Classify real and fake images
                output_real = discriminator(intact)
                output_fake = discriminator(restored.detach())

                real_loss = adversarial_loss(output_real, is_real=True)
                fake_loss = adversarial_loss(output_fake, is_real=False)

                # Compute gradient penalty
                gradient_penalty = compute_gradient_penalty(discriminator, intact, restored.detach())

                # Compute the total discriminator loss
                dis_loss = 0.5 * (real_loss + fake_loss) + lambda_gp * gradient_penalty

                # Compute gradients
                dis_loss.backward()

                # Clip gradients for discriminator
                clip_grad_norm_(discriminator.parameters(), max_norm=max_grad_norm)
                
                # Optimize
                dis_optim.step()

            # ---------------------------------------------------
            # Update Generator 
            # ---------------------------------------------------
            gen_optim.zero_grad()

            output_fake_for_gen = discriminator(restored)
            gen_loss            = combined_gen_loss(restored, intact, output_fake_for_gen, loss_weights.weights)

            # Backprop
            gen_loss.backward()
            
            # Clip gradients for generator
            clip_grad_norm_(generator.parameters(), max_norm=max_grad_norm)

            # Optimize
            gen_optim.step()

            writer.add_scalar('Generator Loss',     gen_loss, global_step=epoch * len(train_dataloader) + i)
            writer.add_scalar('Discriminator Loss', dis_loss, global_step=epoch * len(train_dataloader) + i)

        # ---------------------------------------------------
        # Validation step
        # ---------------------------------------------------
        with torch.no_grad():
            # Clip val gradients
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)

            for intact, damaged in val_dataloader:
                intact  = intact.to(device)
                damaged = damaged.to(device)

                # Generate restored images
                restored = generator(damaged)

                # Clamp the values to be between 0 and 1
                restored = restored.clamp(0, 1)
                intact   = intact.clamp(0, 1)

                writer.add_images('Restored Images', restored, global_step=epoch)

                # Compute PSNR for the current batch
                batch_psnr = compute_psnr(restored, intact)

                # Compute SSIM for the current batch
                batch_ssim = compute_ssim(restored, intact)

                # Compute ESI for the current batch
                batch_edge_similarity = compute_edge_similarity(restored, intact, device=device)

                psnrs.append(batch_psnr)
                ssims.append(batch_ssim)
                esis.append(batch_edge_similarity)

                # Update epoch metrics
                epoch_metrics['psnr']['total'] += batch_psnr
                epoch_metrics['psnr']['count'] += 1
                epoch_metrics['ssim']['total'] += batch_ssim
                epoch_metrics['ssim']['count'] += 1
                epoch_metrics['esi']['total']  += batch_edge_similarity
                epoch_metrics['esi']['count']  += 1
                

        # Switch back to training mode
        generator.train()
        discriminator.train()

        # Move tensors to CPU and convert to float
        psnrs_cpu = [x.cpu().item() if torch.is_tensor(x) else x for x in psnrs]
        ssims_cpu = [x.cpu().item() if torch.is_tensor(x) else x for x in ssims]
        esis_cpu  = [x.cpu().item() if torch.is_tensor(x) else x for x in esis]

        avg_psnr = np.mean(psnrs_cpu)
        avg_ssim = np.mean(ssims_cpu)
        avg_esi  = np.mean(esis_cpu)

        # At the end of the epoch, calculate average metrics
        avg_psnr_epoch = epoch_metrics['psnr']['total'] / epoch_metrics['psnr']['count']
        avg_ssim_epoch = epoch_metrics['ssim']['total'] / epoch_metrics['ssim']['count']
        avg_esi_epoch  = epoch_metrics['esi']['total']  / epoch_metrics['esi']['count']

        # Calculate combined score for the epoch
        epoch_combined_score = combined_score(avg_psnr_epoch, avg_ssim_epoch, avg_esi_epoch)

        # Save model checkpoints at regular intervals and best models
        if epoch % checkpoint_interval == 0 or avg_psnr_epoch > best_psnr or avg_ssim_epoch > best_ssim or avg_esi_epoch > best_esi:
            MODEL_NAME = f"generator_model_epoch_{epoch}.pth"

            torch.save(generator.state_dict(), os.path.join(MODEL_PATH, MODEL_NAME))

        # Early stopping based on combined score
        if epoch_combined_score > best_combined_score:
            best_combined_score = epoch_combined_score
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} due to no improvement.")
            break
        
        # Update loss weights based on performance
        performance_metrics = {
            'psnr': {'improved': avg_psnr_epoch > best_psnr, 'magnitude': abs(avg_psnr_epoch - best_psnr)},
            'ssim': {'improved': avg_ssim_epoch > best_ssim, 'magnitude': abs(avg_ssim_epoch - best_ssim)},
            'esi':  {'improved': avg_esi_epoch > best_esi, 'magnitude': abs(avg_esi_epoch - best_esi)}
        }

        loss_weights.update_weights(performance_metrics)

        # Normalize weights every N epochs
        if epoch % 10 == 0:
            loss_weights.normalize_weights()

        # Calculate min and max values
        min_psnr, max_psnr = np.min(psnrs_cpu), np.max(psnrs_cpu)
        min_ssim, max_ssim = np.min(ssims_cpu), np.max(ssims_cpu)
        min_esi,  max_esi  = np.min(esis_cpu),  np.max(esis_cpu)

        # Calculate standard deviation
        std_psnr, std_ssim, std_esi = np.std(psnrs_cpu), np.std(psnrs_cpu), np.std(esis_cpu)

        # Calculate time taken for this epoch
        epoch_time = time.time() - start_time

        # Store for future analysis if needed
        epoch_times.append(epoch_time)  

        # Logging for each epoch
        print(f" Epoch {epoch+1}/{num_epochs} - Time: {epoch_time:.2f}s")
        print(f" Loss Weights:   {loss_weights.weights}")
        print(f" Average PSNR:   {avg_psnr:.4f} (Min: {min_psnr:.4f}, Max: {max_psnr:.4f}, Std: {std_psnr:.4f}, Epoch Avg: {avg_psnr_epoch:.4f})")
        print(f" Average SSIM:   {avg_ssim:.4f} (Min: {min_ssim:.4f}, Max: {max_ssim:.4f}, Std: {std_ssim:.4f}, Epoch Avg: {avg_ssim_epoch:.4f})")
        print(f" Average ESI:    {avg_esi:.4f}  (Min: {min_esi:.4f},  Max: {max_esi:.4f},  Std: {std_esi:.4f},  Epoch Avg: {avg_esi_epoch:.4f})")
        print(f" Combined Score: {combined_score(avg_psnr, avg_ssim, avg_esi):.4f}")
        print(f" Generator Loss: {gen_loss:.4f}, Discriminator Loss: {dis_loss:.4f}")

        writer.add_scalar('Validation/Avg_PSNR', avg_psnr, global_step=epoch)
        writer.add_scalar('Validation/Avg_SSIM', avg_ssim, global_step=epoch)
        writer.add_scalar('Validation/Avg_ESI',  avg_esi,  global_step=epoch)
        writer.add_histogram('Generator/First_Layer_Weights', list(generator.parameters())[0], global_step=epoch)
        writer.add_scalar('Learning Rate/Generator', gen_optim.param_groups[0]['lr'], global_step=epoch)
        writer.add_scalar('Learning Rate/Discriminator', dis_optim.param_groups[0]['lr'], global_step=epoch)
        writer.flush()

        # Reset for next epoch
        psnrs.clear()
        ssims.clear()
        esis.clear()

        # Check for early stopping
        if epochs_no_improve >= patience:
            print("An early stopping here?!")
            #break
        
        # Step the learning rate scheduler
        gen_scheduler.step()
        dis_scheduler.step()

    hparam_dict = {'lr': generator_lr, 'batch_size': batch_size}
    metric_dict = {'best_psnr': best_psnr}

    writer.add_hparams(hparam_dict, metric_dict)

    writer.close()


####################################################################################################
# Main
# - Load displacement maps
# - Generate synthetic displacement maps
# - Validate the generated data
####################################################################################################
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Load the dataset
    train_dataloader, val_dataloader = load_dataset()

    # Instantiate the generator and discriminator
    generator, discriminator = instantiate_networks()

    # Initialize optimizers
    gen_optim, dis_optim = init_optimizer(generator, discriminator, generator_lr, discriminator_lr)

    # Train the network
    network_training(train_dataloader, generator, discriminator, gen_optim, dis_optim, loss_weights, num_epochs)
