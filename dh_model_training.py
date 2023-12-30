import os
import time
import sys
import numpy as np
import logging
import cv2
import glob

import torch
import torch.nn            as nn
import torch.nn.functional as F

from torch.nn.utils           import clip_grad_norm_
from torch.utils.data         import DataLoader
from torch.optim              import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.cuda.amp import GradScaler, autocast

from torchvision import transforms

from PIL import Image

from sklearn.model_selection import train_test_split

# Add the project directory to the Python path
sys.path.append('./')

# Import the DeepHadad network
import core.networks             as dh_networks
import utils.performance_metrics as dh_metrics


from utils.DisplacementMapDataset  import DisplacementMapDataset
from loss.DynamicLossWeights       import DynamicLossWeights
from loss.WeightedSSIMLoss         import WeightedSSIMLoss
from loss.DepthConsistencyLoss     import DepthConsistencyLoss
from loss.GeometricConsistencyLoss import GeometricConsistencyLoss
from loss.SharpnessLoss            import SharpnessLoss

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

# PyTorch version
print("PyTorch version: " + torch.__version__)

# Set constants
PROJECT_PATH            = './'
TRAINING_DATASET_PATH   = PROJECT_PATH + 'data/small_training_dataset'
X_TRAINING_DATASET_PATH = TRAINING_DATASET_PATH + '/X'
Y_TRAINING_DATASET_PATH = TRAINING_DATASET_PATH + '/Y'
MODEL_PATH              = PROJECT_PATH + 'models/'
IMAGE_EXTENSIONS        = [".png", ".jpg", ".tif"]

# Set hyperparameters
generator_lr        = 2e-5
discriminator_lr    = 3e-5
batch_size          = 32
num_epochs          = 100
checkpoint_interval = 10
max_grad_norm       = 1.0
patience            = 50
weights_type        = 'delta' # 'alpha', 'gamma', 'zeta' 

# Number of critic updates per generator update
critic_updates_per_gen_update = 2

# The gradient penalty coefficient
lambda_gp = 10

# Update loss weights based on performance
improvement_threshold = 0.005  # 1% improvement

# Dynamic Loss Weights for adjusting the loss weights during training
loss_weights = DynamicLossWeights(weights_type=weights_type)

# Mean Squared Error(MSE) Loss
# It ensures the overall structure of the reconstructed image
# is similar to the intact image.
# MSE could encourage blurred details, which can be detrimental for text recovery 
# and for sharp engravings.
mean_sq_error_loss = nn.MSELoss().to(device)

# L1 Loss
# It helps in recovering finer details without overly penalizing slight deviations that 
# aren't perceptually significant.
l1_loss = nn.L1Loss().to(device)

# Perceptual Loss
# Captures textural and stylistic features.
# luminance, and contrast in an image.
# A higher weight is crucial as it emphasizes on the perceptual similarity, which is key for letter reconstruction.
weighted_ssim_loss = WeightedSSIMLoss(data_range=255, size_average=True, channel=1, weight=1.0)

# LPIPS Loss
# LPIPS is a perceptual loss that uses a pretrained VGG19 network to calculate
#lpips_loss = LPIPS(net='vgg').to(device)

# Adversarial Loss
# TO encourage the generator to create images that are indistinguishable
# from the intact displacement maps.
# Encourages realism in the generated maps.
# Keeping this weight lower ensures that the focus remains on structural and textural accuracy rather than just realism.
bce_with_logits_loss = nn.BCEWithLogitsLoss().to(device)

# DepthConsistencyLoss
# This is a robust loss that combines the benefits of L1 and L2 losses.
# It can be particularly useful if there's a lot of noise in the damaged maps.
# Depth data is uncertain in some places
# This loss can help in smoothing the depth map without losing essential details.
depth_consistency_loss = DepthConsistencyLoss().to(device)

# Geometric Consistency Loss
# Maintains geometric integrity of depth information.
# This will help in preserving the contours and shapes of letters in the displacement maps.
geometric_consistency_loss = GeometricConsistencyLoss().to(device)

# Sharpness Loss
sharpness_loss = SharpnessLoss().to(device)


# Get the paths of all images in a directory
def get_image_paths(directory):
    image_paths = []

    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(directory, '*' + ext)))

    return sorted(image_paths)


def calculate_mean_and_std(real_image_paths, synthetic_image_paths, common_transforms):
    # Calculate mean and std for the dataset
    mean = 0.
    std = 0.

     # Use a subset to calculate mean and std for efficiency
    subset_intact_image_paths  = real_image_paths      # real_image_paths[:100]
    subset_damaged_image_paths = synthetic_image_paths # synthetic_image_paths[:100]
    min_val, max_val           = float('inf'), -float('inf')

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

    mean /= len(real_image_paths)
    std  /= len(synthetic_image_paths)

    # Now mean and std should be tensors of size 1
    print(mean.size())  # Should print torch.Size([1])
    print(std.size())  # Should print torch.Size([1])

    return mean, std


def load_dataset():
    """
    Loads the training dataset
    :return: The dataloaders for both training dataset and validation dataset
    """
    # Get images from path
    intact_image_paths  = get_image_paths(X_TRAINING_DATASET_PATH)
    damaged_image_paths = get_image_paths(Y_TRAINING_DATASET_PATH)

    print(f"Path to Intact Images:    {X_TRAINING_DATASET_PATH}")
    print(f"Path to Synhtetic Images: {Y_TRAINING_DATASET_PATH}")

    print(f"Number of Intact Inscriptions:    {len(intact_image_paths)}")
    print(f"Number of Synhtetic Inscriptions: {len(damaged_image_paths)}")

    assert len(intact_image_paths) == len(damaged_image_paths), "Number of intact and damaged images must be the same"

    # Common Data Augmentation
    common_transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.Lambda(lambda x: x.convert('L')),
        transforms.ToTensor()
    ])

    # Calculate mean and std for the dataset
    mean, std = calculate_mean_and_std(intact_image_paths, damaged_image_paths, common_transforms)

    # Data Augmentation for the Real Images
    image_transforms = transforms.Compose([
        #transforms.RandomResizedCrop(512, scale=(0.8, 1.0)),
        transforms.RandomRotation(10), 
        #transforms.RandomCrop(224),
        common_transforms,
        transforms.Normalize(mean=mean.tolist(), std=std.tolist())
    ])

    # Split the data into training and validation sets
    # The validation set will be 20% of the entire dataset
    # The random seed is set to 42 so that the results are reproducible. 
    # Random seed represents the starting point for the random number generator. If you set it to 1
    train_real, val_real, train_synthetic, val_synthetic = train_test_split(
        intact_image_paths, damaged_image_paths, test_size = 0.2, shuffle=False, random_state = None)

    # Create training and validation datasets and dataloaders
    train_dataset = DisplacementMapDataset(train_real, train_synthetic, transform = image_transforms)
    val_dataset   = DisplacementMapDataset(val_real,   val_synthetic,   transform = image_transforms)

    # Create dataset and dataloader
    # The dataloader will return a list of [real, synthetic] images for each batch of images
    # the real images are the ground truth and estimated displacement maps
    # the synthetic images are the procedurally enhanced displacement maps
    # the batch size is 1 because we're providing pairs of images as input
    # the shuffle is set to False because we want to maintain the order of the images
    # the order of the images is important because we're using the same index to retrieve the images
    # from the real and synthetic folders
    # the number of workers is set to 0 because we're using a small dataset
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = False, num_workers = 0)
    val_dataloader   = DataLoader(val_dataset,   batch_size = batch_size, shuffle = False, num_workers = 0)

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
# Initialize Optimizers
# RMSProp or Adagrad
########################################################################################
def init_optimizer(generator, discriminator):
    # Initialize optimizers
    gen_optim = Adam(generator.parameters(),     lr=generator_lr,     betas=(0.5, 0.999))
    dis_optim = Adam(discriminator.parameters(), lr=discriminator_lr, betas=(0.5, 0.999))

    return gen_optim, dis_optim

########################################################################################
# adversarial loss
# Encourages the generator to create images that are indistinguishable
# from the intact displacement maps.
# Encourages realism in the generated maps.
# Keeping this weight lower ensures that the focus remains on 
# structural and textural accuracy rather than just realism.    
########################################################################################
def adversarial_loss(discriminator_preds, is_real=True):
    if is_real:
        labels = torch.ones_like(discriminator_preds).to(device)
    else:
        labels = torch.zeros_like(discriminator_preds).to(device)

    return bce_with_logits_loss(discriminator_preds, labels)


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
# Combined generator loss function
########################################################################################
def combined_gen_loss(gen_imgs, real_imgs, discriminator_preds, loss_weights):
    # Mean Squared Error Loss
    mse_loss = mean_sq_error_loss(gen_imgs, real_imgs)

    # Reconstruction Loss (L1)
    recon_loss = l1_loss(gen_imgs, real_imgs)

    # Perceptual Loss
    ssim_loss = weighted_ssim_loss(gen_imgs, real_imgs).to(device)

    # Adversarial Loss for the generator
    adv_loss = adversarial_loss(discriminator_preds, is_real=False)

    # Charbonnier Loss (robust L1/L2 loss)
    depth_loss = depth_consistency_loss(gen_imgs, real_imgs)

    # predicted_map and target_map are your generated and ground truth displacement maps, respectively
    geom_loss = geometric_consistency_loss(gen_imgs, real_imgs)

    sharp_loss = sharpness_loss(gen_imgs, real_imgs)

    # Check for NaNs in losses
    if any(torch.isnan(loss).any() for loss in [mse_loss, recon_loss, ssim_loss, adv_loss, depth_loss, geom_loss, sharp_loss]):
        print("NaNs detected in generator losses")
        #return torch.tensor(0.0).to(gen_imgs.device)  # Return a default loss value

    # Combine the losses
    combined_loss = torch.mean(
        loss_weights['alpha'] * mse_loss + \
        loss_weights['beta'] * recon_loss + \
        loss_weights['gamma'] * ssim_loss + \
        loss_weights['delta']* adv_loss + \
        loss_weights['epsilon'] * depth_loss + \
        loss_weights['zeta'] * geom_loss + \
        loss_weights['eta'] * sharp_loss
        #loss_weights['eta'] * tv_loss_value
    )

    return combined_loss


def train_discriminator_step(discriminator, dis_optim, synthetic_dm, enhanced_dm, lambda_gp, max_grad_norm):
    """
    Update Discriminator Step
    """
    # Log min, max, and NaN count of inputs
    #print("Synthetic_dm Min:", synthetic_dm.min().item(), "Max:", synthetic_dm.max().item(), "NaNs:", torch.isnan(synthetic_dm).sum().item())
    #print("Enhanced_dm Min:", enhanced_dm.min().item(), "Max:", enhanced_dm.max().item(), "NaNs:", torch.isnan(enhanced_dm).sum().item())

    # Check for NaNs in inputs and skip the step if found
    if torch.isnan(synthetic_dm).any() or torch.isnan(enhanced_dm).any():
        print("Skipping step due to NaNs in input")
        return torch.tensor(0.0).to(synthetic_dm.device)  # Return a default value
    
    for _ in range(critic_updates_per_gen_update):
        dis_optim.zero_grad()

        # Check for NaNs in inputs
        if torch.isnan(synthetic_dm).any():
            print("NaNs detected in synthetic_dm")
            continue

        if torch.isnan(enhanced_dm).any():
            print("NaNs detected in enhanced_dm")
            continue

        # Classify real and fake images
        output_real = discriminator(synthetic_dm) # the synthetic_dm is the real image here
        output_fake = discriminator(enhanced_dm.detach())

        real_loss = adversarial_loss(output_real, is_real=True)
        fake_loss = adversarial_loss(output_fake, is_real=False)

        # Compute gradient penalty
        gradient_penalty = dh_metrics.compute_gradient_penalty(discriminator, enhanced_dm.detach(), synthetic_dm)

        # Compute the total discriminator loss
        dis_loss = 0.5 * (real_loss + fake_loss) + lambda_gp * gradient_penalty

        # Check for NaNs in the loss
        assert not torch.isnan(dis_loss).any(), "dis_loss contains NaN values"

        # Compute gradients
        dis_loss.backward()

        # Validate gradients
        for param in discriminator.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print("NaNs detected in discriminator gradients")
                return dis_loss
        
        # Clip gradients for discriminator
        clip_grad_norm_(discriminator.parameters(), max_norm=max_grad_norm)
            
        # Optimize
        dis_optim.step()
    
    return dis_loss


def train_generator_step(generator, gen_optim, enhanced_dm, real_dm, output_fake_for_gen, loss_weights, max_grad_norm):
    """
    Update Generator Step
    """
    gen_optim.zero_grad()

    gen_loss = combined_gen_loss(enhanced_dm, real_dm, output_fake_for_gen, loss_weights.weights)

    # Backprop
    gen_loss.backward()
    
    # Clip gradients for generator
    clip_grad_norm_(generator.parameters(), max_norm=max_grad_norm)

    # Optimize
    gen_optim.step()

    return gen_loss

    
def train_step(generator, gen_optim, discriminator, dis_optim, train_dataloader):
    """
    Training step for each batch
    data is a list of [real, synthetic] images
    real_dms are the ground truth and estimated displacement maps
    synthetic_dms are the procedurally enhanced displacement maps
    """
    generator.train()
    discriminator.train()

    for i, (real_dm, synthetic_dm) in enumerate(train_dataloader):
        # Move images to GPU
        real_dm      = real_dm.to(device)
        synthetic_dm = synthetic_dm.to(device)

        # Generate the enhanced image from the real image
        enhanced_dm = generator(real_dm)

        # train the discriminator first
        dis_loss = train_discriminator_step(discriminator, dis_optim, synthetic_dm, enhanced_dm, lambda_gp, max_grad_norm)

        # Classify real and fake images again after training the discriminator
        output_fake_for_gen = discriminator(enhanced_dm)

        # train the generator next
        gen_loss = train_generator_step(generator, gen_optim, enhanced_dm, real_dm, output_fake_for_gen, loss_weights, max_grad_norm)

    return gen_loss, dis_loss

def validation_step(generator, discriminator, val_dataloader, psnrs, ssims, esis, max_grad_norm):
    """
    Validation step after each epoch
    """
    # Switch to evaluation mode
    generator.eval()
    discriminator.eval()

    with torch.no_grad():
        # Clip val gradients
        clip_grad_norm_(generator.parameters(), max_norm=max_grad_norm)

        for real_dm, synthetic_dm in val_dataloader:
            real_dm      = real_dm.to(device)
            synthetic_dm = synthetic_dm.to(device)
            enhanced_dm  = generator(real_dm)

            # Clamp the values to be between 0 and 1
            enhanced_dm  = enhanced_dm.clamp(0, 1)
            synthetic_dm = synthetic_dm.clamp(0, 1)

            # Compute PSNR, SSIM, ESI for the current batch
            batch_psnr = dh_metrics.compute_psnr(enhanced_dm, synthetic_dm)
            batch_ssim = dh_metrics.compute_ssim(enhanced_dm, synthetic_dm)
            batch_esi  = dh_metrics.compute_edge_similarity(enhanced_dm, synthetic_dm)

            # Update epoch metrics
            psnrs.append(batch_psnr)
            ssims.append(batch_ssim)
            esis.append(batch_esi)

    # Switch back to training mode
    generator.train()
    discriminator.train()


def calculate_performance_metrics(epoch, best_psnr, psnrs, best_ssim, ssims, best_esi, esis, 
                                  epoch_time, gen_loss, dis_loss, gen_optim, dis_optim):
    """
    Calculate the performance metrics for the current epoch
    """
    # Move tensors to CPU and convert to float
    psnrs_cpu = [x.cpu().item() if torch.is_tensor(x) else x for x in psnrs]
    ssims_cpu = [x.cpu().item() if torch.is_tensor(x) else x for x in ssims]
    esis_cpu  = [x.cpu().item() if torch.is_tensor(x) else x for x in esis]

    # Calculate min and max values
    min_psnr, max_psnr = np.min(psnrs_cpu), np.max(psnrs_cpu)
    min_ssim, max_ssim = np.min(ssims_cpu), np.max(ssims_cpu)
    min_esi,  max_esi  = np.min(esis_cpu),  np.max(esis_cpu)

    # Calculate standard deviation
    std_psnr, std_ssim, std_esi = np.std(psnrs_cpu), np.std(psnrs_cpu), np.std(esis_cpu)

    # Average metrics for this epoch
    avg_psnr, avg_ssim, avg_esi = np.mean(psnrs_cpu), np.mean(ssims_cpu), np.mean(esis_cpu)

    # Calculate combined score
    avg_combined_score = dh_metrics.combined_score(avg_psnr, avg_ssim, avg_esi)

    # Check if the average PSNR for this epoch is higher than the best seen so far
    is_psnr_improved = avg_psnr > best_psnr * (1 + improvement_threshold)
    is_ssim_improved = avg_ssim > best_ssim * (1 + improvement_threshold)
    is_esi_improved  = avg_esi  > best_esi  * (1 + improvement_threshold)

    # Performance metrics
    performance_metrics = {
        'psnr': {'value': avg_psnr, 'improved': is_psnr_improved, 'magnitude': abs(avg_psnr - best_psnr)},
        'ssim': {'value': avg_ssim, 'improved': is_ssim_improved, 'magnitude': abs(avg_ssim - best_ssim)},
        'esi':  {'value': avg_esi,  'improved': is_esi_improved,  'magnitude': abs(avg_esi - best_esi)}
    }

    # Logging for each epoch 
    print("")
    print(f" Epoch:                            {epoch + 1}/{num_epochs}")
    print(f" Time:                             {epoch_time:.2f}s Current: {time.strftime('%H:%M:%S', time.gmtime())}")
    print(f" Epoch Loss Weights:               {loss_weights.weights}")
    print(f" Epoch Average PSNR:               {avg_psnr:.4f} (Min: {min_psnr:.4f}, Max: {max_psnr:.4f}, Std: {std_psnr:.4f})")
    print(f" Epoch Average SSIM:               {avg_ssim:.4f} (Min: {min_ssim:.4f}, Max: {max_ssim:.4f}, Std: {std_ssim:.4f})")
    print(f" Epoch Average ESI:                {avg_esi:.4f}  (Min: {min_esi:.4f},  Max: {max_esi:.4f},  Std: {std_esi:.4f})")
    print(f" Epoch Combined Score:             {avg_combined_score:.4f}")
    print(f" Epoch Generator Loss:             {gen_loss.item():.4f}")
    print(f" Epoch Discriminator Loss:         {dis_loss.item():.4f}")
    print(f" Epoch Learning Rate:  Generator   {gen_optim.param_groups[0]['lr']:.6f}, Discriminator -> {dis_optim.param_groups[0]['lr']:.6f}")
    print(f" Epoch Performance:                {performance_metrics}")

    # Reset for next epoch 
    psnrs.clear()
    ssims.clear()
    esis.clear()
    
    return performance_metrics, avg_combined_score


def save_model(generator, epoch, loss_weights):
    """
    Save the model checkpoint
    """
    MODEL_NAME = f"dh_{weights_type}_model_ep_{epoch}{loss_weights.get_weights_as_string()}.pth"

    print(f"-> Saving model checkpoint at epoch {epoch + 1}")

    torch.save(generator.state_dict(), os.path.join(MODEL_PATH, MODEL_NAME))

########################################################################################
# Training Loop
########################################################################################
def network_training(train_dataloader, val_dataloader, generator, discriminator, gen_optim, dis_optim):
    #Learning Rate Scheduling
    gen_scheduler = ReduceLROnPlateau(gen_optim, mode='min', factor=0.1, patience=10, verbose=True)
    dis_scheduler = ReduceLROnPlateau(dis_optim, mode='min', factor=0.1, patience=10, verbose=True)

    # Initialize some variables for averaging
    best_psnr  = -float('inf')
    best_ssim  = -float('inf')
    best_esi   = -float('inf')

    epochs_no_improve  = 0           # Counter for epochs without improvement
    psnrs, ssims, esis = [], [], []  # Lists to store all PSNR and SSIM values for each epoch
    epoch_times        = []          # List to store time taken for each epoch

    # Enable anomaly detection for debugging
    torch.autograd.set_detect_anomaly(True)

    # Initialize the gradient scaler for mixed precision training
    #scaler = GradScaler()

    # TRAINING LOOP
    for epoch in range(num_epochs):
        # Start time for this epoch
        start_time = time.time()

        # Training step
        gen_loss, dis_loss = train_step(generator, gen_optim, discriminator, dis_optim, train_dataloader)

        # Validation step
        validation_step(generator, discriminator, val_dataloader, psnrs, ssims, esis, max_grad_norm)

        # Calculate time taken for this epoch
        epoch_time = time.time() - start_time

        # Store for future analysis
        epoch_times.append(epoch_time) 

        # Calculate performance metrics for this epoch
        performance_metrics, avg_combined_score = calculate_performance_metrics(
            epoch, best_psnr, psnrs, best_ssim, ssims, best_esi, esis, 
            epoch_time, gen_loss, dis_loss, gen_optim, dis_optim)

        # Save model checkpoints at regular intervals and best models
        save_checkpoint = False
        
        # If the average PSNR for this epoch is higher than the best seen so far, update best_psnr
        if performance_metrics['psnr']['improved'] == True:
            best_psnr = performance_metrics['psnr']['value']
            save_checkpoint = True

        # If the average SSIM for this epoch is higher than the best seen so far, update best_ssim
        if performance_metrics['ssim']['improved'] == True:
            best_ssim = performance_metrics['ssim']['value']
            save_checkpoint = True
        
        # If the average ESI for this epoch is higher than the best seen so far, update best_esi
        if performance_metrics['esi']['improved'] == True:
            best_esi = performance_metrics['esi']['value']
            save_checkpoint = True
        
        # Save model checkpoints at regular intervals and best models
        if save_checkpoint==True or (epoch + 1) % checkpoint_interval == 0:
            save_model(generator, epoch, loss_weights)

        # Early stopping if there's no improvement in the average PSNR, SSIM, or ESI for a certain number of epochs
        if save_checkpoint==True:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        #Update loss weights when there's no improvement
        if save_checkpoint==False and epochs_no_improve > 2:
            loss_weights.update_weights(performance_metrics, epoch, num_epochs)

        if epoch == 10:
            loss_weights.weights = loss_weights.delta_weights_stage_2
            epochs_no_improve = 0
        
        if epoch == 15:
            loss_weights.weights = loss_weights.delta_weights_stage_3
            epochs_no_improve = 0

        if epoch == 20:
            loss_weights.weights = loss_weights.delta_weights_stage_4
            epochs_no_improve = 0
        
        if epoch == 25:
            loss_weights.weights = loss_weights.gamma_weights
            epochs_no_improve = 0

        if epoch == 30:
            loss_weights.weights = loss_weights.delta_weights
            epochs_no_improve = 0
        
        # Break if there's no improvement for a certain number of epochs
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} due to no improvement.")
            break

        # Step the learning rate scheduler based on the average combined score
        gen_scheduler.step(avg_combined_score)
        dis_scheduler.step(avg_combined_score)

    #save the model
    save_model(generator, epoch, loss_weights)

    print(f"Average time per epoch: {np.mean(epoch_times):.2f}s")
    print(f"Final Loss Weights:     {loss_weights.weights}")

########################################################################################
# Main
# - Load displacement maps
# - Generate synthetic displacement maps
# - Validate the generated data
########################################################################################
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Load the dataset
    train_dataloader, val_dataloader = load_dataset()

    # Instantiate the generator and discriminator
    generator, discriminator = instantiate_networks()

    # Initialize optimizers
    gen_optim, dis_optim = init_optimizer(generator, discriminator)

    # Train the network
    network_training(train_dataloader, val_dataloader, generator, discriminator, gen_optim, dis_optim)



