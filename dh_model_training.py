import os
import time
import sys
import numpy as np
import logging
import glob
import shutil
import torch

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

# Import the DeepHadad networks
from core.DHadadGenerator     import DHadadGenerator
from core.DHadadDiscriminator import DHadadDiscriminator
from core.DHadadLossFunctions import DHadadLossFunctions
from core.DHadadLossWeights   import DHadadLossWeights

from utils.DisplacementMapDataset  import DisplacementMapDataset

import utils.performance_metrics as dh_metrics

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    device = torch.device("cuda")

    print("CUDA device found.")
else:
    device = torch.device("cpu")

    print("No CUDA device found, using CPU.")

# Check if MPS (Multi-Process Service) is available
if torch.backends.mps.is_available():
    device = torch.device("mps")

    print("MPS device found.")
else:
    print("MPS device not found.")

# PyTorch version
print("PyTorch version: " + torch.__version__)

# Set hyperparameter values for training 
generator_lr        = 2e-5
discriminator_lr    = 3e-5
batch_size          = 1
num_epochs          = 100
checkpoint_interval = 10
max_grad_norm       = 1.0
patience            = 50
improvement_threshold = 0.004  # 1% improvement
lambda_gp           = 10    # The gradient penalty coefficient
weights_type        = 'depth' # 'alpha', 'gamma', 'zeta' 

# Set the project paths
PROJECT_PATH            = './'
TRAINING_DATASET_PATH   = PROJECT_PATH + 'data/small_training_dataset'
X_TRAINING_DATASET_PATH = TRAINING_DATASET_PATH + '/X'
Y_TRAINING_DATASET_PATH = TRAINING_DATASET_PATH + '/Y'
MODEL_PATH              = PROJECT_PATH + 'models/'
MODEL_WEIGHTS_PATH      = MODEL_PATH + 'dh_depth_model_ep_2_r1.00_p1.10_a0.60_g1.50_d1.20_s1.00.pth'

IMAGE_EXTENSIONS        = [".png", ".jpg", ".tif"]

# Number of critic updates per generator update
critic_updates_per_gen_update = 2

# Dynamic Loss Weights for adjusting the loss weights during training
loss_weights  = DHadadLossWeights(weights_type=weights_type)

# Loss Functions
loss_functions = DHadadLossFunctions()

# Get the paths of all images in a directory
def get_image_paths(directory):
    image_paths = []

    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(directory, '*' + ext)))

    return sorted(image_paths)

def clear_local_directory(path):
    if os.path.exists(path):
      shutil.rmtree(path)
    
    os.makedirs(path)


# def clean():
#     """
#     Clean the local directory
#     """
#     # Clear the existing data in the local directory and create a fresh directory
#     clear_local_directory(project_local_path)

#     # Copy data to local storage for faster access
#     # Make sure to append the trailing slash to ensure contents are copied into the directory
#     shutil.copytree(os.path.join(project_drive_path), project_local_path)

#     # Set the project path to the local directory
#     project_path = os.path.join(project_local_path, 'DeepHadadProject')


def calculate_mean_and_std(real_image_paths, synthetic_image_paths, common_transforms):
    # Calculate mean and std for the dataset
    mean = 0.
    std  = 0.

     # Use a subset to calculate mean and std for efficiency
    subset_intact_image_paths  = real_image_paths[:100]
    subset_damaged_image_paths = synthetic_image_paths[:100]
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

    # Common Data Augmentation for both Real and Synthetic Images
    # It takes a PIL image as input and returns a tensor
    # Resize to 512x512, convert to grayscale, and convert to tensor
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
    generator = DHadadGenerator(gen_in_channels, gen_out_channels).to(device)

    # Specify the input channel configurations (it typically takes two inputs)
    # we assume we're providing pairs of images (intact and restored) as input
    disc_in_channels = 1

    discriminator = DHadadDiscriminator(disc_in_channels).to(device)

    return generator, discriminator


def load_model_weights(model, model_path):
    """
    Loads the model weights from the specified path
    :param model: The model to load the weights into
    :param model_path: The path to the model weights if available
    :return: The model with the loaded weights
    """
    # Check if the model weights exist
    if not os.path.exists(model_path):
        print(f"Model weights not found at {model_path}. Training from scratch.")
        return model
    else:
        print(f"Loading model weights from {model_path}")

    # Load the model weights
    checkpoint = torch.load(model_path, map_location=device)

    # Load the model state dictionary
    model.load_state_dict(checkpoint)

    return model


########################################################################################
# Initialize Optimizers
# RMSProp or Adagrad
########################################################################################
def init_optimizer(generator, discriminator):
    # Initialize optimizers
    gen_optim = Adam(generator.parameters(),     lr=generator_lr,     betas=(0.5, 0.999))
    dis_optim = Adam(discriminator.parameters(), lr=discriminator_lr, betas=(0.5, 0.999))

    return gen_optim, dis_optim


def train_discriminator_step(discriminator, dis_optim, fake_dm, real_dm, lambda_gp, max_grad_norm):
    """
    Update Discriminator Step
    """
    # Log min, max, and NaN count of inputs
    #print("Synthetic_dm Min:", synthetic_dm.min().item(), "Max:", synthetic_dm.max().item(), "NaNs:", torch.isnan(synthetic_dm).sum().item())
    #print("Enhanced_dm Min:", enhanced_dm.min().item(), "Max:", enhanced_dm.max().item(), "NaNs:", torch.isnan(enhanced_dm).sum().item())

    # Check for NaNs in inputs and skip the step if found
    if torch.isnan(real_dm).any() or torch.isnan(fake_dm).any():
        print("Skipping step due to NaNs in input")
    
    for _ in range(critic_updates_per_gen_update):
        dis_optim.zero_grad()

        # Detach the enhanced_dm from the generator
        detached_fake_dm = fake_dm.detach()

        # Compute discriminator loss
        dis_loss = loss_functions.calculate_dis_loss(discriminator, detached_fake_dm, real_dm, lambda_gp)

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

    gen_loss = loss_functions.calculate_gen_loss(enhanced_dm, real_dm, output_fake_for_gen, loss_weights.weights)

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
        dis_loss = train_discriminator_step(discriminator, dis_optim, enhanced_dm, synthetic_dm, lambda_gp, max_grad_norm)

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
    is_psnr_improved = avg_psnr >= best_psnr * (1 + improvement_threshold)
    is_ssim_improved = avg_ssim >= best_ssim * (1 + improvement_threshold)
    is_esi_improved  = avg_esi  >= best_esi  * (1 + improvement_threshold)

    # Performance metrics
    performance_metrics = {
        'psnr': {'value': avg_psnr, 'improved': is_psnr_improved, 'magnitude': abs(avg_psnr - best_psnr)},
        'ssim': {'value': avg_ssim, 'improved': is_ssim_improved, 'magnitude': abs(avg_ssim - best_ssim)},
        'esi':  {'value': avg_esi,  'improved': is_esi_improved,  'magnitude': abs(avg_esi - best_esi)}
    }

    # Logging for each epoch 
    dh_metrics.print_performance_metrics(epoch, num_epochs, epoch_time, loss_weights,
                            avg_psnr, min_psnr, max_psnr, std_psnr,
                            avg_ssim, min_ssim, max_ssim, std_ssim,
                            avg_esi, min_esi, max_esi, std_esi,
                            avg_combined_score, performance_metrics,                          
                            gen_loss, dis_loss, gen_optim, dis_optim)

    return performance_metrics, avg_combined_score


def save_model(generator, epoch, loss_weights):
    """
    Save the model checkpoint
    """
    MODEL_NAME = f"dh_{weights_type}_model_ep_{epoch}{loss_weights.get_weights_as_string()}.pth"

    print(f"-> Saving model checkpoint at epoch {epoch}")

    torch.save(generator.state_dict(), os.path.join(MODEL_PATH, MODEL_NAME))

########################################################################################
# Training Loop
########################################################################################
def network_training(train_dataloader, val_dataloader, generator, discriminator, gen_optim, dis_optim):
    #Learning Rate Scheduling
    gen_scheduler = ReduceLROnPlateau(gen_optim, mode='min', factor=0.1, patience=5)
    dis_scheduler = ReduceLROnPlateau(dis_optim, mode='min', factor=0.1, patience=5)

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
            (epoch + 1), best_psnr, psnrs, best_ssim, ssims, best_esi, esis, 
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
            save_model(generator, (epoch + 1), loss_weights)

        # Early stopping if there's no improvement in the average PSNR, SSIM, or ESI for a certain number of epochs
        if save_checkpoint==True:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        #Update loss weights when there's no improvement
        # if save_checkpoint==False and epochs_no_improve > 2:
        #     loss_weights.update_weights(performance_metrics, epoch, num_epochs)

        #Update loss weights at the end of each epoch
        loss_weights.manage_epoch_weights(epoch + 1)
        
        # Break if there's no improvement for a certain number of epochs
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} due to no improvement.")
            break

        # Step the learning rate scheduler based on the average combined score
        gen_scheduler.step(avg_combined_score)
        dis_scheduler.step(avg_combined_score)

        # Reset for next epoch 
        psnrs.clear()
        ssims.clear()
        esis.clear()

    #save the model
    save_model(generator, (epoch + 1), loss_weights)

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

    # Load the model weights if available
    generator = load_model_weights(generator, MODEL_WEIGHTS_PATH)

    # Initialize optimizers
    gen_optim, dis_optim = init_optimizer(generator, discriminator)

    # Train the network
    network_training(train_dataloader, val_dataloader, generator, discriminator, gen_optim, dis_optim)



