import functools
import glob
import logging
import os
import time
import numpy as np
import torch

from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from models.DHadadGenerator import DHadadGenerator
from models.UnetGenerator import UnetGenerator
from models.PatchGANDiscriminator import build_discriminator as DHadadDiscriminator
from models.DHadadLossFunctions import DHadadLossFunctions
from models.DHadadLossWeights import DHadadLossWeights

from generation.synthetic_glyph_generation import SyntheticDatasetGenerator

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
generator_lr = 0.0002
discriminator_lr = 0.0003
batch_size = 4
num_epochs = 100
checkpoint_interval = 3
max_grad_norm = 1.0
patience = 100
improvement_threshold = 0.004  # 1% improvement
lambda_gp = 10  # The gradient penalty coefficient
weights_type = 'depth'  #

# Number of critic updates per generator update
critic_updates_per_gen_update = 2

# Set the project paths
PROJECT_PATH = './'
TRAINING_DATASET_PATH = PROJECT_PATH + 'data/training_dataset'
X_TRAINING_DATASET_PATH = TRAINING_DATASET_PATH + '/X'
Y_TRAINING_DATASET_PATH = TRAINING_DATASET_PATH + '/Y'

DISPLACEMENT_MAPS_PATH = PROJECT_PATH + 'data/glyph_dataset/preserved_glyphs/displacement_maps/'
CRACK_D_MAP_DATASET_PATH = PROJECT_PATH + 'data/masks_dataset/'
INPUT_TRAINING_DATASET_PATH = PROJECT_PATH + 'data/training_dataset/X/'
TARGET_TRAINING_DATASET_PATH = PROJECT_PATH + 'data/training_dataset/Y/'

MODEL_PATH = PROJECT_PATH + 'trained_models/'
MODEL_WEIGHTS_PATH = MODEL_PATH + 'dh_depth_model_ep_26_l0.60_s0.60_a0.20_g0.30_s0.10.pth'

IMAGE_EXTENSIONS = [".png", ".jpg", ".tif"]

# Dynamic Loss Weights for adjusting the loss weights during training
loss_weights = DHadadLossWeights(weights_type=weights_type)

# Loss Functions
loss_functions = DHadadLossFunctions()


# Get the paths of all images in a directory
def get_image_paths(directory):
    image_paths = []

    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(directory, '*' + ext)))

    return sorted(image_paths)


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """

    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'group':
        norm_layer = functools.partial(nn.GroupNorm, num_groups=32, affine=True)
    elif norm_type == 'layer':
        norm_layer = functools.partial(nn.LayerNorm, normalized_shape=[512, 512])
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)

    return norm_layer


########################################################################################
# Instantiate the generator and discriminator

########################################################################################
def instantiate_networks(type='unet_512', num_downs=7, ngf=128, norm_type='instance', use_dropout=False,
                         use_se_block=False):
    """
    Instantiates the generator and discriminator
    :param type: The type of network to instantiate (dh, unet, or resnet)
    :return: The generator and discriminator
    """

    # grayscale images, 3 for RGB images
    gen_in_channels = 1
    # to generate grayscale restored images 
    gen_out_channels = 1

    norm_layer = get_norm_layer(norm_type=norm_type)

    # Instantiate the generator with the specified channel configurations
    if type == 'dh':
        generator = DHadadGenerator(gen_in_channels, gen_out_channels).to(device)
    elif type == 'unet_128':
        generator = UnetGenerator(
            gen_in_channels, gen_out_channels, 7, norm_layer=norm_layer, use_dropout=True
        ).to(device)
    elif type == 'unet_256':
        generator = UnetGenerator(
            gen_in_channels,
            gen_out_channels,
            num_downs=8,
            ngf=ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout
        ).to(device)
    elif type == 'unet_512':
        generator = UnetGenerator(
            gen_in_channels,
            gen_out_channels,
            num_downs=num_downs,
            ngf=ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout
        ).to(device)

        generator.apply(generator.initialize_weights)
    elif type == 'resnet':
        pass
    else:
        print(f"Error: Unknown network type {type}")
        return None

    # Specify the input channel configurations (it typically takes two inputs)
    # we assume we're providing pairs of images (intact and restored) as input
    disc_in_channels = 1

    discriminator = DHadadDiscriminator(
        disc_in_channels,
        filter_sizes=[64, 128, 256, 512, 512, 512, 1024],
        use_dropout=use_dropout,
        use_se_block=use_se_block
    ).to(device)

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
    gen_optim = Adam(generator.parameters(), lr=generator_lr, betas=(0.5, 0.999), amsgrad=True)
    dis_optim = Adam(discriminator.parameters(), lr=discriminator_lr, betas=(0.5, 0.999), amsgrad=True)

    return gen_optim, dis_optim


def init_schedulers(gen_optim, dis_optim, opt="plateau"):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions.
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """

    if opt == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l

        gen_scheduler = lr_scheduler.LambdaLR(gen_optim, lr_lambda=lambda_rule)
    elif opt == 'step':
        gen_scheduler = lr_scheduler.StepLR(gen_optim, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt == 'plateau':
        gen_scheduler = lr_scheduler.ReduceLROnPlateau(gen_optim, mode='min', factor=0.2, threshold=0.01, patience=5)
        dis_scheduler = lr_scheduler.ReduceLROnPlateau(dis_optim, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt == 'cosine':
        gen_scheduler = lr_scheduler.CosineAnnealingLR(gen_optim, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)

    return gen_scheduler, dis_scheduler


def train_discriminator_step(discriminator, dis_optim, damaged_dm, enhanced_dm, fake_dm):
    """
        Update Discriminator Step
    """

    for _ in range(critic_updates_per_gen_update):
        dis_optim.zero_grad()

        # Classify the real image 
        # Concatenate the inputs along the channel dimension
        combined_damaged_enhanced = torch.cat([damaged_dm, enhanced_dm], dim=1)
        output_real = discriminator(combined_damaged_enhanced)
        real_labels = torch.ones_like(output_real, device=device)

        # Classify fake image
        combined_damaged_fake = torch.cat([damaged_dm, fake_dm], dim=1)
        output_fake = discriminator(combined_damaged_fake)
        fake_labels = torch.zeros_like(output_fake, device=device)

        # Compute gradient penalty
        gradient_penalty = loss_functions.compute_gradient_penalty(discriminator, damaged_dm, fake_dm,
                                                                   enhanced_dm) * lambda_gp

        # Compute the total discriminator loss
        real_loss = loss_functions.adversarial_loss(output_real, real_labels)
        fake_loss = loss_functions.adversarial_loss(output_fake, fake_labels)
        dis_loss = (real_loss + fake_loss) / 2 + lambda_gp * gradient_penalty

        # print_gradients(discriminator, "Discriminator gradients before backward:")

        # Compute gradients
        dis_loss.backward()

        # print_gradients(discriminator, "Discriminator gradients after backward:")

        # Clip gradients for discriminator
        clip_grad_norm_(discriminator.parameters(), max_norm=max_grad_norm)

        # Optimize
        dis_optim.step()

    return dis_loss


def train_generator_step(generator, gen_optim, fake_dm, real_dm, fake_for_gen, misleading_labels):
    """
        Update Generator Step 
    """
    gen_optim.zero_grad()

    l1_loss = loss_functions.l1_loss(fake_dm, real_dm)
    adv_loss = loss_functions.adversarial_loss(fake_for_gen, misleading_labels)
    ssim_loss = loss_functions.ssim_loss(fake_dm, real_dm)
    sharp_loss = loss_functions.sharpness_loss(fake_dm, real_dm)
    geom_loss = loss_functions.geometric_consistency_loss(fake_dm, real_dm)

    gen_loss = (
            loss_weights.weights['l1'] * l1_loss +
            loss_weights.weights['ssim'] * ssim_loss +
            loss_weights.weights['adversarial'] * adv_loss +
            loss_weights.weights['geometric'] * geom_loss +
            loss_weights.weights['sharp'] * sharp_loss
    )

    # Check if gen_loss is None
    if gen_loss is None:
        raise ValueError("Generator loss (gen_loss) is None. Check the loss calculation.")

    # Check if gen_loss is a scalar
    if not torch.is_tensor(gen_loss) or gen_loss.dim() != 0:
        raise ValueError("Generator loss (gen_loss) is not a scalar. It must be a 0-dimensional tensor.")

    gen_loss.backward()

    # Clip gradients for generator
    clip_grad_norm_(generator.parameters(), max_norm=max_grad_norm)

    # Optimize
    gen_optim.step()

    return gen_loss, [l1_loss, ssim_loss, adv_loss, geom_loss, sharp_loss, sharp_loss]


def train_step(generator, gen_optim, discriminator, dis_optim, train_dataloader):
    """
    Training step for the generator and discriminator
        Training step for each batch
    data is a list of [real, synthetic] images
    real_dms are the ground truth and estimated displacement maps
    synthetic_dms are the procedurally enhanced displacement maps

    :param generator: The generator network
    :param gen_optim: The generator optimizer
    :param discriminator: The discriminator network
    :param dis_optim: The discriminator optimizer
    :param train_dataloader: The training dataset
    """

    # Increment the counter and log every 100 pairs
    counter = 0
    start_time = time.time()

    generator.train()
    discriminator.train()

    for i, (damaged_dm, enhanced_dm) in enumerate(train_dataloader):
        # Move images to GPU
        damaged_dm = damaged_dm.to(device)
        enhanced_dm = enhanced_dm.to(device)

        # Generate fake data for discriminator training and detach it
        fake_dm = generator(damaged_dm).detach()
        dis_loss = train_discriminator_step(discriminator, dis_optim, damaged_dm, enhanced_dm, fake_dm)

        # Generate fake data for generator training (do not detach)
        fake_dm_for_gen = generator(damaged_dm)

        # # Freeze discriminator's parameters
        for param in discriminator.parameters():
            param.requires_grad = False

        combined_damaged_fake = torch.cat([damaged_dm, fake_dm], dim=1)
        fake_for_gen = discriminator(combined_damaged_fake)
        misleading_labels = torch.ones_like(fake_for_gen, device=device)
        gen_loss, loss_components = train_generator_step(generator, gen_optim, fake_dm_for_gen, enhanced_dm,
                                                         fake_for_gen, misleading_labels)

        # Unfreeze discriminator's parameters after generator training
        for param in discriminator.parameters():
            param.requires_grad = True

        counter += 1

        # Log every 500 pairs
        if counter % 50 == 0:
            # print(f"Processed image pair:            {damaged_dm}, {enhanced_dm}")
            print(f"Number of image pairs processed: {counter}")

            # Calculate the time per image pair and log it with two decimal places
            time_per_image_pair = (time.time() - start_time) / counter
            print(f"Time per image pair:             {time_per_image_pair:.2f} seconds")

    return gen_loss, loss_components, dis_loss


def validation_step(generator, discriminator, val_dataloader, psnrs, ssims, esis, max_grad_norm):
    """
    Validation step after each epoch
    """
    # Switch to evaluation mode
    generator.eval()
    discriminator.eval()

    with torch.no_grad():
        for damaged_dm, preserved_dm in val_dataloader:
            damaged_dm = damaged_dm.to(device)
            preserved_dm = preserved_dm.to(device)
            enhanced_dm = generator(damaged_dm)

            # Clamp the values to be between 0 and 1
            enhanced_dm = enhanced_dm.clamp(0, 1)
            synthetic_dm = preserved_dm.clamp(0, 1)

            # Compute PSNR, SSIM, ESI for the current batch
            batch_psnr = dh_metrics.compute_psnr(enhanced_dm, preserved_dm)
            batch_ssim = dh_metrics.compute_ssim(enhanced_dm, preserved_dm)
            batch_esi = dh_metrics.compute_edge_similarity(enhanced_dm, preserved_dm)

            # Update epoch metrics
            psnrs.append(batch_psnr)
            ssims.append(batch_ssim)
            esis.append(batch_esi)

    # Switch back to training mode
    generator.train()
    discriminator.train()


def calculate_performance_metrics(epoch, best_psnr, psnrs, best_ssim, ssims, best_esi, esis,
                                  epoch_time, gen_loss, loss_components, dis_loss, gen_optim, dis_optim):
    """
    Calculate the performance metrics for the current epoch
    """
    # Move tensors to CPU and convert to float
    psnrs_cpu = [x.cpu().item() if torch.is_tensor(x) else x for x in psnrs]
    ssims_cpu = [x.cpu().item() if torch.is_tensor(x) else x for x in ssims]
    esis_cpu = [x.cpu().item() if torch.is_tensor(x) else x for x in esis]

    # Calculate min and max values
    min_psnr, max_psnr = np.min(psnrs_cpu), np.max(psnrs_cpu)
    min_ssim, max_ssim = np.min(ssims_cpu), np.max(ssims_cpu)
    min_esi, max_esi = np.min(esis_cpu), np.max(esis_cpu)

    # Calculate standard deviation
    std_psnr, std_ssim, std_esi = np.std(psnrs_cpu), np.std(psnrs_cpu), np.std(esis_cpu)

    # Average metrics for this epoch
    avg_psnr, avg_ssim, avg_esi = np.mean(psnrs_cpu), np.mean(ssims_cpu), np.mean(esis_cpu)

    # Calculate combined score
    avg_combined_score = dh_metrics.combined_score(avg_psnr, avg_ssim, avg_esi)

    # Check if the average PSNR for this epoch is higher than the best seen so far
    is_psnr_improved = avg_psnr >= best_psnr * (1 + improvement_threshold)
    is_ssim_improved = avg_ssim >= best_ssim * (1 + improvement_threshold)
    is_esi_improved = avg_esi >= best_esi * (1 + improvement_threshold)

    # Performance metrics
    performance_metrics = {
        'psnr': {'value': avg_psnr, 'improved': is_psnr_improved, 'magnitude': abs(avg_psnr - best_psnr)},
        'ssim': {'value': avg_ssim, 'improved': is_ssim_improved, 'magnitude': abs(avg_ssim - best_ssim)},
        'esi': {'value': avg_esi, 'improved': is_esi_improved, 'magnitude': abs(avg_esi - best_esi)}
    }

    # Logging for each epoch 
    dh_metrics.print_performance_metrics(epoch, num_epochs, epoch_time, loss_weights,
                                         avg_psnr, min_psnr, max_psnr, std_psnr,
                                         avg_ssim, min_ssim, max_ssim, std_ssim,
                                         avg_esi, min_esi, max_esi, std_esi,
                                         avg_combined_score, performance_metrics,
                                         gen_loss, loss_components, dis_loss, gen_optim, dis_optim)

    return performance_metrics, avg_combined_score


def save_model(generator, epoch, loss_weights):
    """
    Save the model checkpoint
    """
    MODEL_NAME = f"dh_{weights_type}_model_ep_{epoch}{loss_weights.get_weights_as_string()}.pth"

    print(f"-> Saving model checkpoint at epoch {epoch}")

    torch.save(generator.state_dict(), os.path.join(MODEL_PATH, MODEL_NAME))


# Function to print the gradients of the first layer of the model
def print_gradients(model, message):
    print(message)

    # Assuming the first layer of the model is a Conv2d layer. Adjust if your model differs.
    first_layer = list(model.modules())[1]

    for name, param in first_layer.named_parameters():
        if param.requires_grad:
            # param.grad can be None if the gradient has not been computed yet
            print(f"Layer: {name}, Gradient: {param.grad if param.grad is not None else 'No gradient computed yet'}")


########################################################################################
# Training Loop
########################################################################################
def network_training(
        generator, discriminator,
        gen_optim, dis_optim,
        num_epochs, current_epoch=0,
        train_dataset_size=30, val_dataset_size=6):
    # Learning Rate Scheduling
    gen_scheduler, dis_scheduler = init_schedulers(gen_optim, dis_optim, opt="plateau")

    # Initialize some variables for averaging
    best_psnr = -float('inf')
    best_ssim = -float('inf')
    best_esi = -float('inf')

    epochs_no_improve = 0  # Counter for epochs without improvement
    psnrs, ssims, esis = [], [], []  # Lists to store all PSNR and SSIM values for each epoch
    epoch_times = []  # List to store time taken for each epoch

    # Enable anomaly detection for debugging
    torch.autograd.set_detect_anomaly(True)

    train_dataset_size = train_dataset_size
    val_dataset_size = val_dataset_size

    # Load the dataset
    train_dataloader, val_dataloader = load_dataset(
        train_size=train_dataset_size,
        val_size=val_dataset_size,
        save_images=False)

    # Training loop for each epoch. if the current epoch is 0, it will start from the beginning, if not,
    # it will start from the current epoch
    for epoch in range(current_epoch, num_epochs):
        # Start time for this epoch
        start_time = time.time()

        # Training step
        gen_loss, loss_components, dis_loss = train_step(
            generator, gen_optim, discriminator, dis_optim, train_dataloader)

        # Validation step
        validation_step(generator, discriminator, val_dataloader, psnrs, ssims, esis, max_grad_norm)

        # Calculate time taken for this epoch
        epoch_time = time.time() - start_time

        # Store for future analysis
        epoch_times.append(epoch_time)

        # Calculate performance metrics for this epoch
        performance_metrics, avg_combined_score = calculate_performance_metrics(
            (epoch + 1), best_psnr, psnrs, best_ssim, ssims, best_esi, esis,
            epoch_time, gen_loss.item(), loss_components, dis_loss.item(), gen_optim, dis_optim)

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
        if save_checkpoint == True or (epoch + 1) % checkpoint_interval == 0:
            save_model(generator, (epoch + 1), loss_weights)

        # Early stopping if there's no improvement in the average PSNR, SSIM, or ESI for a certain number of epochs
        if save_checkpoint == True:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Update loss weights at the end of each epoch
        loss_weights.manage_epoch_weights(epoch + 1)

        # Break if there's no improvement for a certain number of epochs
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} due to no improvement.")
            break

        # Step the learning rate scheduler based on the average combined score
        gen_scheduler.step(gen_loss)
        dis_scheduler.step(dis_loss)

        # Reset for next epoch 
        psnrs.clear()
        ssims.clear()
        esis.clear()

        train_dataset_size += 10
        val_dataset_size += 6

    # save the model
    save_model(generator, (epoch + 1), loss_weights)

    print(f"Average time per epoch: {np.mean(epoch_times):.2f}s")
    print(f"Final Loss Weights:     {loss_weights.weights}")


def load_dataset(save_images=False, train_size=3, val_size=2):
    """
    Loads the training dataset
    :return: The dataloaders for both training dataset and validation dataset
    """
    dataset_generator = SyntheticDatasetGenerator(
        displacement_maps_path=DISPLACEMENT_MAPS_PATH,
        crack_d_map_dataset_path=CRACK_D_MAP_DATASET_PATH,
        input_training_dataset_path=INPUT_TRAINING_DATASET_PATH,
        target_training_dataset_path=TARGET_TRAINING_DATASET_PATH
    )

    train_dataset = dataset_generator.generate_synthetic_input_target_pairs(
        size=train_size,
        save_dataset=save_images)

    val_dataset = dataset_generator.generate_synthetic_input_target_pairs(
        size=val_size,
        save_dataset=save_images)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_dataloader, val_dataloader


########################################################################################
# Main
# - Load displacement maps
# - Generate synthetic displacement maps
# - Validate the generated data
########################################################################################
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Instantiate the generator and discriminator
    generator, discriminator = instantiate_networks(
        type='unet_512',
        num_downs=7,
        ngf=128,
        norm_type='instance',
        use_dropout=False
    )

    # Load the model weights if available
    generator = load_model_weights(generator, MODEL_WEIGHTS_PATH)

    # Initialize optimizers
    gen_optim, dis_optim = init_optimizer(generator, discriminator)

    # Train the network
    network_training(
        generator, discriminator,
        gen_optim, dis_optim,
        num_epochs,
        current_epoch=27,
        train_dataset_size=2,
        val_dataset_size=2)

    print("Done!")
