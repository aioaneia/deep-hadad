import functools
import logging
import os
import time
import numpy as np
import torch

from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import RAdam
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from models.DHadadLossFunctions import DHadadLossFunctions
from models.DHadadLossWeights import DHadadLossWeights

from models.DeepHadadDiscriminator import DeepHadadDiscriminator
from models.DHadadGenerator import DHadadGenerator

from utils.SyntheticDataset import SyntheticDataset

from generation.synthetic_glyph_generation import SyntheticDatasetGenerator

import utils.performance_metrics as dh_metrics

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA device found.")
elif torch.backends.mps.is_available():
    # Regular cache cleanup
    device = torch.device("mps")
    print("MPS device found.")
else:
    device = torch.device("cpu")
    print("No CUDA or MPS device found, using CPU.")

# scaler = torch.amp.GradScaler('mps', enabled=False)  # Disable AMP

# PyTorch version
print("PyTorch version: " + torch.__version__)

#### Set hyperparameter values for training ####
generator_lr          = 0.0003
discriminator_lr      = 0.00015
batch_size            = 6
num_epochs            = 150
checkpoint_interval   = 3
max_grad_norm         = 1.0
patience              = 100
improvement_threshold = 0.004  # represents
lambda_gp             = 10     # The gradient penalty coefficient

# Set the project paths
PROJECT_PATH            = './'
TRAINING_DATASET_PATH   = PROJECT_PATH + 'data/training_dataset'
X_TRAINING_DATASET_PATH = TRAINING_DATASET_PATH + '/X'
Y_TRAINING_DATASET_PATH = TRAINING_DATASET_PATH + '/Y'

DISPLACEMENT_MAPS_PATH       = PROJECT_PATH + 'data/glyphs_dataset/preserved_glyphs/displacement_maps/'
DISPLACEMENT_MAPS_PATH_LARGE = PROJECT_PATH + 'data/glyphs_dataset/preserved_glyphs/displacement_maps_large_size/'
CRACKS_DATASET_PATH          = PROJECT_PATH + 'data/cracks_dataset/'
MASKS_DATASET_PATH           = PROJECT_PATH + 'data/masks_dataset/'
INPUT_TRAINING_DATASET_PATH  = PROJECT_PATH + 'data/training_dataset/X/'
TARGET_TRAINING_DATASET_PATH = PROJECT_PATH + 'data/training_dataset/Y/'

MODEL_PATH         = PROJECT_PATH + 'trained_models/'
MODEL_WEIGHTS_PATH = MODEL_PATH + 'dh_model_ep_19_l8.00_s2.00_g3.00_f0.80_c1.00_a0.10.pth'

IMAGE_EXTENSIONS = [".png", ".jpg", ".tif"]

# Dynamic Loss Weights for adjusting the loss weights during training
loss_weights = DHadadLossWeights(total_epochs=num_epochs)

# Initialize loss functions with the device
loss_functions = DHadadLossFunctions(device=device)


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer"""
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


def instantiate_networks(ngf=64, n_down_sampling=3, n_blocks=6):
    """Instantiates the generator and discriminator"""
    # grayscale images, 3 for RGB images
    gen_in_channels  = 1
    gen_out_channels = 1

    generator = DHadadGenerator(
        input_nc       = 1,
        output_nc      = 1,
        ngf            = ngf,               # Number of generator filters
        n_downsampling = n_down_sampling,   # Number of down_sampling layers
         n_blocks       = n_blocks          # Number of ResBlocks
    ).to(device)

    discriminator = DeepHadadDiscriminator(
        input_nc=2,
        ndf=ngf,
        n_layers=n_down_sampling + 1
    ).to(device)

    return generator, discriminator


def load_model_weights(model, model_path):
    """
    Loads the model weights from the specified path
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


def init_optimizer(generator, discriminator):
    # Initialize optimizers
    gen_optim = RAdam(generator.parameters(), lr=generator_lr, betas=(0.5, 0.999))
    dis_optim = RAdam(discriminator.parameters(), lr=discriminator_lr, betas=(0.5, 0.999))

    return gen_optim, dis_optim


def init_schedulers(gen_optim, dis_optim, opt="plateau"):
    if opt == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l

        gen_scheduler = lr_scheduler.LambdaLR(
            gen_optim,
            lr_lambda=lambda_rule
        )
    elif opt == 'step':
        gen_scheduler = lr_scheduler.StepLR(
            gen_optim,
            step_size=opt.lr_decay_iters,
            gamma=0.1
        )
    elif opt == 'plateau':
        gen_scheduler = lr_scheduler.ReduceLROnPlateau(
            gen_optim,
            mode='min',
            factor=0.5,
            threshold=0.01,
            patience=10
        )
        dis_scheduler = lr_scheduler.ReduceLROnPlateau(
            dis_optim,
            mode='min',
            factor=0.5,
            threshold=0.01,
            patience=10
        )
    elif opt == 'cosine':
        gen_scheduler = lr_scheduler.CosineAnnealingLR(
            gen_optim,
            T_max=150,
            eta_min=1e-6
        )

        dis_scheduler = lr_scheduler.CosineAnnealingLR(
            dis_optim,
            T_max=150,
            eta_min=1e-6
        )
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)

    return gen_scheduler, dis_scheduler


def train_discriminator_step(discriminator, damaged_dm, preserved_dm, fake_dm):
    """
    Training step for the discriminator
    damage_mask = torch.abs(damaged_dm - preserved_dm) > threshold
    """
    combined_real = torch.cat([damaged_dm, preserved_dm], dim=1)
    combined_fake = torch.cat([damaged_dm, fake_dm.detach()], dim=1)

    real_pred = discriminator(combined_real)
    fake_pred = discriminator(combined_fake)

    adv_loss = torch.mean(fake_pred) - torch.mean(real_pred)

    # Compute gradient penalty
    gradient_penalty = loss_functions.compute_gradient_penalty(discriminator, damaged_dm, fake_dm, preserved_dm)

    dis_loss = adv_loss + lambda_gp * gradient_penalty

    # Ensure the loss is a scalar
    if not isinstance(dis_loss, torch.Tensor) or dis_loss.numel() > 1:
        dis_loss = dis_loss.mean()

    return dis_loss


def safe_item(tensor):
    if isinstance(tensor, torch.Tensor):
        if tensor.numel() == 1:
            return tensor.item()
        else:
            return tensor.mean().item()
    return tensor


def train_generator_step(fake_dm, preserved_dm, dis_pred_fake):
    """Training step for the generator"""
    l1_loss   = loss_functions.l1_loss(fake_dm, preserved_dm)
    ssim_loss = loss_functions.ssim_loss(fake_dm, preserved_dm)
    gdl       = loss_functions.gradient_loss(fake_dm, preserved_dm)
    adv_loss  = -torch.mean(dis_pred_fake)

    gen_loss = (
            loss_weights.current_weights['l1']   * l1_loss +
            loss_weights.current_weights['ssim'] * ssim_loss +
            loss_weights.current_weights['gdl']  * gdl +
            loss_weights.current_weights['adv']  * adv_loss
    )

    loss_dict = {
        'L1':          safe_item(l1_loss),
        'SSIM':        safe_item(ssim_loss),
        'GDL':         safe_item(gdl),
        'Adversarial': safe_item(adv_loss),
    }

    return gen_loss, loss_dict


def train_step(generator, gen_optim, discriminator, dis_optim, train_dataloader, accumulation_steps=2):
    """Training step for the generator and discriminator"""
    generator.train()
    discriminator.train()

    start_time = time.time()
    total_gen_loss = 0
    total_dis_loss = 0
    accumulated_loss_components = None
    batch_count = 0
    dis_updates = 0
    n_critic = 3

    for i, (damaged_dm, preserved_dm) in enumerate(train_dataloader):
        batch_count += 1
        damaged_dm = damaged_dm.clamp(0, 1).to(device)
        preserved_dm = preserved_dm.clamp(0, 1).to(device)

        if torch.isnan(damaged_dm).any() or torch.isnan(preserved_dm).any():
            print("NaN detected in inputs to the generator!")
            print(f"Skipping batch {i} due to NaN values in input")
            continue

        # Generate fake samples
        with torch.no_grad():
            fake_dm = generator(damaged_dm)

        if torch.isnan(fake_dm).any():
            print("NaN detected in fake depth map!")
            print(f"Skipping batch {i} due to NaN values in fake depth map")
            continue

        # ----- Discriminator Training -----
        dis_batch_loss = 0
        for _ in range(n_critic):
            dis_optim.zero_grad()

            dis_loss = train_discriminator_step(discriminator, damaged_dm, preserved_dm, fake_dm.detach())

            dis_loss = dis_loss / accumulation_steps

            if torch.isnan(dis_loss).any():
                print("NaN detected in discriminator loss!")
                continue

            dis_loss.backward()
            dis_batch_loss += dis_loss.item()
            dis_updates += 1

        # Average the discriminator loss across n_critic updates
        dis_batch_loss = dis_batch_loss * accumulation_steps / n_critic
        total_dis_loss += dis_batch_loss

        # Perform optimization step every accumulation_steps
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_dataloader):
            clip_grad_norm_(discriminator.parameters(), max_norm=max_grad_norm)
            dis_optim.step()

        # ----- Generator Training (less frequent) -----
        gen_optim.zero_grad()

        # Generate new fake samples for generator training
        fake_dm = generator(damaged_dm)

        # Freeze discriminator for generator training
        for param in discriminator.parameters():
            param.requires_grad = False

        # Get discriminator predictions for generator training
        dis_pred_fake = discriminator(torch.cat([damaged_dm, fake_dm], dim=1))

        # Train generator
        gen_loss, loss_components = train_generator_step(fake_dm, preserved_dm, dis_pred_fake)

        # Scale loss if accumulating gradients
        gen_loss = gen_loss / accumulation_steps

        if torch.isnan(gen_loss).any():
            print("NaN detected in generator loss!")

        gen_loss.backward()

        total_gen_loss += gen_loss.item() * accumulation_steps

        # Unfreeze discriminator for its training
        for param in discriminator.parameters():
            param.requires_grad = True

        # Accumulate loss components
        if accumulated_loss_components is None:
            accumulated_loss_components = {k: v for k, v in loss_components.items()}
        else:
            for k in accumulated_loss_components:
                accumulated_loss_components[k] += loss_components[k]

        # Perform optimization step every accumulation_steps
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_dataloader):
            # Gradient clipping
            clip_grad_norm_(generator.parameters(), max_norm=max_grad_norm)
            gen_optim.step()

        # Log progress
        if (i + 1) % (accumulation_steps * 50) == 0:
            print(f"Processed {i + 1}/{len(train_dataloader)} batches")
            time_per_batch = (time.time() - start_time) / (i + 1)
            print(f"Time per batch: {time_per_batch:.2f} seconds")

    # Calculate average losses
    avg_gen_loss = total_gen_loss / batch_count
    avg_dis_loss = total_dis_loss / batch_count
    avg_loss_components = {k: v / batch_count for k, v in accumulated_loss_components.items()}

    return avg_gen_loss, avg_loss_components, avg_dis_loss


def validation_step(generator, discriminator, val_dataloader, psnrs, ssims, esis):
    """
    Validation step after each epoch
    """
    generator.eval()
    discriminator.eval()

    with torch.no_grad():
        for damaged_dm, preserved_dm in val_dataloader:
            damaged_dm = damaged_dm.clamp(0, 1).to(device)
            preserved_dm = preserved_dm.clamp(0, 1).to(device)

            fake_dm = generator(damaged_dm)

            # Compute PSNR, SSIM, ESI for the current batch
            batch_psnr = dh_metrics.compute_psnr(fake_dm, preserved_dm)
            batch_ssim = dh_metrics.compute_ssim(fake_dm, preserved_dm)
            batch_esi = dh_metrics.compute_edge_similarity(fake_dm, preserved_dm)

            # Update epoch metrics
            psnrs.append(batch_psnr)
            ssims.append(batch_ssim)
            esis.append(batch_esi)

    # Switch back to training mode
    generator.train()
    discriminator.train()


def calculate_performance_metrics(epoch, best_psnr, psnrs, best_ssim, ssims, best_esi, esis,
                                  epoch_time, gen_loss, loss_components, dis_loss, gen_optim, dis_optim):
    """Calculate the performance metrics for the current epoch"""
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
    """Save the model checkpoint"""
    MODEL_NAME = f"dh_model_ep_{epoch}{loss_weights.get_weights_as_string()}.pth"

    print(f"-> Saving model checkpoint at epoch {epoch}")

    torch.save(generator.state_dict(), os.path.join(MODEL_PATH, MODEL_NAME))


########################################################################################
# Training Loop
########################################################################################
def network_training(generator, discriminator, gen_optim, dis_optim,
                     num_epochs, current_epoch=0, train_dataset_size=30, val_dataset_size=6,
                     image_sizes=None):
    # Learning Rate Scheduling
    gen_scheduler, dis_scheduler = init_schedulers(gen_optim, dis_optim, opt="cosine")

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
    val_dataset_size   = val_dataset_size

    # Load the dataset
    train_dataloader, val_dataloader = load_dataset(
        DISPLACEMENT_MAPS_PATH,
        train_size=train_dataset_size,
        val_size=val_dataset_size,
        save_images=False,
        image_sizes=image_sizes
    )

    large_train_dataloader, large_val_dataloader = load_dataset(
        DISPLACEMENT_MAPS_PATH_LARGE,
        train_size=train_dataset_size,
        val_size=val_dataset_size,
        save_images=False,
        image_sizes=[(512, 512)]
    )

    for epoch in range(current_epoch, num_epochs):
        start_time = time.time()

        gen_loss, loss_components, dis_loss = train_step(generator, gen_optim, discriminator,
                                                         dis_optim, train_dataloader)

        gen_loss_2, loss_components_2, dis_loss_2 = train_step(generator, gen_optim, discriminator,
                                                         dis_optim, large_train_dataloader)

        validation_step(generator, discriminator, val_dataloader, psnrs, ssims, esis)

        epoch_time = time.time() - start_time

        epoch_times.append(epoch_time)

        # Calculate performance metrics for this epoch
        performance_metrics, avg_combined_score = calculate_performance_metrics(
            (epoch + 1), best_psnr, psnrs, best_ssim, ssims, best_esi, esis,
            epoch_time, gen_loss, loss_components, dis_loss, gen_optim, dis_optim)

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

        # Break if there's no improvement for a certain number of epochs
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} due to no improvement.")
            break

        # Step the learning rate scheduler based on the average combined score
        gen_scheduler.step(gen_loss)
        dis_scheduler.step(dis_loss)

        psnrs.clear()
        ssims.clear()
        esis.clear()

        # Adjust loss weights based on the epoch and model performance
        loss_weights.manage_epoch_weights(epoch)

    # save the model
    save_model(generator, (epoch + 1), loss_weights)

    print(f"Average time per epoch: {np.mean(epoch_times):.2f}s")
    print(f"Final Loss Weights:     {loss_weights.current_weights}")


def load_dataset(dm_maps_path, save_images=False, train_size=3, val_size=2, image_sizes=None):
    """Loads the training dataset"""
    if image_sizes is None:
        image_sizes = [(256, 256)]

    dataset_generator = SyntheticDatasetGenerator(
        displacement_maps_path       = dm_maps_path,
        cracks_dataset_path          = CRACKS_DATASET_PATH,
        masks_dataset_path           = MASKS_DATASET_PATH,
        input_training_dataset_path  = INPUT_TRAINING_DATASET_PATH,
        target_training_dataset_path = TARGET_TRAINING_DATASET_PATH
    )

    combined_train_inputs = []
    combined_train_targets = []

    combined_val_inputs = []
    combined_val_targets = []

    for image_size in image_sizes:
        train_dataset: SyntheticDataset = dataset_generator.generate_synthetic_input_target_pairs(
            dataset_size=train_size,
            image_size=image_size,
            save_dataset=save_images
        )

        val_dataset: SyntheticDataset = dataset_generator.generate_synthetic_input_target_pairs(
            dataset_size=val_size,
            image_size=image_size,
            save_dataset=save_images
        )

        # Combine synthetic and real training datasets
        combined_train_inputs  += train_dataset.input_images
        combined_train_targets += train_dataset.target_images

        combined_val_inputs  += val_dataset.input_images
        combined_val_targets += val_dataset.target_images

    train_dataset = SyntheticDataset(combined_train_inputs, combined_train_targets)

    val_dataset = SyntheticDataset(combined_val_inputs, combined_val_targets)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

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
        ngf=64,
        n_down_sampling=3,
        n_blocks=9
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
        current_epoch      = 0,
        train_dataset_size = 5,
        val_dataset_size   = 3,
        image_sizes         = [(256, 256)]
    )

    print("Done!")
