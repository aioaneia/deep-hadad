import os

# This guide can only be run with the torch backend.
os.environ["KERAS_BACKEND"] = "torch"

import sys
import glob
import cv2

import torch
import keras

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from torchvision import transforms

from keras              import Input, Model
from keras.layers       import Conv2D, Dropout, Conv2DTranspose, LeakyReLU, BatchNormalization, Concatenate, Activation, Flatten
from keras.initializers import RandomNormal
from keras.optimizers   import Adam
from keras.metrics      import Mean
from keras.losses       import BinaryCrossentropy, MeanAbsoluteError

import keras.backend as K

from attention_mechanisms.SelfAttention import SelfAttention
from models.ResidualBlock                 import ResidualBlock
from attention_mechanisms.CBAM          import CBAM

from PIL import Image

from sklearn.model_selection import train_test_split

# Add the project directory to the Python path
sys.path.append('./')

from utils.DisplacementMapDataset  import DisplacementMapDataset


# PyTorch version
print("PyTorch version: " + torch.__version__)
print("Keras version:   " + keras.__version__)

# Set hyperparameter values for training 
generator_lr        = 2e-5
discriminator_lr    = 3e-5
batch_size          = 1
num_epochs          = 100
checkpoint_interval = 3
max_grad_norm       = 1.0
patience            = 50
improvement_threshold = 0.004  # 1% improvement
lambda_gp           = 10    # The gradient penalty coefficient
weights_type        = 'depth' #

# Set the project paths
PROJECT_PATH            = './'
TRAINING_DATASET_PATH   = PROJECT_PATH + 'data/small_training_dataset'
X_TRAINING_DATASET_PATH = TRAINING_DATASET_PATH + '/X'
Y_TRAINING_DATASET_PATH = TRAINING_DATASET_PATH + '/Y'
MODEL_PATH              = PROJECT_PATH + 'models/'
MODEL_WEIGHTS_PATH      = MODEL_PATH + 'dh_depth_model_ep_2_r1.00_p1.10_a0.60_g1.50_d1.20_s1.00.pth'

IMAGE_EXTENSIONS        = [".png", ".jpg", ".tif"]



def encoder_block(layer_in, n_filters, batch_norm=True, activation=LeakyReLU(0.2)):
    """
        Encoder block with optional batch normalization and leaky ReLU activation.
    """

    # weight initialization
    init = RandomNormal(stddev=0.02)

    # add downsampling layer
    g = Conv2D(n_filters, (4,4), strides=(2,2), padding = 'same', kernel_initializer=init)(layer_in)

    # conditionally add batch normalization
    if batch_norm:
        g = BatchNormalization()(g, training=True)

    # leaky relu activation
    g = activation(g)

    return g

# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout = True, activation = Activation('relu')):
    """
        Decoder block with optional dropout, batch normalization and ReLU activation.
    """

    # weight initialization
    init = RandomNormal(stddev=0.02)

    # add upsampling layer
    g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    
    # add batch normalization
    g = BatchNormalization()(g, training=True)
        
    # conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)
        
    # merge with skip connection
    g = Concatenate()([g, skip_in])

    # relu activation
    g = activation(g)

    return g


def get_generator(image_shape, resize_factor = 1.0, dropout_prob = 0.2, filter_sizes=[16, 32, 64, 96, 128, 256, 384, 512]):
    """
        Generator model with skip connections.
    """

    # weight initialization
    init = RandomNormal(stddev=0.02)

    # Add Self-Attention layers
    self_attention_128  = SelfAttention(filter_sizes[4]) # 128
    self_attention_256  = SelfAttention(filter_sizes[5]) # 256
    self_attention_512  = SelfAttention(filter_sizes[7]) # 512
    #self.self_attention_1024 = SelfAttention(filter_sizes[7]) # 1024
        
    # Image input
    in_image = Input(shape = image_shape) #512x512
    #in_image = Lambda(lambda x: K.reshape(x, (None, 512, 512, 1)))(in_image)

    # Encoder model
    e0 = encoder_block(in_image, int(filter_sizes[0] * resize_factor), batch_norm=False) #256x256
    e1 = encoder_block(e0, int(filter_sizes[1] * resize_factor)) #128x128
    e2 = encoder_block(e1, int(filter_sizes[2] * resize_factor)) #64x64
    e3 = encoder_block(e2, int(filter_sizes[3] * resize_factor)) #32x32
    e4 = encoder_block(e3, int(filter_sizes[4] * resize_factor)) #16x16
    e5 = encoder_block(e4, int(filter_sizes[5] * resize_factor)) #8x8
    e6 = encoder_block(e5, int(filter_sizes[6] * resize_factor)) #4x4

        # if idx > 2:
        #         # Adding a residual block for encoder
        #         self.res_blocks_enc.append(ResidualBlock(filter_size))
    
        #         # Add CBAM module after each encoder layer
        #         self.cbam_blocks_enc.append(CBAM(filter_size))

    # bottleneck, no batch norm and relu
    b = Conv2D(int(filter_sizes[7]/resize_factor), (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e6) #1x1
    b = Activation('relu')(b)

    #reversed_filter_sized = list(reversed(filter_sizes))
    # Adding residual blocks for decoder
    #self.res_blocks_dec.append(ResidualBlock(reversed_filter_sized[idx + 1]))
    # Add CBAM module
    #self.cbam_blocks_dec.append(CBAM(reversed_filter_sized[idx + 1]))
        
    # Decoder model
    d1 = decoder_block(b, e6,  int(filter_sizes[6] * resize_factor))
    d2 = decoder_block(d1, e5, int(filter_sizes[5] * resize_factor))
    d3 = decoder_block(d2, e4, int(filter_sizes[4] * resize_factor))
    d4 = decoder_block(d3, e3, int(filter_sizes[3] * resize_factor), dropout=False)
    d5 = decoder_block(d4, e2, int(filter_sizes[2] * resize_factor), dropout=False)
    d6 = decoder_block(d5, e1, int(filter_sizes[1] * resize_factor), dropout=False)
    d7 = decoder_block(d6, e0, int(filter_sizes[0] * resize_factor), dropout=False)

    # output
    g = Conv2DTranspose(1, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
    out_image = Activation('tanh')(g)

    # define model
    model = Model(in_image, out_image)

    return model


def get_discriminator(image_shape, resize_factor=1.0):
    """
        Discriminator model with PatchGAN classifier.
    """

    # weight initialization
    init = RandomNormal(stddev=0.02)
    
    # source image input
    in_src_image = Input(shape=image_shape)
    
    # target image input
    in_target_image = Input(shape=image_shape)
    
    # concatenate images channel-wise (along the width) into a single image called merged
    merged = Concatenate()([in_src_image, in_target_image])
    
    # C64
    d = Conv2D(int(64 * resize_factor), (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(negative_slope=0.2)(d)
     
    # C128
    d = Conv2D(int(128 * resize_factor), (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(negative_slope=0.2)(d)
     
    # C256
    d = Conv2D(int(256 * resize_factor), (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(negative_slope=0.2)(d)

    # C512
    d = Conv2D(int(512 * resize_factor), (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(negative_slope=0.2)(d)

    # second last output layer
    d = Conv2D(int(512 * resize_factor), (4, 4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(negative_slope=0.2)(d)

    # patch output
    d = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)

    # define model
    model = Model([in_src_image, in_target_image], patch_out)

    return model


class GAN(Model):
    def __init__(self, discriminator, generator):
        super().__init__()
        self.discriminator  = discriminator
        self.generator      = generator
        self.d_loss_tracker = Mean(name="d_loss")
        self.g_loss_tracker = Mean(name="g_loss")
        self.seed_generator = keras.random.SeedGenerator(1337)
        self.built = True


    @property
    def metrics(self):
        return [self.d_loss_tracker, self.g_loss_tracker]


    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super().compile()

        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn   = d_loss_fn
        self.g_loss_fn   = g_loss_fn


    def train_step(self, data):
        damaged_maps, real_maps = data

        # # Check if MPS (Multi-Process Service) is available
        # if torch.backends.mps.is_available():
        #     device = torch.device("mps")
        # else:
        #     device = torch.device("cpu")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convert numpy arrays to PyTorch tensors
        damaged_maps = torch.tensor(damaged_maps, dtype=torch.float32).to(device)  # Ensure dtype is float32
        real_maps = torch.tensor(real_maps, dtype=torch.float32).to(device)

        #print(f"Real Maps before permute: {real_maps.shape}")

        damaged_maps = damaged_maps.permute(0, 2, 3, 1)
        real_maps    = real_maps.permute(0, 2, 3, 1)
        
        #print(f"Real Maps after permute:  {real_maps.shape}")

        # Generate to fake images
        generated_maps = self.generator(damaged_maps).detach()

        # Discriminator output for real images
        # Ensure that gradients are being tracked for all parameters in the discriminator

        real_output = self.discriminator([damaged_maps, real_maps])
        real_labels = torch.ones_like(real_output, device=device)

        # Discriminator output for fake images
        fake_output = self.discriminator([damaged_maps, generated_maps])
        fake_labels = torch.zeros_like(fake_output, device=device)

        # Calculate loss for discriminator
        d_real_loss = self.d_loss_fn(real_labels, real_output)
        d_fake_loss = self.d_loss_fn(fake_labels, fake_output)
        d_loss      = (d_real_loss + d_fake_loss) / 2

        print(f"Discriminator requires_grad: {self.discriminator.requires_grad_}")
        print(f"Generator requires_grad:     {self.generator.requires_grad_}")
        print(f"Discriminator Loss is leaf:  {d_loss.is_leaf}")   
        
        # Zero out gradients for the next step
        self.d_optimizer.zero_grad()

        # Calculate gradients for discriminator
        d_loss.backward()

        grads = [param.grad for param in self.discriminator.parameters()]

        with torch.no_grad():
            self.d_optimizer.apply(grads, self.discriminator.trainable_weights)
        
        # Apply gradients
        self.d_optimizer.step()

        # Train the generator
            
        fake_output = self.discriminator([damaged_maps, generated_maps])
        
        misleading_labels = torch.ones_like(fake_output, device=device)

        g_loss = self.g_loss_fn(fake_output, misleading_labels, generated_maps, real_maps)

        # Zero out gradients for the next step
        self.g_optimizer.zero_grad()

        g_loss.backward()

        # Apply gradients
        self.g_optimizer.step()

        # Enable gradient tracking for discriminator
        for param in self.discriminator.parameters():
            param.requires_grad = True

        # Update metrics and return their value.
        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)

        return {
            "d_loss": self.d_loss_tracker.result(),
            "g_loss": self.g_loss_tracker.result(),
        }

# Get the paths of all images in a directory
def get_image_paths(directory):
    image_paths = []

    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(directory, '*' + ext)))

    return sorted(image_paths)


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


class DisplacementMapDataset(Dataset):
    def __init__(self, input_image_paths, target_image_paths, transform=None):
        self.input_image_paths = input_image_paths
        self.target_image_paths = target_image_paths
        self.transform = transform

    def __len__(self):
        return len(self.input_image_paths)

    def load_image(self, path):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            image = Image.fromarray(image)
            return image
        else:
            print(f"Error: Image at {path} could not be read.")
            return None

    def __getitem__(self, idx):
        input_image = self.load_image(self.input_image_paths[idx])
        target_image = self.load_image(self.target_image_paths[idx])

        if input_image is None or target_image is None:
            return None  # Return None or handle as required

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)
        
        return input_image, target_image


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
    # common_transforms = transforms.Compose([
    #     transforms.Resize((512, 512)),
    #     transforms.Lambda(lambda x: x.convert('L')),
    #     transforms.reshape(-1, 512, 512, 1),
    #     transforms.ToTensor()
    # ])

    common_transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.Grayscale(),  # Ensure images are grayscale
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: x.permute(1, 2, 0))  # Permute the dimensions of the tensor
    ])

    # Calculate mean and std for the dataset
    #mean, std = calculate_mean_and_std(intact_image_paths, damaged_image_paths, common_transforms)

    # Data Augmentation for the Real Images
    image_transforms = transforms.Compose([
        #transforms.RandomResizedCrop(512, scale=(0.8, 1.0)),
        transforms.RandomRotation(10), 
        #transforms.RandomCrop(224),
        common_transforms,
        #transforms.Normalize(mean=mean.tolist(), std=std.tolist())
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
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = False)
    val_dataloader   = DataLoader(val_dataset,   batch_size = batch_size, shuffle = False)

    return train_dataloader, val_dataloader


# Define loss functions
adversarial_loss_fn = BinaryCrossentropy(from_logits=True)
content_loss_fn     = MeanAbsoluteError()

# Loss function for the GAN
def combined_loss(discriminator_output, real_labels, generated_images, real_images):
    # Adversarial loss (how well can the generator fool the discriminator)
    adversarial_loss = adversarial_loss_fn(real_labels, discriminator_output)

    # Content loss (how close is the generated image to the real image)
    content_loss = content_loss_fn(real_images, generated_images)

    # Combine the two losses
    return adversarial_loss + content_loss

########################################################################################
# Main
# - Load displacement maps
# - Generate synthetic displacement maps
# - Validate the generated data
########################################################################################
if __name__ == "__main__":
    # Check if MPS (Multi-Process Service) is available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS device found.")
    else:
        device = torch.device("cpu")
        print("MPS device not found.")

    device = torch.device("cpu")

    # Load the dataset
    train_dataloader, val_dataloader = load_dataset()

    # Get the image shape
    image_shape = (512, 512, 1)

    # Instantiate the generator
    generator     = get_generator(image_shape).to(device)

    # Instantiate the discriminator
    discriminator = get_discriminator(image_shape).to(device)

    # Instantiate the GAN model
    gan = GAN(discriminator, generator).to(device)

    # Compile the GAN model
    gan.compile(
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=discriminator_lr, betas=(0.5, 0.999)),
        g_optimizer = torch.optim.Adam(generator.parameters(),     lr=generator_lr,     betas=(0.5, 0.999)),
        d_loss_fn   = adversarial_loss_fn,
        g_loss_fn   = combined_loss
    )

    # Train the GAN model
    gan.fit(
        train_dataloader, 
        epochs          = 100, 
        callbacks       = None,
        validation_data = val_dataloader,

        )

    gan.evaluate(val_dataloader)

    # Save the model state dictionary
    torch.save(gan.state_dict(), 'models/gan.pth')

    print("Done!")