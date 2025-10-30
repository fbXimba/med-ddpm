# sample_ADNI.py

# -*- coding:utf-8 -*-

from diffusion_model.trainer_ADNI import GaussianDiffusion, num_to_groups
from diffusion_model.unet_ADNI import create_model
from torchvision.transforms import Compose, Lambda
import nibabel as nib
import torchio as tio
import numpy as np
import argparse
import torch
import os
import glob

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--inputfolder", type=str, default="dataset/whole_head/mask")
parser.add_argument("-e", "--exportfolder", type=str, default="exports/")
parser.add_argument("--input_size", type=int, default=128)
parser.add_argument("--depth_size", type=int, default=128)
parser.add_argument("--num_channels", type=int, default=64)
parser.add_argument("--num_res_blocks", type=int, default=1)
parser.add_argument("--batchsize", type=int, default=1)
parser.add_argument("--num_samples", type=int, default=1)
parser.add_argument("--num_class_labels", type=int, default=3)
parser.add_argument("--timesteps", type=int, default=250)
parser.add_argument("--diagnosis", type=int, default=0, help="Diagnosis class: 0=CN, 1=MCI, 2=AD")
parser.add_argument("-w", "--weightfile", type=str, default="model/model_128.pt")
parser.add_argument('--with_condition', action='store_true', help='whether to use condition or not with semantic mask and diagnosis label')

args = parser.parse_args()

exportfolder = args.exportfolder
inputfolder = args.inputfolder
input_size = args.input_size
depth_size = args.depth_size
batchsize = args.batchsize
weightfile = args.weightfile
num_channels = args.num_channels
num_res_blocks = args.num_res_blocks
num_samples = args.num_samples
with_condition = args.with_condition
# Fixed: use correct input channels for trained model?
in_channels = 2 if with_condition else 1  # concatenated noisy image + mask
out_channels = 1
device = "cuda"
diagnosis_class = args.diagnosis

mask_list = sorted(glob.glob(f"{inputfolder}/*.nii.gz"))
print(f"Found {len(mask_list)} mask files")


def read_image(file_path):
    """Read and normalize image to [-1,1] like in training"""
    img = nib.load(file_path).get_fdata()
    # Assuming your masks are already preprocessed to [-1,1] like in training
    # If not, uncomment the normalization below:
    # img = (img - img.min()) / (img.max() - img.min())  # 0-1 first
    # img = img * 2.0 - 1.0  # then to [-1,1]
    return img


def resize_img(input_img):
    """Resize 3D image to target dimensions"""
    if input_img.shape != (input_size, input_size, depth_size):
        img = tio.ScalarImage(tensor=input_img[np.newaxis, ...])
        resize_transform = tio.Resize((input_size, input_size, depth_size))
        img = np.asarray(resize_transform(img))[0]
        return img
    return input_img


# Simplified transform that matches training
input_transform = Compose([ # Assuming images are already normalized to [-1,1]
        Lambda(lambda t: torch.tensor(t).float()),
        Lambda(lambda t: t.unsqueeze(0)),  # Add channel dimension [H,W,D] -> [1,H,W,D]
    ])

# Create model with correct parameters
model = create_model(
    input_size,
    num_channels,
    num_res_blocks,
    class_cond=True,  # Enable class conditioning
    in_channels=in_channels,
    #out_channels=out_channels,
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size=input_size,
    depth_size=depth_size,
    timesteps=args.timesteps,
    loss_type="l1",  # Match training
    with_condition=with_condition,
    channels=out_channels
).cuda()

# Load trained weights
checkpoint = torch.load(weightfile)
diffusion.load_state_dict(checkpoint["ema"])
print("Model Loaded!")

# Create output directories
img_dir = os.path.join(exportfolder, "image")
msk_dir = os.path.join(exportfolder, "mask")
os.makedirs(img_dir, exist_ok=True)
os.makedirs(msk_dir, exist_ok=True)

print(f"Generating samples with diagnosis class: {diagnosis_class} (0=CN, 1=MCI, 2=AD)")

for k, inputfile in enumerate(mask_list): # iterate over all mask files
    left = len(mask_list) - (k + 1)
    print(f"Processing file {k + 1}/{len(mask_list)}, LEFT: {left}")

    # Load and process mask
    ref = nib.load(inputfile)
    msk_name = inputfile.split("/")[-1]
    refImg = ref.get_fdata()

    # Process mask image
    img = read_image(inputfile)  # Use the same read_image function as training
    img = resize_img(img)
    input_tensor = input_transform(img)  # Shape: [1, H, W, D]

    # Sampling loop
    batches = num_to_groups(num_samples, batchsize)
    steps = len(batches)
    sample_count = 0

    print(f"All Steps: {steps}")
    counter = 0 # to count saved samples

    for i, bsize in enumerate(batches): # for each batch
        print(f"Step [{i + 1}/{steps}]")
        condition_tensors = []
        diagnosis_labels = []
        counted_samples = []

        for b in range(bsize): # for each sample in the batch
            condition_tensors.append(input_tensor)
            diagnosis_labels.append(diagnosis_class)  # Use specified diagnosis
            counted_samples.append(sample_count)
            sample_count += 1

        condition_tensors = torch.cat(condition_tensors, 0).cuda()
        diagnosis_tensor = torch.tensor(diagnosis_labels).long().cuda()

        # Sample with both mask and diagnosis conditioning
        all_images_list = []
        for n in [bsize]: # process the whole batch at once
            samples = diffusion.sample( # sample function in trainer_ADNI.py
                batch_size=n,
                condition_tensors=condition_tensors,
                diagnosis=diagnosis_tensor,
            )
            all_images_list.append(samples)

        all_images = torch.cat(all_images_list, dim=0)
        sampleImages = all_images.cpu().numpy()

        for b, c in enumerate(counted_samples): # save each sample in the batch
            counter = counter + 1
            sampleImage = sampleImages[b][0]  # Remove batch and channel dims

            # Ensure correct shape matching reference
            sampleImage = sampleImage.reshape(refImg.shape)

            # Create and save NIfTI image
            nifti_img = nib.Nifti1Image(sampleImage, affine=ref.affine)
            output_name = f"{counter}_{diagnosis_class}_{msk_name}"
            nib.save(nifti_img, os.path.join(img_dir, output_name))
            nib.save(ref, os.path.join(msk_dir, output_name))

        torch.cuda.empty_cache()
    print("OK!")

print("Sampling completed!")
