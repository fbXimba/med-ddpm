# Generate AD=2 sample from a specific mask
#python sample_ADNI.py \
#    --checkpoint ./results/results_1/model-150.pt --condition_mask ../ADNI_split/ADNI_test_dataset/mask/941_S_1203_mask.nii.gz --diagnosis 2 --num_samples 1
#    --output ./generated_AD.nii.gz

# Generate 10 samples from random test conditions
#python sample_ADNI.py \
#    --checkpoint ./results/results_1/model-150.pt --num_samples 10

import os 
import torch
import numpy as np
import nibabel as nib
from torchvision.transforms import Compose, Lambda
from diffusion_model.trainer_ADNI import GaussianDiffusion
from diffusion_model.unet_ADNI import create_model
from dataset_ADNI import NiftiPairImageGenerator
import argparse
import pandas as pd
import datetime

idx_to_label = {
    0: "CN",
    1: "MCI",
    2: "AD"
}

################################################################### functions

def load_trained_model(checkpoint_path, input_size=128, depth_size=128, num_channels=64, num_res_blocks=2, timesteps=250):
    """Load the trained diffusion model"""

    # Create model architecture
    model = create_model(
        input_size, 
        num_channels, 
        num_res_blocks, 
        class_cond=True,  # Enable class conditioning
        in_channels=2,    # mask + noisy image
        out_channels=1
    ).cuda()
    
    # Create diffusion wrapper
    diffusion = GaussianDiffusion(
        model,
        image_size=input_size,
        depth_size=depth_size,
        timesteps=timesteps,
        loss_type='l1',
        with_condition=True,
        channels=1
    ).cuda()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
    diffusion.load_state_dict(checkpoint['ema'])  # Use EMA weights
    diffusion.eval()  # Set to evaluation mode to keep unchanged
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Training step: {checkpoint.get('step', 'unknown')}")
    
    return diffusion

def sample_from_condition(diffusion, condition_mask_path, diagnosis_label, output_path, num_samples=1, seed=None):
    """
    Sample images conditioned on a mask and diagnosis label
    
    Args:
        diffusion: model
            Trained GaussianDiffusion model
        condition_mask_path: str
            Path to condition mask (.nii.gz)
        diagnosis_label: int
            Diagnosis class (0=CN, 1=MCI, 2=AD)
        output_path: str
            Where to save generated image
        num_samples: int
            Number of samples to generate
    """

    # Load and preprocess condition mask
    subject_id = os.path.basename(condition_mask_path).replace('_mask.nii.gz', '')
    condition_nifti = nib.load(condition_mask_path)
    condition_img = condition_nifti.get_fdata()
    original_affine = condition_nifti.affine  # Preserve affine for correct orientation
    
    # Transform to tensor
    transform = Compose([
        Lambda(lambda t: torch.tensor(t).float()),
        Lambda(lambda t: t.unsqueeze(0))  # Add channel dimension
    ])
    
    condition_tensor = transform(condition_img).unsqueeze(0).cuda()  # [1, 1, H, W, D]
    
    # Create diagnosis tensor
    diagnosis_tensor = torch.tensor([diagnosis_label]).long().cuda()  # [1]
    
    #print(f"Condition mask shape: {condition_tensor.shape}")
    #print(f"Diagnosis label: {diagnosis_label}")
    
    # Generate samples
    with torch.no_grad():
        for i in range(num_samples):
            print(f"Generating sample {i+1}/{num_samples}...")

            if seed is not None:
                seed_i = seed + i*6
                torch.manual_seed(seed + i*6)  # Increment seed of 6 for each sample
            
            generated = diffusion.sample( # function in trainer Gaussian diffusion class line 245
                batch_size=1,
                condition_tensors=condition_tensor,
                diagnosis=diagnosis_tensor
            )
            
            # Convert to numpy and save
            sample_img = generated.cpu().numpy()[0, 0]  # [H, W, D]
            
            # Save as NIfTI with correct affine transformation
            nifti_img = nib.Nifti1Image(sample_img, affine=original_affine) # affine from mask file
            save_path_img = os.path.join(output_path, f'{subject_id}_sampled_{idx_to_label[diagnosis_label]}{"" if seed is None else "_" + str(seed_i) }.nii.gz')
            nib.save(nifti_img, save_path_img)
            print(f"Saved to {save_path_img}")

def batch_sample_from_dataset(diffusion, dataset, num_samples=10, output_folder='./samples', seed=None):
    """Sample from multiple conditions in the dataset"""

    # Sample random conditions from dataset
    sample_conditions = dataset.sample_conditions(batch_size=num_samples) # function from dataset NiftiPairImageGenerator class
    condition_tensors = sample_conditions['condition_tensors']
    diagnosis_labels = sample_conditions['diagnosis']
    indexes = sample_conditions['indexes']  # Use the same indexes that were sampled
    
    # Get affine matrices from the actual files
    affine_matrices = []
    for idx in indexes:
        input_file = dataset.pair_files[idx][0]
        affine = nib.load(input_file).affine
        affine_matrices.append(affine)
    
    print(f"Generating {num_samples} samples...")

    if seed is not None:
        torch.manual_seed(seed)
    
    with torch.no_grad():
        generated = diffusion.sample(
            batch_size=num_samples,
            condition_tensors=condition_tensors,
            diagnosis=diagnosis_labels
        )
    
    # Save each sample
    for i in range(num_samples):
        folder=os.path.join(output_folder, f"{i+1}")
        os.makedirs(folder, exist_ok=True)

        sample_img = generated[i, 0].cpu().numpy()  # [H, W, D]
        diagnosis = diagnosis_labels[i].item()
        
        # Use affine from original file for correct orientation
        affine = affine_matrices[i] # get corresponding affine from condition file
        nifti_img = nib.Nifti1Image(sample_img, affine=affine)
        save_path = os.path.join(folder, f'sample_{i+1}_diagnosis{idx_to_label[diagnosis]}{"" if seed is None else "_" + str(seed + i) + "B" + str(i)}.nii.gz')
        nib.save(nifti_img, save_path) 
        print(f"Saved sample {i+1} with diagnosis {idx_to_label[diagnosis]}")

################################################################## main

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # TODO: set specific GPU if multiple available
    os.environ["CUDA_VISIBLE_DEVICES"]="1" # TODO: set specific GPU if multiple available
    os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"

    now=datetime.datetime.now().strftime("%y%m_T%H%M%S")

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default="./results/260320_095902/model-118.pt", help='Path to model checkpoint')
    parser.add_argument('--diagnosis_csv', type=str, default="../ADNI_split/ADNI_test_dataset/diagnosis/test_subjects.csv", help='Path to CSV file containing subject IDs and diagnosis labels')
    parser.add_argument('--condition_mask', type=str, help='Path to condition mask')
    parser.add_argument('--diagnosis', type=int, default=0, help='Diagnosis label (0=CN, 1=MCI, 2=AD)')
    parser.add_argument('--output', type=str, default=f'./generated_sample/{now}')
    parser.add_argument('--num_samples', type=int, default=1)
    #parser.add_argument('--batch_sample', action='store_true', help='Sample from dataset conditions')
    parser.add_argument('--input_folder', type=str, default="../head_datasets/ADNI_test_dataset/mask/")
    parser.add_argument('--target_folder', type=str, default="../head_datasets/ADNI_test_dataset/image/")
    #parser.add_argument('--timesteps', type=int, default=250) #NOTE: seen decreasing: CAN'T use diffrent one from training
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Load model
    diffusion = load_trained_model(args.checkpoint,timesteps=250)

    os.makedirs(args.output, exist_ok=True)
    
    if args.condition_mask is None:
        # batch sampling 
        print("Batch sampling from dataset conditions...")

        # Load test dataset
        transform = Compose([ # same as training transforms in dataset_ADNI.py
            Lambda(lambda t: torch.tensor(t).float()),
            Lambda(lambda t: t.unsqueeze(0))
        ])
        
        diagnosis_df = pd.read_csv(args.diagnosis_csv)
        diagnosis_labels = diagnosis_df['Diagnosis'].astype(int).tolist()
        
        dataset = NiftiPairImageGenerator(
            args.input_folder,
            args.target_folder,
            input_size=128,
            depth_size=128,
            transform=transform,
            target_transform=transform,
            full_channel_mask=False,
            diagnosis_label=diagnosis_labels
        )
        
        batch_sample_from_dataset(
            diffusion,
            dataset, 
            num_samples=args.num_samples, 
            output_folder=args.output,
            seed=args.seed
        )

    else:
        # Single condition sampling
        print(f"Sampling from mask: {args.condition_mask} with diagnosis: {args.diagnosis}")
        if not args.condition_mask:
            raise ValueError("--condition_mask required for single sampling")
        
        sample_from_condition(
            diffusion,
            args.condition_mask,
            args.diagnosis,
            args.output,
            args.num_samples,
            seed=args.seed
        )
