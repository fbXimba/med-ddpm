# train_ADNI.py

#python train_ADNI.py --with_condition

import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # TODO: set specific GPU if multiple available
os.environ["CUDA_VISIBLE_DEVICES"]="1" # TODO: set specific GPU if multiple available


from torchvision.transforms import  Compose, Lambda#, ToPILImage, Resize, ToTensor,RandomCrop
from diffusion_model.trainer_ADNI import GaussianDiffusion, Trainer
from diffusion_model.unet_ADNI import create_model
from dataset_ADNI import NiftiImageGenerator, NiftiPairImageGenerator
import argparse
import torch
import pandas as pd
import json
import yaml
import wandb


#device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"CUDA AVAILABLE: {torch.cuda.is_available}")

# -

# Load WandB key from yaml
with open("key.yaml") as file:
    config=yaml.safe_load(file)
key=config["wandb"]["key"]

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputfolder', type=str, default="../ADNI_split/ADNI_train_dataset/mask/")
parser.add_argument('-t', '--targetfolder', type=str, default="../ADNI_split/ADNI_train_dataset/image/")
parser.add_argument('-d', '--diagnosisfolder', type=str, default="../ADNI_split/ADNI_train_dataset/diagnosis/")
parser.add_argument('--key', type=str, default=key ,help="Weights and Biases key for logging")
parser.add_argument('--input_size', type=int, default=128) # already for preprocessed ADNI
parser.add_argument('--depth_size', type=int, default=128) # already for preprocessed ADNI
parser.add_argument('--num_channels', type=int, default=64)
parser.add_argument('--num_res_blocks', type=int, default=2)
parser.add_argument('--num_class_labels', type=int, default=3)
parser.add_argument('--train_lr', type=float, default=2e-4) # ex e5
# exponential lr scheduler parameters + warmup
parser.add_argument('--lr_decay_rate', type=float, default=0.9999)  # learning rate decay rate for ExponentialLR
parser.add_argument('--lr_warmup_steps', type=int, default=5000)  # decay steps for learning rate scheduler 
parser.add_argument('--lr_min', type=float, default=2e-7)  # minimum learning rate after warmup
## plateau lr scheduler parameters
#parser.add_argument('--lr_plateau_factor', type=float, default=0.5)  # factor for ReduceLROnPlateau
#parser.add_argument('--lr_plateau_patience', type=int, default=500)  # patience for ReduceLROnPlateau
parser.add_argument('--batchsize', type=int, default=1)
parser.add_argument('--epochs', type=int, default=100000) # epochs parameter specifies the number of training iterations # ex 50000
parser.add_argument('--timesteps', type=int, default=250)
parser.add_argument('--save_and_sample_every', type=int, default=1000)
parser.add_argument('--with_condition', action='store_true', help='whether to use condition or not with semantic mask and diagnosis label')
parser.add_argument('-r', '--resume_weight', type=str, default="model/model_128.pt")

args = parser.parse_args()

inputfolder = args.inputfolder
targetfolder = args.targetfolder
# Load diagnosis labels from CSV : columns 'subject_id', 'image_id', 'group', 'diagnosis'
diagnosis_df = pd.read_csv(os.path.join(args.diagnosisfolder, 'train_subjects.csv'))
diagnosis_label = diagnosis_df['Diagnosis'].astype(int).tolist()
input_size = args.input_size # already taken care of for preprocessed ADNI: keep consistent value for gaussian diffusion
depth_size = args.depth_size # already taken care of for preprocessed ADNI: keep consistent value for gaussian diffusion
num_channels = args.num_channels
num_res_blocks = args.num_res_blocks
num_class_labels = args.num_class_labels
save_and_sample_every = args.save_and_sample_every
with_condition = args.with_condition
resume_weight = args.resume_weight
train_lr = args.train_lr

# wandb login
wandb.login(key=args.key)

# input tensor: (B, 1, H, W, D)  value range: [-1, 1] 
transform = Compose([ 
    Lambda(lambda t: torch.tensor(t).float()), # to tensor
    # scale to [-1, 1] # not needed for ADNI preprocessed
    Lambda(lambda t: t.unsqueeze(0)), # add channel dimension
    # (H, W, D, C) -> (H, C, W, D) # not needed for ADNI dataset
])

if with_condition:
    dataset = NiftiPairImageGenerator( # implementation in dataset.py for ADNI dataset with binary mask + diagnosis label
        inputfolder,
        targetfolder,
        input_size=input_size,
        depth_size=depth_size,
        transform=transform,
        target_transform=transform,
        full_channel_mask=False, # ADNI mask is single channel !!!!!
        diagnosis_label=diagnosis_label # pass diagnosis labels to the dataset
    )
else: # without condition (not ADNI)
    dataset = NiftiImageGenerator(
        inputfolder,
        input_size=input_size,
        depth_size=depth_size,
        transform=transform
    )


print("=== DATASET VERIFICATION ===")
try:
    print(f"Dataset length: {len(dataset)}")
    print(f"Diagnosis labels length: {len(diagnosis_label)}")
    assert len(dataset) == len(diagnosis_label), "Dataset and diagnosis length mismatch!"
    
    # Test sample loading
    sample = dataset[0]
    print(f"Input shape: {sample['input'].shape}")   # Should be [1, 128, 128, 128]
    print(f"Target shape: {sample['target'].shape}") # Should be [1, 128, 128, 128]
    print(f"Diagnosis: {sample['diagnosis']}")
    print("=== DATASET OK ===")
    
except Exception as e:
    print(f"DATASET ERROR: {e}")
    exit(1)

check_points = args.epochs//save_and_sample_every

print(f"Numero di checkpointss: {check_points}")

# Define model
# ADNI: mask 1+ noisy image 1 = 2 concatenated channels with with_condition=True
in_channels = 2 if with_condition else 1 # mask + noise image when conditioned
out_channels = 1 # different from BraTS where output has multiple channels, only one channel for ADNI dataset bc of single brain mask output (bc binary mask)

# class_cond = True for ADNI dataset with conditioning on mask and diagnosis label
model = create_model(
    input_size, 
    num_channels, 
    num_res_blocks, 
    class_cond=True, 
    in_channels=in_channels, 
    out_channels=out_channels
    ).cuda()#to(device)

diffusion = GaussianDiffusion(
    model,
    image_size = input_size,
    depth_size = depth_size,
    timesteps = args.timesteps,   # number of steps
    loss_type = 'l1',    # L1 or L2 # L1 = (1/n) * Σ|y_true — y_pred|
    with_condition=with_condition,
    channels=out_channels
).cuda()#to(device)

initial_weights=None

try:
    if len(resume_weight) > 0:
        weight = torch.load(resume_weight, map_location='cuda')
        diffusion.load_state_dict(weight['ema'])
        initial_weights = resume_weight
        print("Model Loaded!")

except Exception as e:
    print("NO WEIGHTS PRESENT")

print("=== COMPREHENSIVE PIPELINE TEST ===")
try:
    # Test batch loading
    from torch.utils.data import DataLoader
    dl = DataLoader(dataset, batch_size=1, shuffle=False)
    batch = next(iter(dl))
    
    print(f"Batch input shape: {batch['input'].shape}")   # [1, 1, 128, 128, 128]
    print(f"Batch target shape: {batch['target'].shape}") # [1, 1, 128, 128, 128]
    print(f"Batch diagnosis: {batch['diagnosis']}")
    
    # Test forward pass
    input_tensors = batch['input'].cuda()#to(device)
    target_tensors = batch['target'].cuda()#to(device)
    diagnosis = torch.tensor(batch['diagnosis']).long().cuda()#to(device)
    
    with torch.no_grad():
        loss = diffusion(target_tensors, condition_tensors=input_tensors, diagnosis=diagnosis)
        print(f"Forward pass successful: {loss.item():.6f}")
    
    # Test sampling
    with torch.no_grad():
        sample_result = diffusion.sample(batch_size=1, condition_tensors=input_tensors, diagnosis=diagnosis)
        print(f"Sampling successful: {sample_result.shape}")
    
    # Test condition sampling
    sample_condition = dataset.sample_conditions(batch_size=1)
    print(f"Condition sampling: {sample_condition['condition_tensors'].shape}")
    
    print("=== PIPELINE FULLY READY ===")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

trainer = Trainer(
    diffusion,
    dataset, # supplies both input and target images and diagnosis labels if with_condition=True
    image_size = input_size,
    depth_size = depth_size,
    train_batch_size = args.batchsize,
    train_lr = train_lr,
    train_num_steps = args.epochs,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps: effective batch size # reason of the 2 loss : 2 forword pass (2 images) per 1 backward pass (update of weights) to help with 3d images memory (high)
    ema_decay = 0.995,                # exponential moving average decay
    fp16 = False,#True,                       # turn on mixed precision training with apex
    with_condition=with_condition,
    save_and_sample_every = save_and_sample_every,
    initial_weights=initial_weights,
    # exp lr scheduler parameters
    lr_decay_rate = args.lr_decay_rate,  # learning rate decay rate: for ExponentialLR LRx0.999 every optim update (slow, 0.99 faster)
    lr_warmup_steps = args.lr_warmup_steps,  # warmup--> exp decay steps
    ## plateau lr scheduler parameters
    #lr_plateau_factor = args.lr_plateau_factor,
    #lr_plateau_patience = args.lr_plateau_patience,
    lr_min = args.lr_min,
)

trainer.train()

wandb.log({"timesteps": args.timesteps})
wandb.finish()