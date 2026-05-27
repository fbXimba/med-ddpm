import os
import pandas as pd
import argparse
from sample_ADNI import sample_from_condition, load_trained_model
import torch
import yaml
import time

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # TODO: set specific GPU if multiple available
    os.environ["CUDA_VISIBLE_DEVICES"]="1" # TODO: set specific GPU if multiple available
    os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default=config["directories"]["checkpoints_dir"], help='Path to model checkpoints')
    parser.add_argument('--run', type=str,default=config["dataset"]["run"], help='Run name')
    parser.add_argument('--checkpoint', type=str, default=config["sampling"]["checkpoint"], help='Checkpoint file')
    parser.add_argument('--mask_dirs', type=str, default=config["dataset"]["masks"],  help='Path to condition masks')
    parser.add_argument('--info_masks', default=config["dataset"]["info_masks"],  type=str, help='CSV file with mask subject information')
    parser.add_argument('--len_data', type=int, default=config["dataset"]["data_len"], help='Number of samples to generate')
    parser.add_argument('--output', type=str, default=config["directories"]["sample_dir"], help='Directory to save generated samples')
    parser.add_argument('--num_samples', type=int, default=config["sampling"]["num_samples"], help='Number of samples to generate per subject')
    parser.add_argument('--seed', type=int, default=config["dataset"]["seed"], help='Random seed for reproducibility')
    args = parser.parse_args()

    #only_same_condition = False # whether to sample from same diagnosis condition or different one 

    label_to_idx = {
        "CN": 0,
        "MCI": 1,
        "AD": 2
    }

    checkpoint_path = os.path.join(args.checkpoint_dir, args.run, f"model-{args.checkpoint}.pt")

    # Load model
    t_in = time.time()
    diffusion = load_trained_model(checkpoint_path=checkpoint_path) # input_size=128, depth_size=128, num_channels=64, num_res_blocks=2, timesteps=250

    os.makedirs(args.output, exist_ok=True)
    output_folder = os.path.join(args.output, args.run, args.checkpoint)
    os.makedirs(output_folder, exist_ok=True)

    df = pd.read_csv(args.info_masks)
    len_subj = len(df['Subject'])

    seed = args.seed

    with open(os.path.join(args.output, args.run, f"{args.run}_chkpt{args.checkpoint}.csv"), 'w') as f:
        # header
        f.write("Subject,Group,Seed,Filename,Filename_processed\n")

        for diagnosis in df['Group'].unique():
            count = 0

            df_diag = df #df[df['Group'] == diagnosis] if only_same_condition else df
            len_diag = len(df_diag)

            for i in range(args.len_data):
                round = count // len_diag
                idx = count - len_diag * round
                subject = df_diag['Subject'].iloc[idx]

                for j in range(args.num_samples):
                    filename = f"{subject}_sampled_{diagnosis}_{seed+j*6}.nii.gz" 
                    filename_processed = f"{subject}_sampled_{diagnosis}_{seed+j*6}_processed.nii.gz"     
                    f.write(f"{subject},{diagnosis},{seed+j*6},{filename},{filename_processed}\n")

                mask_path = os.path.join(args.mask_dirs, f"{subject}_mask.nii.gz")
                # Check if files already exist before sampling
                if all(os.path.exists(os.path.join(output_folder, f"{subject}_sampled_{diagnosis}_{seed+j*6}.nii.gz")) for j in range(args.num_samples)):
                    print(f"Samples for subject {subject} {diagnosis} already exists")
                else:
                    sample_from_condition(
                        diffusion,
                        mask_path,
                        label_to_idx[diagnosis],
                        output_folder,
                        args.num_samples,
                        seed=seed
                    )

                seed += args.num_samples*6 # increment of 6
                count += 1

    t_fin = time.time()
    print(f"Dataset creation completed in {(t_fin - t_in)/60:.2f} minutes")