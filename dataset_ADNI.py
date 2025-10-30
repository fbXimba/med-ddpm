# dataset_ADNI.py

from sklearn.preprocessing import MinMaxScaler # to scale images between 0 and 1
from torch.utils.data import Dataset
#from torchvision.transforms import Compose, ToTensor, Lambda # not used
from glob import glob
from utils.dtypes import LabelEnum
import matplotlib.pyplot as plt
import nibabel as nib
import torchio as tio
import numpy as np
import torch
import re
import os


# 1
# class for with_condition = False : NOT ADNI
class NiftiImageGenerator(Dataset):
    def __init__(self, imagefolder, input_size, depth_size, transform=None):
        self.imagefolder = imagefolder
        self.input_size = input_size
        self.depth_size = depth_size
        self.inputfiles = glob(os.path.join(imagefolder, "*.nii.gz"))
        self.scaler = MinMaxScaler()
        self.transform = transform

    def read_image(self, file_path):
        img = nib.load(file_path).get_fdata()
        img = self.scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(
            img.shape
        )  # 0 -> 1 scale
        return img

    def plot_samples(self, n_slice=15, n_row=4):
        samples = [
            self[index] for index in np.random.randint(0, len(self), n_row * n_row)
        ]
        for i in range(n_row):
            for j in range(n_row):
                sample = samples[n_row * i + j]
                sample = sample[0]
                plt.subplot(n_row, n_row, n_row * i + j + 1)
                plt.imshow(sample[:, :, n_slice])
        plt.show()

    def __len__(self):
        return len(self.inputfiles)

    def __getitem__(self, index):
        inputfile = self.inputfiles[index]
        img = self.read_image(inputfile)
        h, w, d = img.shape
        if h != self.input_size or w != self.input_size or d != self.depth_size:
            img = tio.ScalarImage(inputfile)
            cop = tio.Resize((self.input_size, self.input_size, self.depth_size))
            img = np.asarray(cop(img))[0]

        if self.transform is not None:
            img = self.transform(img)
        return img


# 2
# class for with_condition = True : ADNI
# Dataset class for paired Nifti images (e.g., mask and image) implemented with optional diagnosis label handling
class NiftiPairImageGenerator(Dataset):
    def __init__(
        self,
        input_folder: str,
        target_folder: str,
        input_size: int,
        depth_size: int,
        input_channel: int = 3,
        transform=None,
        target_transform=None,
        full_channel_mask=False,  # whether input mask has multiple channels!!! FALSE for ADNI
        combine_output=False,
        diagnosis_label: list = None,  # new parameter for diagnosis labels
    ):
        self.input_folder = input_folder
        self.target_folder = target_folder
        self.pair_files = self.pair_file()
        self.input_size = input_size
        self.depth_size = depth_size
        self.input_channel = input_channel
        self.scaler = MinMaxScaler() # scaler to 0-1
        self.transform = transform
        self.target_transform = target_transform
        self.full_channel_mask = full_channel_mask
        self.combine_output = combine_output
        assert diagnosis_label is not None, "Diagnosis labels must be provided for conditional dataset."
        self.diagnosis_label = diagnosis_label  # store diagnosis labels

    # Create list of paired input and target files
    def pair_file(self):
        input_files = sorted(glob(os.path.join(self.input_folder, "*")))
        target_files = sorted(glob(os.path.join(self.target_folder, "*")))
        pairs = []
        for input_file, target_file in zip(input_files, target_files):
            assert int("".join(re.findall("\d", input_file))) == int(
                "".join(re.findall("\d", target_file))
            )
            pairs.append((input_file, target_file))
        return pairs

    # Convert label image to multi-channel binary masks
    def label2masks(self, masked_img):
        result_img = np.zeros(masked_img.shape + (self.input_channel - 1,))
        result_img[masked_img == LabelEnum.BRAINAREA.value, 0] = 1
        result_img[masked_img == LabelEnum.TUMORAREA.value, 1] = 1
        return result_img

    # Read image from file and scale if needed
    # pass_scaler = True to avoid scaling mask images implemented for ADNI dataset preprocessed
    # False for other datasets if scaling is needed
    def read_image(self, file_path, pass_scaler=True):
        img = nib.load(file_path).get_fdata()
        if not pass_scaler: # scale to [0, 1] only if needed
            img = self.scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape( # to 0-1 scale
                img.shape
            )
        return img

    # Plot input and target images for a given index and slice
    def plot(self, index, n_slice=30):
        data = self[index]
        input_img = data["input"]
        target_img = data["target"]
        plt.subplot(1, 2, 1)
        plt.imshow(input_img[:, :, n_slice])
        plt.subplot(1, 2, 2)
        plt.imshow(target_img[:, :, n_slice])
        plt.show()

    # Resize 3D image to specified input and depth size : not necessary for ADNI
    def resize_img(self, img):
        # Return directly if already correct shape
        if img.shape == (self.input_size, self.input_size, self.depth_size):
            return img
        else:
            # Otherwise, perform resize
            h, w, d = img.shape
            img = tio.ScalarImage(tensor=img[np.newaxis, ...])
            cop = tio.Resize((self.input_size, self.input_size, self.depth_size))
            return np.asarray(cop(img))[0]

        # h, w, d = img.shape
        # if h != self.input_size or w != self.input_size or d != self.depth_size:
        #    img = tio.ScalarImage(tensor=img[np.newaxis, ...])
        #    cop = tio.Resize((self.input_size, self.input_size, self.depth_size))
        #    img = np.asarray(cop(img))[0]
        # return img

    # Resize 4D image (with channels) to specified input and depth size: NOT ADNI
    def resize_img_4d(self, input_img):
        h, w, d, c = input_img.shape
        result_img = np.zeros((self.input_size, self.input_size, self.depth_size, 2))
        if h != self.input_size or w != self.input_size or d != self.depth_size:
            for ch in range(c):
                buff = input_img.copy()[..., ch]
                img = tio.ScalarImage(tensor=buff[np.newaxis, ...])
                cop = tio.Resize((self.input_size, self.input_size, self.depth_size))
                img = np.asarray(cop(img))[0]
                result_img[..., ch] += img
            return result_img
        else:
            return input_img

    # Sample conditions for a batch of images
    def sample_conditions(self, batch_size: int):
        indexes = np.random.randint(0, len(self), batch_size)
        input_files = [self.pair_files[index][0] for index in indexes]
        input_tensors = []
        diagnosis_labels = []  # new list for diagnosis labels

        for i, input_file in enumerate(input_files):
            input_img = self.read_image(input_file, pass_scaler=True)
            # next two useless for ADNI but kept for generality
            input_img = (
                self.label2masks(input_img) if self.full_channel_mask else input_img
            )
            input_img = (
                self.resize_img(input_img)
                if not self.full_channel_mask
                else self.resize_img_4d(input_img)
            )  # full_channel_mask= False for ADNI

            if self.transform is not None:
                input_img = self.transform(input_img).unsqueeze(0)
                input_tensors.append(input_img)

            if self.diagnosis_label is not None:  # new part for diagnosis labels
                diagnosis_labels.append(self.diagnosis_label[indexes[i]])

        # return torch.cat(input_tensors, 0).cuda()
        return {
            "condition_tensors": torch.cat(input_tensors, 0).cuda(),
            "diagnosis": torch.tensor(diagnosis_labels).long().cuda(),
        }

    # Get length of dataset
    def __len__(self):
        return len(self.pair_files)

    # Get item at specific index: input image, target image, and diagnosis label if available
    def __getitem__(self, index):
        # Get input and target file paths
        input_file, target_file = self.pair_files[index]

        # Read and process input image
        input_img = self.read_image(input_file, pass_scaler=True)  # pass_scaler=True to avoid scaling mask previously= self.full_channel_mask

        # next two useless for ADNI but kept for generality
        input_img = self.label2masks(input_img) if self.full_channel_mask else input_img  # full_channel_mask= False for ADNI
        input_img = (self.resize_img(input_img) if not self.full_channel_mask else self.resize_img_4d(input_img))

        # Read and process target image
        target_img = self.read_image(target_file, pass_scaler=True)
        target_img = self.resize_img(target_img)

        # Apply transformations if any
        if self.transform is not None:
            input_img = self.transform(input_img)
        if self.target_transform is not None:
            target_img = self.target_transform(target_img)

        # Get diagnosis label if available
        diagnosis = self.diagnosis_label[index] if self.diagnosis_label is not None else None # local variable to avoid overwriting instance variable
            
        # Combine output if specified
        if self.combine_output:
            return torch.cat([target_img, input_img], 0)
        
        # Return dictionary with input image, target image, and diagnosis label
        return {
            "input": input_img,  # return input image tensor
            "target": target_img,  # return target image tensor
            "diagnosis": diagnosis,  # return diagnosis label if available
        }
