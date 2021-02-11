import numpy as np
import pandas as pd
import torch
import os

from torch.utils.data import Dataset
from PIL import Image

def df_image_mask_path(root):
    remnants = []
    dataset = []
    
    # Loop over all folders
    for folder_name in os.listdir(root):
        try:
            nrgb = []
            nmask = []
            
            # Loop over all files
            for file_name in os.listdir(os.path.join(root, folder_name)):
                found = False
                
                # If the file_name containing the string of 'mask'
                if 'mask' in file_name.lower():
                    for name in nrgb:
                        if name.split('.')[0] == (file_name.split('.')[0])[:-len('_mask')]:
                            nrgb.remove(name)
                            found = True
                            break
                    if found:
                        mask_path = os.path.join(root, folder_name, file_name)
                        containing_mask = 0
                        if np.abs(np.max(np.array(Image.open(mask_path)))) > 0:
                            containing_mask = 1
                        
                        dataset.append([os.path.join(root, folder_name, name),
                                        os.path.join(root, folder_name, file_name),
                                        containing_mask])
                    else: nmask.append(file_name)
                
                # If the file_name is a name of rgb file
                else:
                    for name in nmask:
                        if file_name.split('.')[0] == (name.split('.')[0])[:-len('_mask')]:
                            nmask.remove(name)
                            found = True
                            break
                    if found:
                        mask_path = os.path.join(root, folder_name, name)
                        containing_mask = 0
                        if np.abs(np.max(np.array(Image.open(mask_path)))) > 0:
                            containing_mask = 1
                            
                        dataset.append([os.path.join(root, folder_name, file_name),
                                        os.path.join(root, folder_name, name),
                                        containing_mask])
                    else: nrgb.append(file_name)
        except:
            remnants.append(folder_name)
    
    print('The remnants:', remnants)

    return pd.DataFrame(dataset, columns=['path_img_rgb', 'path_img_mask', 'mask'])
                    
                    
class BrainMRIDataset(Dataset):
    """Brain Image Dataset."""

    def __init__(self,  df, transform=None):
        """
        Args:
            df (dataframe): Pandas dataframe
            transform (callable, optional): Optional transform on samples
        """
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get the image & mask path
        row = self.df.iloc[idx]
        
        # Open the file
        image = Image.open(row['path_img_rgb'])
        mask = Image.open(row['path_img_mask'])

        # Image augmentation / transformation
        if self.transform:
            image, mask = self.transform([image, mask])
        
        # Normalize the image and change to tensor
        image = BrainMRIDataset.normalize(image, 3)
        mask = BrainMRIDataset.normalize(mask, 1)
        
        return image, mask
    
    @staticmethod
    def normalize(image, channels):
        size = image.size
        image = np.array(image).reshape((size[0], size[1], channels)) / 255.
        
        # Conv2D in the model expect channel-first
        image = np.swapaxes(image, -2, -1)
        image = np.swapaxes(image, -3, -2)
        
        return torch.from_numpy(image)