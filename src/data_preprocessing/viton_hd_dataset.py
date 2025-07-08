import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class VitonHDDataset(Dataset):
    """
    Custom PyTorch Dataset for loading the VITON-HD dataset.
    """
    # UPDATED __init__ function
    def __init__(self, data_dir, pairs_file_path, image_size=(512, 384)):
        """
        Args:
            data_dir (str): The root directory for image folders (e.g., '.../test').
            pairs_file_path (str): The full, absolute path to the pairs file.
            image_size (tuple): The target size to resize images to (height, width).
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.pairs_path = pairs_file_path # Use the direct path now
        self.clip_image_size = clip_image_size # Add CLIP specific size
        self.data_pairs = self._load_pairs()

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        self.map_transform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

        self.clip_transform = transforms.Compose([
            transforms.Resize(self.clip_image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            # This is the normalization CLIP was trained with
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ])


    def _load_pairs(self):
        """Loads the person-garment pairs from the provided text file."""
        try:
            return pd.read_csv(self.pairs_path, sep=' ', header=None, names=['person_img', 'cloth_img'])
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find the pairs file at {self.pairs_path}")

    def __len__(self):
        """Returns the total number of pairs in the dataset."""
        return len(self.data_pairs)

    def __getitem__(self, idx):
        """
        Fetches a single data sample.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        person_fn, cloth_fn = self.data_pairs.iloc[idx]
        
        person_img_path = os.path.join(self.data_dir, 'image', person_fn)
        person_image = Image.open(person_img_path).convert('RGB')
        
        cloth_img_path = os.path.join(self.data_dir, 'cloth', cloth_fn)
        cloth_image = Image.open(cloth_img_path).convert('RGB')

        pose_img_path = os.path.join(self.data_dir, 'openpose_img', person_fn.replace('.jpg', '_rendered.png'))
        pose_map = Image.open(pose_img_path).convert('RGB')
        
        parse_img_path = os.path.join(self.data_dir, 'image-parse-v3', person_fn.replace('.jpg', '.png'))
        parse_map = Image.open(parse_img_path).convert('L')

        person_image_tensor = self.transform(person_image)
        #cloth_image_tensor = self.transform(cloth_image)
        pose_map_tensor = self.transform(pose_map)
        parse_map_tensor = self.map_transform(parse_map)
        clip_cloth_image_tensor = self.clip_transform(cloth_image)
        target_image_tensor = person_image_tensor

        return {
            'person_image': person_image_tensor,
            #'cloth_image': cloth_image_tensor,
            'pose_map': pose_map_tensor,
            'parse_map': parse_map_tensor,
            'clip_cloth_image': clip_cloth_image_tensor, # New key for CLIP
            'target_image': target_image_tensor,
            'person_filename': person_fn
        }