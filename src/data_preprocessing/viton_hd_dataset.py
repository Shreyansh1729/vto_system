import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class VitonHDDataset(Dataset):
    """
    Custom PyTorch Dataset for loading the VITON-HD dataset.
    This dataset is pre-processed and provides pairs of:
    - Person image
    - Garment image
    - Ground truth try-on image (which is the original person image)
    - Control maps: human parsing, OpenPose skeleton
    """
    def __init__(self, data_dir, pairs_file, image_size=(512, 384)):
        """
        Args:
            data_dir (str): The root directory of the dataset (e.g., '.../datasets/test').
            pairs_file (str): The name of the file defining the pairs (e.g., 'test_pairs.txt').
            image_size (tuple): The target size to resize images to (height, width).
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.pairs_path = os.path.join(data_dir, pairs_file)
        
        # Load the pairs from the text file
        self.data_pairs = self._load_pairs()

        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) # Normalize to [-1, 1] for all images
        ])
        
        # Define mask/pose map transformations (no normalization)
        self.map_transform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
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
        A sample consists of the person image, cloth image, and all control maps.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the filenames for the current pair
        person_fn, cloth_fn = self.data_pairs.iloc[idx]

        # --- Load and Transform all required images ---
        
        # Person Image (Ground Truth)
        person_img_path = os.path.join(self.data_dir, 'image', person_fn)
        person_image = Image.open(person_img_path).convert('RGB')
        
        # Garment Image
        cloth_img_path = os.path.join(self.data_dir, 'cloth', cloth_fn)
        cloth_image = Image.open(cloth_img_path).convert('RGB')

        # OpenPose Image (Pose Map)
        pose_img_path = os.path.join(self.data_dir, 'openpose_img', person_fn.replace('.jpg', '_rendered.png'))
        pose_map = Image.open(pose_img_path).convert('RGB')
        
        # Human Parsing Segmentation Mask
        parse_img_path = os.path.join(self.data_dir, 'image-parse-v3', person_fn.replace('.jpg', '.png'))
        parse_map = Image.open(parse_img_path).convert('L') # Grayscale

        # --- Apply transformations ---
        person_image_tensor = self.transform(person_image)
        cloth_image_tensor = self.transform(cloth_image)
        pose_map_tensor = self.transform(pose_map)
        parse_map_tensor = self.map_transform(parse_map)

        # The 'target' for our model is the original person's image
        target_image_tensor = person_image_tensor

        return {
            'person_image': person_image_tensor,
            'cloth_image': cloth_image_tensor,
            'pose_map': pose_map_tensor,
            'parse_map': parse_map_tensor,
            'target_image': target_image_tensor,
            'person_filename': person_fn
        }