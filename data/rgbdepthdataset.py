import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
import numpy as np
from torchvision.transforms import Transform


class ConcatenateRGBDepth(Transform):
    def __init__(self):
        super().__init__()

    def __call__(self, rgb_image, depth_image_np):
        # Normalize depth image to [0, 1]
        depth_image_np = (depth_image_np - depth_image_np.min()) / (depth_image_np.max() - depth_image_np.min())

        # Resize depth image to match the spatial dimensions of the RGB image
        rgb_image_np = rgb_image.permute(0, 2, 3, 1).numpy()
        batch_size, height, width, _ = rgb_image_np.shape
        depth_image_resized = cv2.resize(depth_image_np, (width, height))

        # Convert the resized depth image to a PyTorch tensor and add a channel dimension
        depth_image_resized = torch.from_numpy(depth_image_resized).unsqueeze(0).unsqueeze(1)

        # Concatenate the RGB image and resized depth image
        concatenated_image = torch.cat((rgb_image, depth_image_resized), dim=1)

        return concatenated_image


class RGBDepthDataset(Dataset):
    def __init__(self, rgb_dir, depth_dir, transform=None):
        """
        This dataset assumes that the RGB and depth images 
        are sorted alphabetically and correspond to each other.
        Make sure your image filenames are properly sorted and aligned.
        """
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.transform = transform

        self.rgb_filenames = sorted(os.listdir(rgb_dir))
        self.depth_filenames = sorted(os.listdir(depth_dir))

        assert len(self.rgb_filenames) == len(self.depth_filenames), "Number of RGB and depth images must be equal."

    def __len__(self):
        return len(self.rgb_filenames)

    def __getitem__(self, idx):
        rgb_path = os.path.join(self.rgb_dir, self.rgb_filenames[idx])
        depth_path = os.path.join(self.depth_dir, self.depth_filenames[idx])

        # Load the RGB image using PIL and convert it to a PyTorch tensor
        rgb_image = Image.open(rgb_path).convert("RGB")
        rgb_image = transforms.ToTensor()(rgb_image)

        # Load the depth image using numpy and convert it to a float32 array
        depth_image_np = np.array(Image.open(depth_path)).astype(np.float32)

        if self.transform:
            sample = self.transform(rgb_image, depth_image_np)
        else:
            sample = {'rgb': rgb_image, 'depth': depth_image_np}

        return sample
