# standard library
from pathlib import Path
from typing import Union, Optional, List, Dict, Tuple
# third party
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import lightning as pl
from torch.utils.data import random_split
# utils
from utils.llamas_utils import llamas_sample_points_horizontal, llamas_extend_lane
from utils.visualize import vis_lane_circle, vis_lane_line
from .const import START_CODE, END_CODE, TOKEN_LANE, PAD_CODE


class LLAMAS(Dataset):
    def __init__(
        self, 
        root: Union[str, Path], 
        split: str = 'train',
        image_size: int = 224,
    ):
        super(LLAMAS, self).__init__()
        self.root = Path(root).absolute()
        self.split = split
        self.image_size = image_size

        self.data_dir = self.root / split
        self.images = self.glob_file_recursive(self.data_dir, '.png')
        self.images = sorted(list(self.images))

        if split in ['train', 'valid']:
            self.data_label_dir = self.root / 'labels' / split
            self.labels = self.glob_file_recursive(self.data_label_dir, '.json')
            self.labels = sorted(list(self.labels))
            
            assert len(self.images) == len(self.labels), 'number of images and labels must be the same'


    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        """
        Args:
            idx (int): index of the image
        Returns:
            Tuple[np.ndarary, List[List[int]]]: image tensor and label strings
        Note:
            image: [3, H, W]
            label: List[List[int]] (list of lanes)
        """
        # load image
        image = self.load_image(self.images[idx])
        H, W = image.shape[:2]

        # to tensor
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = torch.from_numpy(image).float().permute(2, 0, 1) # [3, H, W]

        if self.split == 'test':
            return (image, None, None)
        
        # load label
        format_specific_sequence = []
        lanes_markers = self.load_json(self.labels[idx], (H, W))
        for i, markers in enumerate(lanes_markers):
            markers:list = np.array(markers).flatten().tolist() # [x1, y1, x2, y2, ...], 14 points
            # if i == 0: 
            #     markers.insert(0, 0); markers.insert(0, 0)
            markers = self.quantize_points(markers, (H, W))     # quantize points to [1, 1000]
            markers.append(TOKEN_LANE)
            format_specific_sequence.extend(markers)
        # format_specific_sequence.insert(0, TOKEN_ANCHOR)
        
        # input sequence
        input_sequence = [START_CODE]
        input_sequence.extend(format_specific_sequence)
        input_sequence = np.array(input_sequence).flatten()
        input_sequence = torch.from_numpy(input_sequence).type(torch.int64)

        # target sequence
        target_sequence = format_specific_sequence.copy()
        target_sequence.append(END_CODE)
        target_sequence = np.array(target_sequence).flatten()
        target_sequence = torch.from_numpy(target_sequence).type(torch.int64)

        return (image, input_sequence, target_sequence)
    

    def quantize_points(self, points:List[int], image_size:Tuple[int, int]):
        """
        Format points to the required format to [1, 1000].
        Args:
            points (List[int]): list of points
        Returns:
            List[int]: formatted points
        """
        H, W = image_size[:2]
        for i in range(0, len(points), 2):
            x, y = points[i], points[i + 1]
            points[i] = int(x / W * 999) + 1
            points[i + 1] = int(y / H * 999) + 1
        return points

    
    def glob_file_recursive(self, directory: Path, suffix: str) -> List[Path]:
        """
        Recursively search for files with the given suffix in the given directory.
        Args:
            directory (Path): directory to search
            suffix (str): file suffix to search
        Returns:
            List[Path]: list of files with the given suffix
        """
        if not directory.exists():
            raise FileNotFoundError(f'{directory} does not exist')
        files = []
        for d in directory.iterdir():
            if d.is_dir():
                files.extend(self.glob_file_recursive(d, suffix))
            if d.is_file() and d.suffix == suffix:
                files.append(d)
        return files
    

    def load_image(self, path: Path) -> torch.Tensor:
        """
        Load image from the given path.
        Args:
            path (Path): path to the image file
        Returns:
            torch.Tensor: image tensor
        """
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        return image


    def load_json(self, path: Path, image_size: Tuple[int, int]) -> List[int]:
        """
        Load label from the given path.
        Args:
            path (Path): path to the label file
        Returns:
            List[str]: list of label strings
        """
        H, W = image_size[:2]
        with open(path) as f:
            json_data_raw = json.load(f)
        
        lanes = []
        for lane in json_data_raw['lanes']:
            lane = self.convert_str_single_lane(lane)
            lane = llamas_extend_lane(lane)
            x_points = llamas_sample_points_horizontal(lane)
            points = [(x, y) for x, y in zip(x_points, range(H)) if x >= 0]
            if len(points) >= 14:
                # randomly sample 14 points
                points = [points[i] for i in range(0, len(points), len(points) // 14)]
                lanes.append(points)

        return lanes

    @staticmethod
    def convert_str_single_lane(lane):
        keys_to_float = ['world_start', 'world_end']
        keys_to_int = ['pixel_start', 'pixel_end']
        for marker in lane['markers']:
            for key in keys_to_float:
                if key in marker:
                    marker[key] = {k: float(v) for k, v in marker[key].items()}
            for key in keys_to_int:
                if key in marker:
                    marker[key] = {k: int(v) for k, v in marker[key].items()}
        return lane


def collate_fn(batch: torch.Tensor):
    """
    Collate function for DataLoader.
    Args:
        batch (torch.Tensor): batch of data
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: image tensor, input sequence, target sequence
    """
    images = [item[0] for item in batch]
    inputs = [item[1] for item in batch]
    targets = [item[2] for item in batch]
    device = images[0].device
    # take maximum size of input and target sequence
    max_input_size = max([len(input) for input in inputs])
    max_target_size = max([len(target) for target in targets])

    # mask
    input_padding_mask = torch.zeros(len(inputs), max_input_size, dtype=torch.bool).to(device)
    # pad input and target sequence
    for i in range(len(inputs)):
        inputs[i] = torch.cat([inputs[i], 
                torch.zeros(max_input_size - len(inputs[i]),dtype=torch.int64).to(device)])
        input_padding_mask[i, len(inputs[i]):] = True
        targets[i] = torch.cat([targets[i],
                torch.zeros(max_target_size - len(targets[i]), dtype=torch.int64).to(device)])
    
    images = torch.stack(images)
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)
    return images, inputs, targets, input_padding_mask



class LLAMASModule(pl.LightningDataModule):
    def __init__(
        self,
        root: Union[str, Path],
        batch_size: int = 32,
        device: Union[str, torch.device] = 'cuda',
    ) -> None:
        super(LLAMASModule, self).__init__()
        self.root = Path(root).absolute()
        self.batch_size = batch_size
        self.device = device


    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit':
            self.train_dataset = LLAMAS(root=self.root, split='train')
            # random split
            n = len(self.train_dataset)
            n_train = int(n * 0.9)
            n_val = n - n_train
            # manual seed to ensure the same split
            torch.manual_seed(42)
            self.train_dataset, self.val_dataset = random_split(self.train_dataset, [n_train, n_val])
            
        elif stage == 'test' or stage == 'predict':
            self.test_dataset = LLAMAS(root=self.root, split='valid')
        

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
        )
    
    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4
        )


if __name__ == '__main__':
    dataset = LLAMAS('/data/datasets/LLAMAS', 'valid')
    print(len(dataset))
    print(dataset.images[150])
    image, input_sequence, target_sequence = dataset[150]
    img = vis_lane_line(image, target_sequence)
    cv2.imwrite('test.png', img)