# standard library
from pathlib import Path
from typing import Union, Optional, List, Dict, Tuple
# third party
from scipy.interpolate import UnivariateSpline, interp1d
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import lightning as pl
from torch.utils.data import random_split
# utils
from utils.llamas_utils import llamas_read_json, llamas_extend_lane
from utils.visualize import visualize_interp_lines, visualize_markers
from .const import START_CODE, END_CODE, LANE_CODE, PAD_CODE


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
        lanes_markers = self.load_markers(self.labels[idx])
        for i, markers in enumerate(lanes_markers):
            markers:list = np.array(markers).flatten().tolist() # [x1, y1, x2, y2, ...], 14 points
            # if i == 0: 
            #     markers.insert(0, 0); markers.insert(0, 0)
            markers = self.quantize_points(markers, (H, W))     # quantize points to [1, 1000]
            markers.append(LANE_CODE)
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


    def load_markers(self, path: Path) -> List[int]:
        """
        Load label from the given path.
        Args:
            path (Path): path to the label file
        Returns:
            List[str]: list of label strings
        """
        lanes_raw = llamas_read_json(path, min_lane_height=20)['lanes']

        lanes_markers = []
        for lane in lanes_raw:
            # extend lane
            lane = llamas_extend_lane(lane)
            # to list of points
            lane = self.marker_dict_to_lists(lane)
            # uniform sample points
            lanes_markers.append(self.sample_points(lane, 14))

        return lanes_markers
    

    @staticmethod
    def marker_dict_to_lists(lane: Dict) -> List[Tuple[int, int]]:
        """Convert single lane to list of points
        Args:
            lanes (json): lane markers 
        Returns:
            list: list of points
        """
        markers = lane['markers']
        points = []
        for marker in markers:
            points.append((int(marker['pixel_start']['x']), int(marker['pixel_start']['y'])))
            points.append((int(marker['pixel_end']['x']), int(marker['pixel_end']['y'])))
        return points
    

    @staticmethod
    def sample_points(points: List[Tuple[int, int]], num_points: int) -> List[Tuple[int, int]]:
        """Sample points from list of points
        Args:
            points (list): list of points
            num_points (int): number of points to sample
        Returns:
            list: list of sampled points
        """
        x, y = zip(*points)
        f = interp1d(y, x, kind='linear')
        y_new = np.linspace(min(y), max(y), num_points)
        x_new = f(y_new)

        return [(int(x), int(y)) for x, y in zip(x_new, y_new)]



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


# if __name__ == '__main__':
#     dataset = LLAMAS('/data/datasets/LLAMAS', 'valid')
#     print(len(dataset))
#     print(dataset.images[150])
#     image, input_sequence, target_sequence = dataset[150]
#     img = visualize_interp_lines(image, target_sequence)
#     cv2.imwrite('test.png', img)