# standard library
from pathlib import Path
from typing import Union, Optional, List, Dict, Tuple
# third party
import argparse
import cv2
import torch
# lane2seq
from model.model import Lane2Seq
from utils.visualize import visualize_interp_lines, visualize_markers


def main(args):
    # data
    image = args.image
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    H, W = image.shape[:2]
    image = cv2.resize(image, (224, 224))
    image = torch.from_numpy(image).float().permute(2, 0, 1).to('cuda') # [3, H, W]

    # model
    model = Lane2Seq.load_from_checkpoint(args.ckpt).to('cuda')
    model.eval()

    # predict
    seq = model.predict(image)
    # draw
    img = visualize_interp_lines(image, seq[0])
    cv2.imwrite('predict.png', img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='./test.png')
    parser.add_argument('--ckpt', type=str, default='./checkpoints/tune/last.ckpt')
    args = parser.parse_args()

    main(args)