# standard library
from pathlib import Path
from typing import Union, Optional, List, Dict, Tuple
# third party
import argparse
import cv2
import torch
# lane2seq
from model.model import Lane2Seq
from dataset.llamas import LLAMAS
from utils.metric import batch_evaluate
from utils.visualize import visualize_interp_lines, visualize_markers


def main(args):
    # data module
    dataset = LLAMAS(root=args.llamas_root, split='valid')

    # read data
    data_idx = args.idx
    image, _, gt_seq = dataset[data_idx]
    image = image.unsqueeze(0).to('cuda')

    # model
    model = Lane2Seq.load_from_checkpoint(args.ckpt).to('cuda')
    model.eval()

    # predict
    pred_seq = model.predict(image)

    # draw GT and predict
    img_raw = cv2.resize(image[0].permute(1, 2, 0).detach().cpu().numpy(), dsize=(1276, 717))
    img_gt = visualize_interp_lines(image[0], gt_seq)
    img_pred = visualize_markers(image[0], pred_seq[0])

    # metric
    metric = batch_evaluate(pred_seq, gt_seq[None])
    print(metric)

    # concatenate
    if args.mode == 'h':
        img = cv2.hconcat([img_raw, img_gt, img_pred])
    elif args.mode == 'v':
        img = cv2.vconcat([img_raw, img_gt, img_pred])
    else:
        raise ValueError(f'Invalid mode: {args.mode}')

    cv2.imwrite(args.out, img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llamas_root', type=str, required=True)
    parser.add_argument('--idx', type=int, default=0)
    parser.add_argument('--out', type=str, default='./result.png')
    parser.add_argument('--ckpt', type=str, default='./checkpoints/tune/last.ckpt')
    parser.add_argument('--mode', type=str, default='h')
    args = parser.parse_args()
    
    main(args)