# standard library
from pathlib import Path
from typing import Union, Optional, List, Dict, Tuple
# third party
import argparse
import torch
# lane2seq
from model.model import Lane2Seq
from dataset.llamas import LLAMASModule
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger


def main(args):
    # data module
    data_module = LLAMASModule(
        root=args.llamas_root,
        batch_size=args.batch_size,
    )

    # model
    model = Lane2Seq(reinforce_tuning=True).to('cuda')
    if not args.resume:
        model.load_state_dict(torch.load(args.ckpt)['state_dict'])

    # logger
    logger = TensorBoardLogger(save_dir='./logs',name='Lane2Seq-LLAMAS-Tuning', log_graph=True)

    # checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints/tune',
        filename='weight-llamas-tune-{epoch:02d}-{val_loss:.2f}-{val_f1:.2f}',
        save_top_k=10,
        save_last=True,
        monitor='val_f1',
        mode='min'
    )

    # trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu',
        devices=[0],
        logger=logger,
        callbacks=[checkpoint_callback]
    )

    # train
    ckpt_path = './checkpoints/tuning/last.ckpt' if args.resume else None
    trainer.fit(model=model, datamodule=data_module, ckpt_path=ckpt_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llamas_root', type=str, required=True)
    parser.add_argument('--ckpt', type=str, default='./checkpoints/train/last.ckpt')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_epochs', type=int, default=60)
    parser.add_argument('--resume', action='store_true', default=False)
    args = parser.parse_args()
    main(args)