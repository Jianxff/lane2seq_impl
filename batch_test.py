# standard library
from pathlib import Path
from typing import Union, Optional, List, Dict, Tuple
# third party
import argparse
# lane2seq
from model.model import Lane2Seq
from dataset.llamas import LLAMASModule
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger


def main(args):
    # data module
    data_module = LLAMASModule(
        root=args.llamas_root,
        batch_size=args.batch_size,
    )

    # model
    model = Lane2Seq().to('cuda')

    # logger
    logger = TensorBoardLogger(save_dir='./logs',name='Lane2Seq-LLAMAS-Test', log_graph=True)

    # trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[0],
        logger=logger,
    )

    # test
    trainer.test(model=model, datamodule=data_module, ckpt_path=args.ckpt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llamas_root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--ckpt', type=str, default='./checkpoints/tune/last.ckpt')
    args = parser.parse_args()
    main(args)