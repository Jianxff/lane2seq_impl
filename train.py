# standard library
from pathlib import Path
from typing import Union, Optional, List, Dict, Tuple
# third party
import argparse
# lane2seq
from model import Lane2Seq
from llamas_dataset import LLAMASModule
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
    model = Lane2Seq().to('cuda')

    # logger
    logger = TensorBoardLogger(save_dir='./logs',name='Lane2Seq-LLAMAS', log_graph=True)

    # checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints',
        filename='model-llamas-{epoch:02d}-{val_loss:.2f}',
        save_last=True,
        monitor='val_loss',
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
    trainer.fit(model=model, datamodule=data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llamas_root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_epochs', type=int, default=15)
    args = parser.parse_args()
    main(args)