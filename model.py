# standard library
from pathlib import Path
from typing import Union, Optional, List, Dict, Tuple
# third party
import cv2
import torch
import torch.nn as nn
import lightning as pl
import timm
# objective
from loss import WeightedCrossEntropyLoss


class Lane2Seq(pl.LightningModule):
    def __init__(
        self,
        transformer_blocks: Optional[int] = 2,
        hidden_size: Optional[int] = 256,
        attention_heads: Optional[int] = 8,
        feedforward_size: Optional[int] = 1024,
        n_bins: Optional[int] = 1000,
        mlp_layers: Optional[int] = 3,
    ) -> None:
        super(Lane2Seq, self).__init__()

        # ViT-Base backbone encoder [3, 24, 24] -> [b, 768]
        self.backbone = timm.create_model(
            f'vit_base_patch16_224.mae', 
            pretrained=True,
            num_classes=0, # no classification
        )

        # bottleneck
        self.bottleneck = nn.Linear(in_features=768, out_features=hidden_size) \
            if hidden_size != 768 else nn.Identity()

        # transformer decoder
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=hidden_size,                # 256
                nhead=attention_heads,              # 8
                dim_feedforward=feedforward_size,   # 1024
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=transformer_blocks,  # 2
        )

        # vocabulary embedding
        self.embedding = nn.Embedding(
            num_embeddings=n_bins + 7,      # 1000 + 3 (for <start>, <end>, <lane>) + 3 (for <segment>, <anchor>, <parameter>) + 1 (for <pad>)
            embedding_dim=hidden_size,      # 256
        )

        # feature to likelihood
        self.mlp_layer_list = []
        for _ in range(mlp_layers - 1):
            self.mlp_layer_list.append(nn.Linear(in_features=hidden_size, out_features=hidden_size))
            self.mlp_layer_list.append(nn.ReLU(True))
        self.mlp_layer_list.append(
            nn.Linear(in_features=hidden_size, out_features=n_bins + 7))
        
        self.mlp = nn.Sequential(*self.mlp_layer_list)

        # objective
        self.objective = WeightedCrossEntropyLoss()

    
    def configure_optimizers(self):
        # AdamW optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=1e-4,
            weight_decay=1e-2,
        )
        return optimizer


    def forward(
        self,
        img: torch.Tensor, # [b, c, 224, 224]
        seq: torch.Tensor, # [b, seq_len]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # feature extraction
        feat = self.backbone.forward_features(img)       # [b, 197, 768]
        mem = self.bottleneck(feat)     # [b, patch_num, hidden_size]
        # embedding
        emb = self.embedding(seq)       # [b, seq_len, hidden_size]
        # decode
        out = self.decoder(tgt=emb, memory=mem)  # [b, seq_len, hidden_size]
        # to likelihood
        out = self.mlp(out)                 # [b, seq_len, n_bins + 7]
        # out = torch.softmax(out, dim=-1)    # [b, seq_len, n_bins + 7]
        return out


    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        images, input_sequences, target_sequences = batch
        # forward pass
        predict_sequences = self.forward(images, input_sequences) # [b, seq_len, n_bins + 7]
        # compute loss
        loss = self.objective(predict_sequences, target_sequences)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        images, input_sequences, target_sequences = batch
        # forward pass
        predict_sequences = self.forward(images, input_sequences)
        # compute loss
        loss = self.objective(predict_sequences, target_sequences)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)


    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        raise NotImplementedError("Test step is not implemented yet.")


    @staticmethod
    def likelihood_to_quantized_points(
        x: torch.Tensor, # [b, seq_len, n_bins + 7]
    ) -> List[int]:
        """
        Convert likelihood to quantized points.
        Args:
            out (torch.Tensor): likelihood [seq_len, n_bins + 7]
        Returns:
            List[int]: quantized points
        """
        # softmax
        x = torch.softmax(x, dim=-1)    # [b, seq_len, n_bins + 7]
        # get the most likely
        x = torch.argmax(x, dim=-1)     # [b, seq_len]
        return x


if __name__ == '__main__':
    # test model
    model = Lane2Seq()
    # print(model)
    # test data
    img = torch.randn(2, 3, 224, 224)
    seq = torch.randint(0, 1000, (2, 5))
    out = model.forward(img, seq)
    print(out.shape)  # [2, 5, 1003]