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
from .loss import CrossEntropy
from utils.metric import batch_evaluate
from dataset.const import END_CODE, START_CODE, PAD_CODE, SPECIAL_CODE_NUM
# reinforcement learning
from .reinforce import REINFORCE


class Lane2Seq(pl.LightningModule):
    def __init__(
        self,
        transformer_blocks: Optional[int] = 2,
        hidden_size: Optional[int] = 256,
        attention_heads: Optional[int] = 8,
        feedforward_size: Optional[int] = 1024,
        n_bins: Optional[int] = 1000,
        mlp_layers: Optional[int] = 3,
        reinforce_tuning: Optional[bool] = False,
    ) -> None:
        super(Lane2Seq, self).__init__()

        # ViT-Base backbone encoder [3, 224, 224] -> [b, 197, 768]
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
        self.embedding = nn.Sequential(
            nn.Embedding(
                num_embeddings=n_bins + SPECIAL_CODE_NUM,      # 1000 + 3 (for <start>, <end>, <lane>) + 1 (for <pad>)
                embedding_dim=hidden_size,      # 256
                padding_idx=PAD_CODE,          # 0
            ),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1),
        )

        # feature to likelihood
        self.mlp_layer_list = []
        for _ in range(mlp_layers - 1):
            self.mlp_layer_list.append(nn.Linear(in_features=hidden_size, out_features=hidden_size))
            self.mlp_layer_list.append(nn.ReLU(True))
        self.mlp_layer_list.append(
            nn.Linear(in_features=hidden_size, out_features=n_bins + SPECIAL_CODE_NUM))
        
        self.mlp = nn.Sequential(*self.mlp_layer_list)

        # objective
        self.reinforce_tuning = reinforce_tuning
        if self.reinforce_tuning:
            self.objective = REINFORCE()
        else:
            self.objective = CrossEntropy()

    
    def configure_optimizers(self):
        # AdamW optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=1e-6 if self.reinforce_tuning else 1e-4,
            weight_decay=1e-2,
        )
        return optimizer

    
    def forward_encoder(
        self, 
        img: torch.Tensor # [b, 3, 224, 224]
    ) -> torch.Tensor:
        # feature extraction
        feat = self.backbone.forward_features(img)       # [b, 197, 768]
        # skip CLS token
        mem = self.bottleneck(feat[:, 1:, :])     # [b, 196, hidden_size]
        return mem
    

    def forward_decoder(
        self,
        seq: torch.Tensor, # [b, seq_len]
        mem: torch.Tensor, # [b, 196, hidden_size]
        pad_mask: Optional[torch.Tensor] = None, # [b, seq_len]
    ) -> torch.Tensor:
        # embedding
        emb = self.embedding(seq)       # [b, seq_len, hidden_size]
        # transformer decode
        seq_len = seq.size(-1)
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len, device=seq.device)
        out = self.decoder(tgt=emb, memory=mem,
            tgt_mask=tgt_mask, tgt_key_padding_mask=pad_mask)  # [b, seq_len, hidden_size]
        # to likelihood
        out = self.mlp(out)                 # [b, seq_len, n_bins + k]
        return out


    @torch.no_grad()
    def predict(
        self,
        img: torch.Tensor, # [b, 3, 224, 224]
        max_seq_len: Optional[int] = (16 * (14 * 2 + 1) + 1), # 16 lanes <end>
    ) -> torch.Tensor:
        # dim check
        if len(img.shape) == 3: img = img.unsqueeze(0)
        # variable
        batch_size = img.size(0)
        device = img.device
        # feature encode
        mem = self.forward_encoder(img) # [b, patch_num, hidden_size]
        # initial seq [b, 2] (<start>, <anchor>)
        seq = torch.tensor([[START_CODE]]).repeat(batch_size, 1).to(device)
        # full prediction
        for _ in range(max_seq_len - 2):
            out = self.forward_decoder(seq=seq, mem=mem) # [b, seq, n_bins + k]
            # next token
            next_token_logit = out[:, -1]   # [b, n_bins + k]
            next_token = self.filter_argmax(next_token_logit).unsqueeze(-1)  # [b, 1]
            seq = torch.cat([seq, next_token], dim=-1)      
            # check end if batch_size is one  
            if batch_size == 1:
                if next_token[0].item() == END_CODE:
                    break
        return seq
    

    @staticmethod
    def filter_argmax(x: torch.Tensor) -> torch.Tensor: # [b, seq_len, n_bins + k] -> [b, seq_len]
        """
        Convert likelihood to quantized points.
        Args:
            out (torch.Tensor): likelihood [seq_len, n_bins + k]
        Returns:
            torch.Tensor: index [seq_len]
        """
        # softmax
        x = torch.softmax(x, dim=-1)    # [b, seq_len, n_bins + k]
        # get index of the most likely
        x = torch.argmax(x, dim=-1)     # [b, seq_len]
        return x


    #####################################################################
    ##                     Lightning Module                            ##
    #####################################################################

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        images, input_sequences, target_sequences, padding_mask = batch
        
        # forward pass
        mem = self.forward_encoder(images) # [b, patch_num, hidden_size]
        out = self.forward_decoder(seq=input_sequences, mem=mem, pad_mask=padding_mask) # [b, seq_len, n_bins + k]
        
        # training loss 
        loss = self.objective.compute(x=out, target=target_sequences)
        # predict_sequences = self.filter_argmax(out)
        # metirc = batch_evaluate(predict_sequences, target_sequences)
        # self.log('train_F1', metirc['f1'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log('train_Acc', metirc['precision'], on_step=True, on_epoch=True, logger=True)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        images, input_sequences, target_sequences, padding_mask = batch
        
        # forward pass
        mem = self.forward_encoder(images) # [b, patch_num, hidden_size]
        out = self.forward_decoder(seq=input_sequences, mem=mem, pad_mask=padding_mask) # [b, seq_len, n_bins + k]
        
        # compute loss
        loss = self.objective.compute(out, target_sequences)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # compute evaluation metric
        predict_sequences = self.filter_argmax(out)
        metirc = batch_evaluate(predict_sequences, target_sequences)
        self.log('val_f1', metirc['f1'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', metirc['precision'], on_step=True, on_epoch=True, logger=True)


    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        images, _, target_sequences, _ = batch
        
        # forward pass
        predict_sequences = self.predict(img=images) # [b, seq_len]
        
        # compute evaluation metric
        metirc = batch_evaluate(predict_sequences, target_sequences)
        self.log('test_f1', metirc['f1'], on_step=True, prog_bar=True, logger=True)
        self.log('test_acc', metirc['precision'], on_step=True, logger=True)
