import torch
import torch.nn as nn
import torch.nn.functional as F
from .autoformer_layers.Embed import DataEmbedding, DataEmbedding_wo_pos, DataEmbedding_wo_posandtime
from .autoformer_layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from .autoformer_layers.Autoformer_EncDec import Encoder, Decoder, DecoderLayer, my_Layernorm, series_decomp
from .autoformer_layers.Autoformer_EncDec import EncoderLayer, EncoderLayerNoAutoCorrelation, EncoderLayerNoDetrending

import math
import numpy as np
from .build import MODEL
from timm.models.vision_transformer import VisionTransformer
from .heads import FCActHead

@MODEL.register('Autoformer')
class AutoformerEncoder(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, cfg):
        super(AutoformerEncoder, self).__init__()
        configs = cfg.model
        self.output_attention = configs.output_attention

        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_posandtime(configs.enc_in, configs.d_model, configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        self.vit = VisionTransformer(img_size=(300, 512), in_chans=1, depth=6)
        self.head = FCActHead('relu')
        
        
    def forward_feature(self, x):
        enc_out = self.enc_embedding(x)
        enc_out, attns = self.encoder(enc_out)
        enc_out = enc_out.unsqueeze(1)
        return enc_out
        
    def forward_pred(self, x):
        y = self.vit(x)
        y = self.head(y)
        return y
    
    def forward(self, x):
        f = self.forward_feature(x)
        y = self.forward_pred(f)
        return y
    
    def loss_function(self, pred, label):
        return F.mse_loss(pred, label)

@MODEL.register('AutoformerMultiTask')
class AutoformerMultiTask(AutoformerEncoder):
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def loss_function(self, pred, label, feature, bvps, writer, iter):
        l1 =  super().loss_function(pred, label)
        l2 = F.mse_loss(torch.mean(feature.squeeze(1), dim=2), bvps)
        writer.add_scalar('Loss/train_bpm', l1.item(), iter)
        writer.add_scalar('Loss/train_bvp', l2.item(), iter)
        loss = (l1 + l2)/2
        return loss
    
@MODEL.register('AutoformerNoAutoCorrelation')
class AutoformerNoAutoCorrelation(AutoformerEncoder):
    def __init__(self, cfg):
        super(AutoformerNoAutoCorrelation, self).__init__(cfg)
        configs = cfg.model
        self.output_attention = configs.output_attention

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayerNoAutoCorrelation(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )


@MODEL.register('AutoformerNoDetrending')
class AutoformerNoDetrending(AutoformerEncoder):
    def __init__(self, cfg):
        super(AutoformerNoDetrending, self).__init__(cfg)
        configs = cfg.model
        self.output_attention = configs.output_attention

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayerNoDetrending(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
    
if __name__ == '__main__':
    class Cfg:
        pass
    cfg = Cfg()
    
    cfg.seq_len = 96
    cfg.label_len = 48
    cfg.pred_len = 96
    cfg.output_attention  = False
    cfg.d_ff=2048
    cfg.d_layers=1
    cfg.d_model=512
    cfg.moving_avg = 25
    cfg.enc_in = 63
    cfg.factor = 1
    cfg.embed = 'timeF'
    cfg.freq='h'
    cfg.dropout=0.05
    cfg.e_layers=2
    cfg.dec_in=7
    cfg.n_heads=8
    cfg.activation='gelu'
    cfg.c_out=7
    cfg.model = cfg
    x = torch.randn(3, 300, 63)
    model = AutoformerEncoder(cfg)
    y = model(x)
    import pdb
    pdb.set_trace()
    print(model)