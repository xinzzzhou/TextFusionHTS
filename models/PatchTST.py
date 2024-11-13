import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.Attention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
# class FlattenHead(nn.Module):
#     def __init__(self, n_vars, nf, target_window, head_dropout=0):
#         super().__init__()
#         # self.n_vars = n_vars
#         self.flatten = nn.Flatten(start_dim=-2)
#         self.linear = nn.Linear(nf, target_window)
#         self.dropout = nn.Dropout(head_dropout)

#     def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
#         x = self.flatten(x)
#         x = self.linear(x)
#         x = self.dropout(x)
#         return x


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs, patch_len=1, stride=1):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = 'long_term_forecast'
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride
        self.args = configs
        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.seq_len, patch_len, stride, padding, configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention1), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation1
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        ##########  here is enc_in   ####################
        # Prediction Head
        # self.flatten = nn.Flatten(start_dim=-2)
        # self.head_nf = configs.d_model * \
        #                int((configs.seq_len - patch_len) / stride + 2)
        # self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
        #                             head_dropout=configs.dropout)
        
        
    def forecast(self, x_enc):
        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)
        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        # enc_out = enc_out.permute(0, 1, 3, 2)
        # enc_out = self.flatten(enc_out)
        # Decoder
        # dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        # dec_out = dec_out.permute(0, 2, 1)
        return enc_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc)
        return dec_out  # [bs x nvars x patch_num x d_model]
