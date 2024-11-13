
import torch
import torch.nn as nn
from models.basic_model import BasicModel
from layers.Attention_Family import CrossAttention

class Model(BasicModel):
    def __init__(self, configs, device, projection_dims=[768,768]):
        '''
        project_dim = dimension of pre-extract text and image embeddings
        '''
        super().__init__(configs, device)
        self.projection_dims = projection_dims
        self.flatten = nn.Flatten(start_dim=-2)
        # text embeddings
        self.txt_fc = nn.Linear(configs.d_txt, projection_dims[0])
        # adaptors
        self.relu = nn.ReLU()
        self.projection_mlp = nn.Sequential(
            nn.Linear(self.projection_dims[0], self.projection_dims[1]),
            nn.ReLU(),
            nn.Linear(self.projection_dims[0], self.projection_dims[1]))
        # linear layer
        nf = configs.d_model * int((configs.seq_len - configs.patch_len) / configs.patch_stride + 1)
        self.linear = nn.Linear(configs.d_model, configs.pred_len)
        # self.linear = nn.Linear(nf + self.projection_dims[1], configs.pred_len)
        # fusion
        self.crossatn = CrossAttention(configs.d_model, self.projection_dims[1])
        
    
    
    def forward(self, txt_enc, x_enc, x_mark_enc, x_dec, x_mark_dec, x_date, y_date):
        '''
        txt_enc: embedding of text information from freezed LLM: llama2 7b
        '''
        # Normalization [bs x seq_len x ndim=1]
        if self.args.revin:   
            x_enc = self .rev_in(x_enc, 'norm').to(self.device)
        else:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
            x_enc = x_enc / stdev
        # Encoder
        # time series model
        ts_emb = self.ts_model(x_enc, x_mark_enc, x_dec, x_mark_dec)  # [bs x ndim=1 x patch_num x d_model]
        # text 
        txt_emb = self.txt_fc(txt_enc) # txt_enc: [bs x d_txt=4096], txt_emb: [bs x projection_dim=768]
        txt_emb = self.projection_mlp(txt_emb) # [bs x projection_dim]
        txt_emb = txt_emb.reshape(txt_emb.shape[0], 1, txt_emb.shape[1]) # [bs x 1 x projection_dim]
   
        # fusion 
        enc_out, _ = self.crossatn(ts_emb, txt_emb) # [batch_size, 1, d_model]
        # enc_out = torch.concat((ts_emb, txt_emb), dim=2) # [ndim x bs x (patch_num x d_model + projection_dim)]
        # Decoder
        dec_out = self.linear(enc_out) # 
        dec_out = dec_out.permute(0, 2, 1)
        outputs = dec_out[:, -self.pred_len:, :]
        # De-Normalization
        if self.args.revin:
            outputs = self.rev_in(outputs, 'denorm').to(self.device)
        else:
            outputs = outputs * stdev
            outputs = outputs + means
        
        return outputs