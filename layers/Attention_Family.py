import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange, repeat
import torch.nn.functional as F



class CrossAttention(nn.Module):
    def __init__(self, d_model, d_txt, n_heads=8):
        super(CrossAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        
        # Linear layers to project Q, K, V
        self.query_proj = nn.Linear(d_txt, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        
        # Scaling factor
        self.scale = torch.sqrt(torch.tensor(d_model, dtype=torch.float32))

    def forward(self, seq_features, text_features):
        """
        seq_features: [batch_size, 1, patch_num, d_model]
        text_features: [batch_size, 1, d_txt]
        """
        batch_size = seq_features.size(0)
        patch_num = seq_features.size(2)
        
        # Project Q, K, V
        Q = self.query_proj(text_features)  # [batch_size, 1, d_model]
        K = self.key_proj(seq_features)     # [batch_size, 1, patch_num, d_model]
        V = self.value_proj(seq_features)   # [batch_size, 1, patch_num, d_model]
        
        # Reshape Q, K, V for multi-head attention
        Q = Q.view(batch_size, self.n_heads, 1, self.d_model // self.n_heads)  # [batch_size, n_heads, 1, d_model/n_heads]
        K = K.view(batch_size, self.n_heads, patch_num, self.d_model // self.n_heads)  # [batch_size, n_heads, patch_num, d_model/n_heads]
        V = V.view(batch_size, self.n_heads, patch_num, self.d_model // self.n_heads)  # [batch_size, n_heads, patch_num, d_model/n_heads]
        
        # Compute Attention Scores (QK^T) / sqrt(d_model)
        attn_scores = torch.einsum('bhqd,bhkd->bhqk', Q, K)  # [batch_size, n_heads, 1, patch_num]
        attn_scores = attn_scores / self.scale
        
        # Apply Softmax to get Attention Weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch_size, n_heads, 1, patch_num]
        
        # Compute the weighted sum of the values
        attn_output = torch.einsum('bhqk,bhkd->bhqd', attn_weights, V)  # [batch_size, n_heads, 1, d_model/n_heads]
        attn_output = attn_output.view(batch_size, 1, self.d_model)  # [batch_size, 1, d_model]
        
        return attn_output, attn_weights



# class CrossAttention(nn.Module):
#     def __init__(self, embed_dim, output_attention=False, num_heads=8):
#         super(CrossAttention, self).__init__()
#         self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
#         self.output_attention = output_attention
        
#     def forward(self, query, key, value):
#         # query: [1 x batch_size x d_model], txt_emb: [1 x batch_size x d_model]
#         # MultiheadAttention expects the input in (ndim, batch_size, embed_dim) format
#         attn_output, attn_output_weights = self.multihead_attn(query, key, value)
#         # 
#         if self.output_attention:
#             return attn_output, attn_output_weights
#         else:
#             return attn_output, None
        

class CrossAttention_Mask(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(CrossAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        B, L, E = queries.shape
        _, S, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        
        # Computing the attention scores
        scores = torch.einsum("ble,bse->bls", queries, keys)
        
        # Applying the mask if needed
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        
        # Applying softmax to get the attention weights
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        
        # Computing the weighted sum of values
        V = torch.einsum("bls,bsd->bld", A, values)
        
        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None

class TriangularCausalMask():
    def __init__(self, B, L, device):
        mask_shape = [B, 1, L, L]
        self.mask = torch.triu(torch.ones(mask_shape, device=device), diagonal=1).bool()
    

class DSAttention(nn.Module):
    '''De-stationary Attention'''
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(DSAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        tau = 1.0 if tau is None else tau.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x 1
        delta = 0.0 if delta is None else delta.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x S

        # De-stationary Attention, rescaling pre-softmax score with learned de-stationary factors
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(
            L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H,
                                                L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) /
                     L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[
                                                  None, :, None], index, :] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * \
                 np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * \
            np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(
            queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        # print(f'B={B}, L={L}, S={S}, H={H}')
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

class NewAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        # print(f'B={B}, L={L}, S={S}, H={H}')
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
    
class ReformerLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, causal=False, bucket_size=4, n_hashes=4):
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal
        )

    def fit_length(self, queries):
        # inside reformer: assert N % (bucket_size * 2) == 0
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            # fill the time series
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        # in Reformer: defalut queries=keys
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None


class TwoStageAttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''

    def __init__(self, configs,
                 seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                           output_attention=configs.output_attention), d_model, n_heads)
        self.dim_sender = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                       output_attention=configs.output_attention), d_model, n_heads)
        self.dim_receiver = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                         output_attention=configs.output_attention), d_model, n_heads)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc, attn = self.time_attention(
            time_in, time_in, time_in, attn_mask=None, tau=None, delta=None
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)
        
        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
        dim_buffer, attn = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None, tau=None, delta=None)
        dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=None, tau=None, delta=None)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)
        del time_in, time_enc, dim_in, attn, dim_enc, dim_receive, dim_buffer, dim_send, batch_router
        torch.cuda.empty_cache()
        return final_out




class TwoStageAttentionLayer_TD(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''

    def __init__(self, configs,
                 seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer_TD, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                           output_attention=configs.output_attention), d_model, n_heads)
        self.dim_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                       output_attention=configs.output_attention), d_model, n_heads)
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        # exchange dim: [batch_size, Seg_num(L), Data_dim(D), d_model] -> [ (batch_size, Data_dim(D)), Seg_num(L), d_model]
        # time_in = rearrange(x, 'b seg_num ts_d d_model -> (b ts_d) seg_num d_model')
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc, attn = self.time_attention(
            time_in, time_in, time_in, attn_mask=None, tau=None, delta=None
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)
        
        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        dim_in = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        dim_enc, attn = self.dim_attention(
            dim_in, dim_in, dim_in, attn_mask=None, tau=None, delta=None
        )
        
        dim_enc = dim_in + self.dropout(dim_enc)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)
        return final_out, None



class TwoStageAttentionLayer_DT(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''

    def __init__(self, configs,
                 seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer_DT, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                           output_attention=configs.output_attention), d_model, n_heads)
        self.dim_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                       output_attention=configs.output_attention), d_model, n_heads)
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        
        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        dim_in = rearrange(x, 'b ts_d seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        dim_enc, attn = self.dim_attention(
            dim_in, dim_in, dim_in, attn_mask=None, tau=None, delta=None
        )
        
        dim_enc = dim_in + self.dropout(dim_enc)
        dim_enc = self.norm1(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP1(dim_enc))
        dim_enc = self.norm2(dim_enc)
        
        
        # exchange dim: [batch_size, Seg_num(L), Data_dim(D), d_model] -> [ (batch_size, Data_dim(D)), Seg_num(L), d_model]
        # time_in = rearrange(x, 'b seg_num ts_d d_model -> (b ts_d) seg_num d_model')
        time_in = rearrange(dim_in, '(b seg_num) ts_d d_model -> (b ts_d) seg_num d_model', b=batch)
        time_enc, attn = self.time_attention(
            time_in, time_in, time_in, attn_mask=None, tau=None, delta=None
        )
        time_enc = time_in + self.dropout(time_enc)
        time_enc = self.norm3(time_enc)
        time_enc = time_enc + self.dropout(self.MLP2(time_enc))
        time_enc = self.norm4(time_enc)
        
        final_out = rearrange(time_enc, '(b ts_d) seg_num d_model -> b ts_d seg_num d_model', b=batch)
        return final_out, None
    

class TimeAttentionLayer(nn.Module):
    '''
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''

    def __init__(self, configs,
                 seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(TimeAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                           output_attention=configs.output_attention), d_model, n_heads)
        self.dropout = nn.Dropout(dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.MLP = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        # exchange dim: [batch_size, Seg_num(L), Data_dim(D), d_model] -> [ (batch_size, Data_dim(D)), Seg_num(L), d_model]
        # time_in = rearrange(x, 'b seg_num ts_d d_model -> (b ts_d) seg_num d_model')
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc, attn = self.time_attention(
            time_in, time_in, time_in, attn_mask=None, tau=None, delta=None
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP(dim_in))
        dim_in = self.norm2(dim_in)

        final_out = rearrange(dim_in, '(b ts_d) seg_num d_model -> b ts_d seg_num d_model', b=batch)
        return final_out, None
 
    
class DimAttentionLayer(nn.Module):
    '''
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''

    def __init__(self, configs,
                 seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(DimAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.dim_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                       output_attention=configs.output_attention), d_model, n_heads)
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.MLP = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        
        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        dim_in = rearrange(x, 'b ts_d seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        dim_enc, attn = self.dim_attention(
            dim_in, dim_in, dim_in, attn_mask=None, tau=None, delta=None
        )
        
        dim_enc = dim_in + self.dropout(dim_enc)
        dim_enc = self.norm1(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP(dim_enc))
        dim_enc = self.norm2(dim_enc)
        
        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)
        return final_out, None
    
    

# class FullAttention(nn.Module):
#     def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
#         super(FullAttention, self).__init__()
#         self.scale = scale
#         self.mask_flag = mask_flag
#         self.output_attention = output_attention
#         self.dropout = nn.Dropout(attention_dropout)

#     def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
#         B, L, H, E = queries.shape
#         _, S, _, D = values.shape
#         scale = self.scale or 1. / sqrt(E)

#         scores = torch.einsum("blhe,bshe->bhls", queries, keys)

#         if self.mask_flag:
#             if attn_mask is None:
#                 attn_mask = TriangularCausalMask(B, L, device=queries.device)

#             scores.masked_fill_(attn_mask.mask, -np.inf)

#         A = self.dropout(torch.softmax(scale * scores, dim=-1))
#         V = torch.einsum("bhls,bshd->blhd", A, values)

#         if self.output_attention:
#             return V.contiguous(), A
#         else:
#             return V.contiguous(), None




# class AttentionLayer(nn.Module):
#     def __init__(self, attention, d_model, n_heads, d_keys=None,
#                  d_values=None):
#         super(AttentionLayer, self).__init__()

#         d_keys = d_keys or (d_model // n_heads)
#         d_values = d_values or (d_model // n_heads)

#         self.inner_attention = attention
#         self.query_projection = nn.Linear(d_model, d_keys * n_heads)
#         self.key_projection = nn.Linear(d_model, d_keys * n_heads)
#         self.value_projection = nn.Linear(d_model, d_values * n_heads)
#         self.out_projection = nn.Linear(d_values * n_heads, d_model)
#         self.n_heads = n_heads

#     def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
#         B, L, _ = queries.shape
#         _, S, _ = keys.shape
#         H = self.n_heads
#         # print(f'B={B}, L={L}, S={S}, H={H}')
#         queries = self.query_projection(queries).view(B, L, H, -1)
#         keys = self.key_projection(keys).view(B, S, H, -1)
#         values = self.value_projection(values).view(B, S, H, -1)

#         out, attn = self.inner_attention(
#             queries,
#             keys,
#             values,
#             attn_mask,
#             tau=tau,
#             delta=delta
#         )
#         out = out.view(B, L, -1)

#         return self.out_projection(out), attn
