# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
from collections import OrderedDict
from turtle import distance
from typing import Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import einops

from .diffusion_model import extract, cosine_beta_schedule, linear_beta_schedule, quadratic_beta_schedule, sigmoid_beta_schedule, SinusoidalPositionEmbeddings
import ipdb


"""
Transformer functions adapted from https://github.com/openai/CLIP/blob/main/clip/model.py
"""
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, pad_mask=None):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask, key_padding_mask=pad_mask)[0]

    def forward(self, x: torch.Tensor, pad_mask=None):
        x = x + self.attention(self.ln_1(x), pad_mask=pad_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class TemporalModelling(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, dropout: float, attn_mask: torch.Tensor = None, ):
        super(TemporalModelling, self).__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, dropout, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor, pad_mask=None):
        # return self.resblocks(x)
        for i in range(len(self.resblocks)):
            x = self.resblocks[i](x, pad_mask=pad_mask)
        return x


class DiffusionTransformer(torch.nn.Module):
    def __init__(self, num_seg=8, tfm_layers=4, tfm_heads=8, hidden_size=512, dropout=0.0, cfg=None):
        super(DiffusionTransformer, self).__init__()
        self.cfg = cfg
        self.dropout = 0.0
        self.hidden_size = hidden_size
        self.num_seg = num_seg
        self.tfm_layers = tfm_layers
        self.tfm_heads = tfm_heads
        self.max_len = self.cfg.DEV.ORDER_PRETRAIN_MAX_LEN # self.num_seg + 1
        attn_mask = None # self.build_attention_mask() # bi-directional attention or uni-directional attention
        
        # create modules
        self.pad_embedding = torch.nn.Embedding(1, self.hidden_size) # pad token embedding
        self.type_embedding = torch.nn.Embedding(2, self.hidden_size) # type embedding to differentiate mask and the other tokens
        self.temporalEmbedding = torch.nn.Embedding(self.max_len, self.hidden_size) # position embedding, including mask token
        self.temporalModelling = TemporalModelling(width=self.hidden_size, layers=self.tfm_layers, heads=self.tfm_heads, \
                                                   dropout=self.dropout, attn_mask=attn_mask)
        # time level embedding of diffusion process
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.hidden_size // 4),
            nn.Linear(self.hidden_size // 4, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        
        # initialization
        self.initialize_parameters()

        # try larger time levels
        self.total_levels = self.tfm_layers # 4
        self.level_batch = self.tfm_layers # 4

        # pre-compute configuration of diffusion process
        self.configure_diffusion()

    def configure_diffusion(self,):
        """ Define coefficients of diffusion schedule
        """ 
        timesteps = self.total_levels

        # define beta schedule
        self.betas = linear_beta_schedule(timesteps=timesteps)

        # define alphas 
        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        self.schedule_coef = [self.betas, self.sqrt_recip_alphas, self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod, self.posterior_variance]

    def forward(self, x, is_pretrain=False):
        """
        Input clip-level features (b*t, c), temporal modeling over clips (t), return the embeddings of masked positions (b, c)
        """
        device = x.device

        # clip-level temporal modelling
        if self.training:
            if is_pretrain: # randomly mask clips (past, middle and future)
                # prepare input dimension
                clip_feats = einops.rearrange(x, '(b t) c -> t b c', t=self.max_len)
                tempEmbedding = einops.repeat(self.temporalEmbedding(torch.arange(self.max_len).to(device)), 't c -> t b c', b=clip_feats.size(1))

                # get the indices of random mask position
                batch_size = clip_feats.size(1)
                bs_inds = torch.arange(batch_size, device=device)
                mask_inds = torch.randint(0, self.max_len, (batch_size,), device=x.device)

                # recored the clip embs that will be masked out
                clip_feats_x0 = clip_feats[mask_inds, bs_inds]

                # replace clips with padding embeding
                clip_feats, pad_mask = self.pad_sequence(clip_feats, mask_inds) # randomly replace the video embeddings after mask token with padding embedding

                # denoising process            
                clip_feats, all_noise, all_pred_noise, intermediate_denoise = self.diffusion_signal_training(clip_feats, clip_feats_x0, mask_inds, bs_inds, tempEmbedding, pad_mask, batch_size, device)
                
                return clip_feats, mask_inds, [all_noise, all_pred_noise], intermediate_denoise
            else: # finetuning
                clip_feats = self.diffusion_signal_forecast(x, device)
                return clip_feats

        if not self.training: # inference
            clip_feats = self.diffusion_signal_forecast(x, device) # step forecasting
            return clip_feats  

    def diffusion_signal_training(self, clip_feats, clip_feats_x0, mask_inds, bs_inds, tempEmbedding, pad_mask, batch_size, device):
        """ Denoising process: the inputs of next denoising level are computed using x_0 (property 1)
        """
        orig_feats = clip_feats
        intermediate_denoise = [] # collect the denoised tokens in intermediate time levels
        sample_levels = np.arange(self.tfm_layers)

        for time_i in sample_levels:
            # clone it everytime since we have in-place operation
            clip_feats = orig_feats.clone()

            # parallel time levels, the inputs of next level are computed using x_0 (property 1)
            # compute noisy tokens at t_index level using x_0
            t_index = self.total_levels - 1 - time_i # reverse index
            t = torch.full((batch_size,), t_index, device=device, dtype=torch.long)
            noise = torch.randn_like(clip_feats_x0)
            if time_i == 0: # the most noisy level
                noisy_clip_feats = ennoise(clip_feats_x0.clone().detach(), noise, t, schedule_coef=self.schedule_coef)
            else: # other levels
                noisy_clip_feats = ennoise(denoised_clips.clone().detach(), noise, t, schedule_coef=self.schedule_coef)
            clip_feats[mask_inds, bs_inds] = noisy_clip_feats

            # forward for a time level
            typeEmbedding = self.type_embedding(torch.full((self.max_len, batch_size), 0, device=device, dtype=torch.long)) # type embedding of normal tokens
            typeEmbedding[mask_inds, bs_inds] = self.type_embedding(torch.full((batch_size,), 1, device=device, dtype=torch.long)) # type embedding of mask tokens
            clip_type_feats = clip_feats + typeEmbedding
            clip_pos_feats = clip_type_feats + tempEmbedding # add position embs to both noisy tokens and context tokens
            clip_pos_feats = clip_pos_feats + einops.repeat(self.time_mlp(t), 't c -> b t c', b=clip_pos_feats.size(0)) # add diffusion time embs
            block_feats = self.temporalModelling(clip_pos_feats, pad_mask=pad_mask)

            # record predicted signals
            denoised_clips = block_feats[mask_inds, bs_inds]
            intermediate_denoise.append(denoised_clips)
        
        # the final denoised tokens (x_0)
        clip_feats = denoised_clips
        clip_feats_x0 = clip_feats_x0.unsqueeze(0).expand(self.total_levels, -1, -1).reshape(-1, clip_feats_x0.size(-1)) # MSE target at each level
        intermediate_denoise = torch.cat(intermediate_denoise) # [batch * #time_levels, d]

        return clip_feats, clip_feats_x0, intermediate_denoise, intermediate_denoise

    def diffusion_signal_forecast(self, x, device):
        """ Assume that the future single step is the one to be generated/denoised and to be classified.
        """
        # prepare input dimension
        clip_feats = einops.rearrange(x, '(b t) c -> t b c', t=self.num_seg)
        tempEmbedding = einops.repeat(self.temporalEmbedding(torch.arange(self.max_len).to(device)), 't c -> t b c', b=clip_feats.size(1))
        
        # prepare indices of tokens to be generated/denoised
        batch_size = clip_feats.size(1)
        bs_inds = torch.arange(batch_size, device=device)
        mask_inds = torch.full((batch_size,), self.max_len-1, device=device, dtype=torch.long)
        noise = torch.zeros((1, batch_size, clip_feats.size(-1)), dtype=clip_feats.dtype, device=device) # torch.randn((1, batch_size, clip_feats.size(-1)), dtype=clip_feats.dtype, device=device)
        clip_feats = torch.cat((clip_feats, noise), dim=0) # append noise token to the end of sequence

        orig_feats = clip_feats
        clip_feats = orig_feats.clone()
        for time_i in range(self.tfm_layers):
            # compute time level
            t_index = self.total_levels - 1 - time_i # reverse index
            t = torch.full((batch_size,), t_index, device=device, dtype=torch.long)
            # ennoise case: similar performance to the case w/o ennoise
            if time_i != 0:
                noisy_clip_feats = ennoise(denoised_clips.clone().detach(), noise, t, schedule_coef=self.schedule_coef)
                clip_feats[mask_inds, bs_inds] = noisy_clip_feats

            # forward for a time level
            typeEmbedding = self.type_embedding(torch.full((self.max_len, batch_size), 0, device=device, dtype=torch.long)) # type embedding of normal tokens
            typeEmbedding[mask_inds, bs_inds] = self.type_embedding(torch.full((batch_size,), 1, device=device, dtype=torch.long)) # type embedding of mask tokens
            clip_type_feats = clip_feats + typeEmbedding
            clip_pos_feats = clip_type_feats + tempEmbedding # add position embs to both noisy tokens and context tokens
            clip_pos_feats = clip_pos_feats + einops.repeat(self.time_mlp(t), 't c -> b t c', b=clip_pos_feats.size(0)) # add diffusion time embs
            block_feats = self.temporalModelling(clip_pos_feats)

            # record predicted signals
            denoised_clips = block_feats[mask_inds, bs_inds]

            # replace noisy tokens with denoised tokens
            clip_feats = orig_feats.clone()
            clip_feats[mask_inds, bs_inds] = denoised_clips
        
        # the final denoised tokens (x_0)
        clip_feats = clip_feats[mask_inds, bs_inds]

        return clip_feats

    def initialize_parameters(self):
        nn.init.normal_(self.pad_embedding.weight, std=0.01)  # pad token embedding
        # nn.init.normal_(self.mask_embedding.weight, std=0.01)
        nn.init.normal_(self.temporalEmbedding.weight, std=0.01)

        proj_std = (self.temporalModelling.width ** -0.5) * ((2 * self.temporalModelling.layers) ** -0.5)
        attn_std = self.temporalModelling.width ** -0.5
        fc_std = (2 * self.temporalModelling.width) ** -0.5
        for block in self.temporalModelling.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def build_attention_mask(self):
        # pytorch uses additive attention mask when it's float dtype; fill with -inf (the mask values will be added to the attention weight)
        mask = torch.empty(self.max_len, self.max_len)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
 
    def pad_sequence(self, clip_feats, mask_inds):
        """ Given a batch of sequences and mask token position, randomly replace the samples after mask token with the padding embedding
        """
        bs = clip_feats.shape[1]  # shape: [t, b, c]
        device = clip_feats.device
        all_pad_mask = []

        for i in range(bs):
            if mask_inds[i].item() + 1 == self.max_len: # mask token at last position
                pad_start = self.max_len
            else: # mask token not at last position
                pad_start = torch.randint(mask_inds[i].item() + 1, self.max_len, (1,)).item()
                clip_feats[pad_start:, i] = self.pad_embedding.weight
            pad_mask = torch.BoolTensor([True if pad_i >= pad_start else False for pad_i in range(self.max_len)]).unsqueeze(0).to(device)
            all_pad_mask.append(pad_mask)
        all_pad_mask = torch.cat(all_pad_mask, dim=0)

        return clip_feats, all_pad_mask

def ennoise(x_start, noise, t, schedule_coef):
    """ Given x_0 and time level, compute the noisy x_t using property 1 (q_sample)
    """
    # precomputed schedule coefficients
    betas, sqrt_recip_alphas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, posterior_variance = schedule_coef

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def denoise(x, pred_noise, t, t_index, schedule_coef):
    """ Given predicted noise and x_t, compute the denoised features x_t-1 (p_sample)
    """
    # return pred_noise
    # precomputed schedule coefficients
    betas, sqrt_recip_alphas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, posterior_variance = schedule_coef

    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    
    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise 


