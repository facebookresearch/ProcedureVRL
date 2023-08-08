# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# Copyright 2020 Ross Wightman
# Modified Model definition

import torch
import torch.nn as nn
from functools import partial
import math
import warnings
import torch.nn.functional as F
import numpy as np

from lib.models.slowfast_mvit.mvit import MViT_encoder
from lib.models.vit_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from lib.models.helpers import load_pretrained
from lib.models.vit_utils import DropPath, to_2tuple, trunc_normal_
from lib.models.tfm_model import DiffusionTransformer as OrderTransformer

from .build import MODEL_REGISTRY
from torch import einsum
from einops import rearrange, reduce, repeat
import clip
import ipdb

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'mvit': _cfg(
        url='https://dl.fbaipublicfiles.com/mvit/mvitv2_models/MViTv2_S_in1k.pyth',
    ),
}

class VisionTransformer(nn.Module):
    """ Vision Transformer
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, num_frames=8, attention_type='divided_space_time',
                 label_emb = '', mlp=0,text_model = '', lp=False,num_seg=0,extra_tr='order', drope=0., cfg=None):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.temp = self.cfg.DEV.TEMP
        self.order_pretrain = cfg.DEV.ORDER_PRETRAIN_ENABLED
        self.order_max_len = cfg.DEV.ORDER_PRETRAIN_MAX_LEN
        self.order_fix_recognition = cfg.DEV.ORDER_FIX_RECOGNITION
        self.order_tfm_layers = cfg.DEV.ORDER_TFM_LAYERS
        self.order_recog_batch = cfg.DEV.ORDER_RECOG_BATCH
        self.softmax = nn.Softmax(dim=1)
        
        ############## Frame-level TimeSformer Encoder ##############
        assert self.cfg.MODEL.MODEL_NAME == 'MViT'
        self.depth = self.cfg.MVIT.DEPTH
        self.video_encoder = MViT_encoder(self.cfg)
        embed_dim = self.video_encoder.norm.weight.shape[0]
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        ############## Frame-level TimeSformer Encoder ##############

        # Classifier head
        self.mlp = mlp
        self.label = label_emb # text embeddings
        if label_emb != '':  # pretraining
            self.label_emb = torch.load(label_emb)
            # match video emb to language emb
            self.head = nn.Linear(embed_dim, self.label_emb.shape[1])  # project to the dim of emb   
            # order pretraining
            self.order_tfm = OrderTransformer(num_seg=self.order_max_len-1, tfm_layers=self.order_tfm_layers, dropout=self.cfg.MODEL.DROP_E, hidden_size=self.head.weight.shape[0], cfg=self.cfg)
        else:  # finetuning
            if self.cfg.DEV.MATCH_LANG_EMB: # match video emb to language emb
                self.label_emb = torch.load(self.cfg.DEV.TEST_LANG_EMB)
                self.head = nn.Linear(embed_dim, self.label_emb.shape[1])  # project to the dim of emb
                for p in self.head.parameters(): p.requires_grad = False # fix head
            else: # classify video emb into classes
                self.label_emb = False
                self.test_lang_emb = torch.load(self.cfg.DEV.TEST_LANG_EMB)
                self.head = nn.Linear(embed_dim, self.test_lang_emb.shape[1]) 
                for p in self.head.parameters(): p.requires_grad = False
                
                if cfg.TRAIN.DATASET == 'Epickitchens':
                    self.head_n = nn.Linear(self.test_lang_emb.shape[1], 300)
                    self.head_v = nn.Linear(self.test_lang_emb.shape[1], 97)
                else:
                    self.head_cls = nn.Linear(self.test_lang_emb.shape[1], num_classes)  
            self.apply(self._init_weights)
        
        # text encoder
        self.text = text_model  
        if text_model == 'clip_vit_b_16': # create text encoder
            clip_model, _ = clip.load("ViT-B/16", jit=False)
            del clip_model.visual # remove visual branch
            self.text_model = clip_model.float()
            for p in self.text_model.parameters(): p.requires_grad = False

        # clip-level temporal modeling
        if num_seg > 0:
            self.num_seg = num_seg
            self.order_tfm = OrderTransformer(num_seg=self.num_seg, tfm_layers=self.order_tfm_layers, dropout=self.cfg.MODEL.DROP_E, hidden_size=self.head.weight.shape[0], cfg=self.cfg)

    def forward(self, x):
        # seperate input video frames and input text (e.g., ASR)
        if len(self.text) > 0 and self.training:
            x, text = x
        batch_size = x.shape[0]

        # divide m*t video frames into m clips
        if self.order_pretrain: # order pretraining
            x = rearrange(x, 'b m c t h w -> (b m) c t h w', m=self.order_max_len)
        elif hasattr(self, 'num_seg') and self.num_seg > 0: # step forecasting
            x = rearrange(x, 'b c (m t) h w -> (b m) c t h w', m=self.num_seg, t=x.shape[2]//self.num_seg)
        
        # TimeSformer encoder
        x = self.video_encoder(x) # input: [b, 3, t, h, w] --> output: [b, d]
        
        # projection layers
        if self.cfg.DEV.MATCH_LANG_EMB: # match video emb to language emb (pretraining)
            self.label_emb = self.check_device_norm(self.label_emb, x.device, norm=True)
            x = self.head(x)
            x = x / x.norm(dim=1, keepdim=True)
            video_emb = x
            if hasattr(self, 'num_seg') and self.num_seg > 0: # zero-shot step forecasting
                x = self.order_tfm(video_emb)
                x = x / x.norm(dim=1, keepdim=True)
            x = x @ self.label_emb.t() / self.temp
        else: # classify video emb into classes (finetuning)
            if hasattr(self, 'num_seg') and self.num_seg > 0: # step / action forecasting
                x = self.head(x)
                video_emb = x / x.norm(dim=1, keepdim=True)
                x = self.order_tfm(video_emb)
                x = self.head_cls(x)
            else: # step / action classification
                x = self.head(x)
                x = x / x.norm(dim=1, keepdim=True)
                if hasattr(self, 'head_n'): # EPIC-Kitchen step classification
                    v = self.head_v(x) / self.temp
                    n = self.head_n(x) / self.temp
                    return (v,n)
                else:
                    x = self.head_cls(x) / self.temp
                
        # create pseudo labels using language embeddings during pre-training
        if isinstance(self.label_emb,torch.Tensor) and len(self.text) > 0 and self.training: # order pretraining
            # use teacher model to match video frames&ASR and step descriptions
            teacher_x = self.get_pseudo_labels(x.device, text)

            # recover video embedding of mask token
            pred_video_emb, mask_inds, mse_loss, intermediate_denoise = self.order_tfm(video_emb, is_pretrain=True)

            # get the matching score of mask token
            pred_video_emb = pred_video_emb / pred_video_emb.norm(dim=1, keepdim=True)
            mask_pred = pred_video_emb @ self.label_emb.t() / self.temp 
            
            # get the matching scores of masked-out clip
            masked_teacher_x = self.get_mask_samples(teacher_x, mask_inds=mask_inds) # teacher prediction

            # create teacher target for intermediate denoised tokens
            intermediate_denoise = intermediate_denoise / intermediate_denoise.norm(dim=1, keepdim=True)
            intermediate_pred = intermediate_denoise @ self.label_emb.t() / self.temp 
            intermediate_teacher_x = masked_teacher_x.unsqueeze(0).expand(self.order_tfm.level_batch, -1, -1).reshape(-1, masked_teacher_x.size(-1))

            # organize the batch
            rand_inds = torch.randperm(x.shape[0], device=x.device)[:batch_size*self.order_recog_batch] # get a subset of batch to avoid OOM
            x = x[rand_inds]
            teacher_x = teacher_x[rand_inds]

            # sampled time levels from diffusion model
            x = torch.cat((x, intermediate_pred), dim=0) 
            teacher_x = torch.cat((teacher_x, intermediate_teacher_x), dim=0) 
            return x, teacher_x, mse_loss
                
        # testing
        if not self.training:
            x = self.softmax(x)
        
        return x

    def get_mask_samples(self, all_samples, mask_inds):
        all_samples = rearrange(all_samples, '(b m) c -> b m c', m=self.order_max_len, b=all_samples.shape[0]//self.order_max_len)
        mask_samples = all_samples[torch.arange(all_samples.shape[0], device=all_samples.device), mask_inds, :]
        return mask_samples

    def get_pseudo_labels(self, device, text):
        # encode ASR with langauge encoder
        vis_emb = text['clip_vis_feat']
        text_emb = self.text_model.encode_text(text['clip_text_ids'].squeeze(dim=1)) # torch.FloatTensor(x.shape[0], 512).to(x.device) # vis_emb # 
        text_emb = (text_emb + vis_emb) / 2.0
        self.label_emb = self.check_device_norm(self.label_emb, device, norm=True)
        text_emb = text_emb / text_emb.norm(dim=1, keepdim=True)
        teacher_x = text_emb @ self.label_emb.t() / self.temp
        return teacher_x  

    def check_device_norm(self, query_tensor, target_device, norm=False):
        if query_tensor.device != target_device:
            query_tensor = query_tensor.to(target_device)
            if norm:
                query_tensor = query_tensor / query_tensor.norm(dim=1, keepdim=True)
        return query_tensor

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)            
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

@MODEL_REGISTRY.register()
class MViT(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(MViT, self).__init__()
        self.pretrained = cfg.MODEL.PRETRAINED
        patch_size = 16
        mlp = cfg.MODEL.MLP
        emb = cfg.TRAIN.LABEL_EMB
        lp = cfg.MODEL.TEXT_LP
        num_seg = cfg.MODEL.NUM_SEG
        extra = cfg.MODEL.EXTRA_TR
        drope = cfg.MODEL.DROP_E
        dpr = cfg.MODEL.DROP_PATH
        depth = cfg.TIMESFORMER.DEPTH
        self.model = VisionTransformer(img_size=cfg.DATA.TRAIN_CROP_SIZE, num_classes=cfg.MODEL.NUM_CLASSES, patch_size=patch_size, \
            embed_dim=768, depth=depth, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), \
            drop_rate=0., attn_drop_rate=0., drop_path_rate=dpr, num_frames=cfg.DATA.NUM_FRAMES, \
            attention_type=cfg.TIMESFORMER.ATTENTION_TYPE, label_emb = emb, mlp = mlp, text_model = cfg.MODEL.TEXT_MODEL, \
            lp=lp,num_seg=num_seg,extra_tr=extra,drope=drope,cfg=cfg,**kwargs)

        self.attention_type = cfg.TIMESFORMER.ATTENTION_TYPE
        self.model.default_cfg = default_cfgs['mvit']
        self.num_patches = (cfg.DATA.TRAIN_CROP_SIZE // patch_size) * (cfg.DATA.TRAIN_CROP_SIZE // patch_size)
        pretrained_model = cfg.TIMESFORMER.PRETRAINED_MODEL
        if self.pretrained:
            load_pretrained(self.model, num_classes=self.model.num_classes, in_chans=kwargs.get('in_chans', 3), \
                filter_fn=None, img_size=cfg.DATA.TRAIN_CROP_SIZE, num_patches=self.num_patches, attention_type="", \
                pretrained_model=pretrained_model, num_frames=cfg.DATA.NUM_FRAMES, pre_num=cfg.MODEL.PRE_CLASSES)
        else:
            print('not loading any pretrained weights!')

    def forward(self, x):
        x = self.model(x)
        return x

    
