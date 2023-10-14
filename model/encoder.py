import os
import urllib.request
import timm
import torch
from torch import nn
import model.backbones.resnets as resnets
from model.backbones.swin import SwinEncoder
from model.backbones.xception import AlignedXception
from model.backbones.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d



def build_encoder(config):
    if config.encoder_name == 'swin':
        if config.norm_layer == 'layer':
            norm_layer = nn.LayerNorm
            
        return SwinEncoder(
            img_size=config.img_size, # 224 * 224
            patch_size=config.patch_size, # 4 * 4 * 3
            in_chans=config.in_chans, # 3
            high_level_idx=config.high_level_idx, #2
            low_level_idx=config.low_level_idx,  # 0
            high_level_after_block=config.high_level_after_block, # True
            low_level_after_block=config.low_level_after_block,  # True
            embed_dim=config.embed_dim,  # 96 
            depths=config.depths,   # [2, 2, 6]
            num_heads=config.num_heads,  # [3, 6, 12]
            window_size=config.window_size,  # 7 
            mlp_ratio=config.mlp_ratio, # 4
            qkv_bias=config.qkv_bias, # True
            qk_scale=config.qk_scale,  #None
            drop_rate=config.drop_rate,  # 0
            attn_drop_rate=config.attn_drop_rate, # 0
            drop_path_rate=config.drop_path_rate, # 0.1
            norm_layer=norm_layer,
            high_level_norm=config.high_level_norm, # False
            low_level_norm=config.low_level_norm, # True 
            middle_level_norm = config.middle_level_norm,
            ape=config.ape,  # False
            patch_norm=config.patch_norm, # True
            use_checkpoint=config.use_checkpoint #False
        )
        
    if config.encoder_name == 'xception':
        if config.sync_bn:
            bn = SynchronizedBatchNorm2d
        else:
            bn = nn.BatchNorm2d
        return AlignedXception(output_stride=config.output_stride,
                               input_size=config.img_size,
                               BatchNorm=bn, pretrained=config.pretrained,
                               high_level_dim=config.high_level_dim)
    
    if config.encoder_name == 'resnet':
        model = timm.create_model('resnet50_encoder', 
                                  pretrained=False,
                                  high_level=None,
                                  num_classes=0)
        if config.load_pretrained:
            #path = os.path.expanduser("~") + '/.cache/torch/hub/checkpoints/resnet50_a1_0-14fe96d1.pth'
            path="/home/y212202015/SSEG/transdeeplab-main/transdeeplab-main/pretrained_ckpt/resnet50_a1_0-14fe96d1.pth"
            if not os.path.isfile(path):
                print("downloading ResNet50 pretrained weights...")
                urllib.request.urlretrieve('https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1_0-14fe96d1.pth',
                                           path)
                
            weight = torch.load(path)
            msg = model.load_state_dict(weight, strict=False)
            print(msg)
        
        model.layer4 = nn.Identity()
        model.high_level_size = 14
        model.high_level_dim = 384
        model.low_level_dim = 128
        
        return model
        
        
