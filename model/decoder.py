import math
import copy
from collections import OrderedDict

import torch
from torch import nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange

from model.backbones.swin import SwinTransformerBlock

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 4*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x) # 通道扩大4倍
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4) # 尺寸扩大2倍
        x = x.view(B,-1,C//4)
        x= self.norm(x)

        return x

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim 
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.view(B,-1,self.output_dim)
        x= self.norm(x)

        return x

class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)]) # 尺度是不变的

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x

class SwinDecoder(nn.Module):
    def __init__(self, low_level_idx, high_level_idx, 
                 input_size, input_dim,input_high_dim, num_classes,
                 depth, last_layer_depth, num_heads, window_size, mlp_ratio, qkv_bias, qk_scale,
                 drop_rate, attn_drop_rate, drop_path_rate, norm_layer, decoder_norm, use_checkpoint):
        super().__init__()
        self.low_level_idx = low_level_idx # 0
        self.high_level_idx = high_level_idx # 2

        self.proj_high = nn.Linear(input_high_dim, input_dim,bias=False) # 通道从384转换成96
        self.proj_middle = nn.Linear(input_high_dim//2,input_dim,bias=False) # 通道数从192转换成96

        self.layers_up = nn.ModuleList()
        for i in range(high_level_idx - low_level_idx):# 0 , 1
            layer_up = BasicLayer_up(dim=int(input_dim),# 输入的通道是96
                                    input_resolution=(input_size*2**i, input_size*2**i),
                                    depth=depth,
                                    num_heads=num_heads,
                                    window_size=window_size,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                    drop_path=drop_path_rate,
                                    norm_layer=norm_layer,
                                    upsample=PatchExpand,
                                    use_checkpoint=use_checkpoint)
            
            self.layers_up.append(layer_up)
        self.upsample = BasicLayer_up(dim=int(input_dim),# 输入的通道是96
                                    input_resolution=(input_size*2, input_size*2),
                                    depth=depth,
                                    num_heads=num_heads,
                                    window_size=window_size,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                    drop_path=drop_path_rate,
                                    norm_layer=norm_layer,
                                    upsample=PatchExpand,
                                    use_checkpoint=use_checkpoint)

        self.last_layers_up = nn.ModuleList()
        for _ in range(low_level_idx+1): # 1
            i+=1
            last_layer_up = BasicLayer_up(dim=int(input_dim)*3, # 96 * 3
                                            input_resolution=(input_size*2**i, input_size*2**i),
                                            depth=last_layer_depth,
                                            num_heads=num_heads,
                                            window_size=window_size,
                                            mlp_ratio=mlp_ratio,
                                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop=drop_rate, attn_drop=attn_drop_rate,
                                            drop_path=0.0,
                                            norm_layer=norm_layer,
                                            upsample=PatchExpand,
                                            use_checkpoint=use_checkpoint)
            self.last_layers_up.append(last_layer_up)
        
        i += 1
        self.final_up = PatchExpand(input_resolution=(input_size*2**i, input_size*2**i),
                                    dim=int(input_dim)*3,
                                    dim_scale=2,
                                    norm_layer=norm_layer)
        
        if decoder_norm: # True
            self.norm_up = norm_layer(int(input_dim)*3)
        else:
            self.norm_up = None
        self.output = nn.Conv2d(int(input_dim)*3, num_classes, kernel_size=1, bias=False)

    def forward(self, low_level,middle_level,high_level, aspp):
        """
        low_level: B, Hl, Wl, C
        aspp: B, Ha, Wa, C
        """
        high_trans = self.proj_high(high_level)
        target = high_trans + aspp
        middle = self.proj_middle(middle_level)
        B,HM,WM,MC = middle.shape
        middle = middle.view(B,HM*WM,MC)

        B, Hl, Wl, C = low_level.shape
        _, Ha, Wa, _ = aspp.shape
        # _,hm,wm,mc = middle_level.shape
        # _,hh,hw,hc = high_level.shape
        _,ht,wt,ct = target.shape
        low_level = low_level.view(B, Hl*Wl, C) # 56 * 56 * 96
        # middle = middle_level.view(B,hm*wm,mc) # 28 * 28 * 192
        # high = high_level.view(B,hh*hw,hc)   # 14*14*384
        aspp = aspp.view(B, Ha*Wa, C)    # 14 * 14 * 96
        target = target.view(B,ht*wt,ct)
        index = 0
        up = None
        for layer in self.layers_up:
            target = layer(target) # ASPP的特征先经过上采样 最后得到 56×56×96 的特征图
            if index == 0:
                up = target
                target = target + middle
                index = 1
        up_1 = self.upsample(up)
        up_2 = target 
        up_3 = target + low_level

        x = torch.cat([up_1,up_2,up_3], dim=-1) # 在通道维数上进行拼接 56×56×192

        for layer in self.last_layers_up:
            x = layer(x) # 上采样到 112 × 112 × 192

        if self.norm_up is not None: #True
            x = self.norm_up(x)
            
        x = self.final_up(x) # 放大到 225 × 225 × 192  
    
        B, L, C = x.shape
        H = W = int(math.sqrt(L))
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.output(x) # 得到了概率分布图  225 × 225 × 9
        
        return x

    def load_from(self, pretrained_path):
        pretrained_path = pretrained_path
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin decoder---")

            model_dict = self.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 1 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    current_k_2 = 'last_layers_up.' + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
                    full_dict.update({current_k_2:v})
                    
            found = 0
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        # print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]
                    else:
                        found += 1

            msg = self.load_state_dict(full_dict, strict=False)
            # print(msg)
            
            print(f"Decoder Found Weights: {found}")
        else:
            print("none pretrain")
    
    def load_from_extended(self, pretrained_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pretrained_dict = torch.load(pretrained_path, map_location=device)
        pretrained_dict = pretrained_dict['model']
        model_dict = self.state_dict()
        
        selected_weights = OrderedDict()
        for k, v in model_dict.items():
            # if 'relative_position_index' in k: continue
            if 'blocks' in k:
                name = ".".join(k.split(".")[2:])
                shape = v.shape
                
                for pre_k, pre_v in pretrained_dict.items():
                    if name in pre_k and shape == pre_v.shape:
                        selected_weights[k] = pre_v
                        
        msg = self.load_state_dict(selected_weights, strict=False)
        found = len(model_dict.keys()) - len(msg.missing_keys)
        
        print(f"Decoder Found Weights: {found}")



def build_decoder(input_size, input_high_dim, input_dim, config):
    if config.norm_layer == 'layer':
        norm_layer = nn.LayerNorm
    
    if config.decoder_name == 'swin':
        return SwinDecoder(
            input_dim=input_dim,# 输入的通道数为96
            input_high_dim = input_high_dim, # 384
            input_size=input_size, # 14 × 14
            low_level_idx=config.low_level_idx, # 0
            high_level_idx=config.high_level_idx, # 2
            num_classes=config.num_classes,
            depth=config.depth, # 2
            last_layer_depth=config.last_layer_depth, # 6
            num_heads=config.num_heads, # 3
            window_size=config.window_size, # 7
            mlp_ratio=config.mlp_ratio, # 4
            qk_scale=config.qk_scale,
            qkv_bias=config.qkv_bias,
            drop_path_rate=config.drop_path_rate,
            drop_rate=config.drop_rate,
            attn_drop_rate=config.attn_drop_rate,
            norm_layer=norm_layer,
            decoder_norm=config.decoder_norm, # True
            use_checkpoint=config.use_checkpoint
        )



if __name__ == '__main__':
    from config import DecoderConfig
    
    low_level = torch.randn(2, 96, 96, 96)
    aspp = torch.randn(2, 24, 24, 96)

    decoder = build_decoder(24, 96, DecoderConfig)
    print(sum([p.numel() for p in decoder.parameters()])/10**6)

    features = decoder(low_level, aspp)
    print(features.shape)