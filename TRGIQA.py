import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import antialiased_cnns
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torchvision.ops import deform_conv2d


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MH_attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads      # 384/12=32
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape    # [8,145,384]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)      # qkv[8,145,1152]-->[8,145,3,12,32]-->[3,8,12,145,32]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)  [8,12,145,32]

        attn = (q @ k.transpose(-2, -1)) * self.scale    # [8,12,145,145]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)    # [8,145,384]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MH_attention_mix(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads      # 384/12=32
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, q):
        B, N, C = x.shape    # [8,145,384]
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)    #[8,145,12,32]
        kv = self.kv(x)   # [8,577,768]
        kv = kv.reshape(B, N, 2, self.num_heads, C // self.num_heads)      # [8,145,2,12,32]
        kv = kv.permute(2, 0, 3, 1, 4)      # qkv [2,8,12,577,32]
        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)  [8,12,145,32]

        attn = (q @ k.transpose(-2, -1)) * self.scale    # [8,12,145,145]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)    # [8,145,384]
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Trans_block_mix(nn.Module):    # ...

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=partial(nn.LayerNorm, eps=1e-6)):           # dim=384
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MH_attention_mix(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x, q):
        x = x + self.drop_path(self.attn(self.norm1(x), q))    # [8,145,384]
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Trans_block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=partial(nn.LayerNorm, eps=1e-6)):           # dim=384
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MH_attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))    # [8,145,384]
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class conv_block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(conv_block, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_dim)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.blur = antialiased_cnns.BlurPool(out_dim, stride=2)

    def forward(self, x):
        x = self.blur(self.maxpool(self.relu(self.bn(self.conv1(x)))))       # [8,3,384,384]-->[8,64,96,96]
        return x

class eca_attention(nn.Module):
    """Constructs a ECA module.
       Args:
           channel: Number of channels of the input feature map
           k_size: Adaptive selection of kernel size
       """

    def __init__(self, channel, k_size=3):
        super(eca_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class CNN_block(nn.Module):

    def __init__(self, in_dim, out_dim, stride=1):     #
        super(CNN_block, self).__init__()

        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_dim)
        self.relu2 = nn.ReLU(inplace=True)

        self.residual_conv = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride, padding=0, bias=False)
        self.residual_bn = nn.BatchNorm2d(num_features=out_dim)


    def forward(self, x, x_t=None):
        residual = x
        x = self.conv1(x)     # [8,256,48,48]
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x) if x_t is None else self.conv2(x + x_t)
        x = self.bn2(x)

        residual = self.residual_conv(residual)
        residual = self.residual_bn(residual)

        x = x + residual
        x = self.relu2(x)

        return x

class FCUDown(nn.Module):
    """ CNN feature maps -> Transformer patch embeddings
    """

    def __init__(self, in_planes, out_planes, dw_stride, act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(FCUDown, self).__init__()
        self.dw_stride = dw_stride

        self.conv_project = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0)
        self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)

        self.ln = norm_layer(out_planes)
        self.act = act_layer()

    def forward(self, x, x_t):
        x = self.conv_project(x)  # [N, C, H, W]  [8,64,96,96]-->[8,384,96,96]        [8,256,24,24]-->[8,384,24,24]

        x = self.sample_pooling(x)        # [8,384,96,96]-->[8,384,12,12]   为保持和trans维度一致
        x = x.flatten(2).transpose(1, 2)  # Down   [8,384,12,12]-->[8,144,384]
        x = self.ln(x)
        x = self.act(x)

        x = torch.cat([x_t[:, 0][:, None, :], x], dim=1)    # [8,144,384]-->[8,145,384]

        return x

class FCUUp(nn.Module):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, in_planes, out_planes, up_stride, act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), ):
        super(FCUUp, self).__init__()

        self.up_stride = up_stride
        self.conv_project = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0)
        self.bn = norm_layer(out_planes)
        self.act = act_layer()

    def forward(self, x, H, W):
        B, _, C = x.shape    # x[8,577,384]
        x_r = x[:, 1:].transpose(1, 2).reshape(B, C, H, W)   # Up   x_r[8,384,24,24]
        x_r = self.act(self.bn(self.conv_project(x_r)))      #[8,384,24,24]-->[8,64,24,24]

        return F.interpolate(x_r, size=(H * self.up_stride, W * self.up_stride))  # size: output size

class Downsample(nn.Module):      # residual map
    def __init__(self, in_dim, out_dim, dw_stride):
        super(Downsample, self).__init__()
        self.dw_stride = dw_stride
        self.conv_project = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)
        self.sample_pooling = nn.MaxPool2d(kernel_size=dw_stride, stride=dw_stride)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, out_dim))

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = self.conv_project(x)
        x = self.sample_pooling(x)
        x = x.flatten(2).transpose(1, 2)
        res_q = torch.cat([cls_tokens, x], dim=1)
        return res_q

class ConvTransBlock(nn.Module):
    """
    Basic module for ConvTransformer, keep feature maps for CNN block and patch embeddings for transformer encoder block
    """

    def __init__(self, in_planes, out_planes, stride, dw_stride, hazy_stride, embed_dim, num_heads=12, mlp_ratio=4.,          # dw_stride for FCUDown and  FCUUp
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., ):

        super(ConvTransBlock, self).__init__()
        self.cnn_block = CNN_block(in_dim=in_planes, out_dim=out_planes, stride=stride)
        self.fusion_block = CNN_block(in_dim=out_planes, out_dim=out_planes)
        self.squeeze_block = FCUDown(in_planes=out_planes, out_planes=embed_dim, dw_stride=dw_stride)        # in_plane=64
        self.expand_block = FCUUp(in_planes=embed_dim, out_planes=out_planes, up_stride=dw_stride)

        self.trans_block_mix = Trans_block_mix(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate)

        self.dw_stride = dw_stride
        self.embed_dim = embed_dim

        # hazy compensation
        self.hazy_conv = nn.Conv2d(in_channels=64, out_channels=out_planes, kernel_size=1, stride=1)
        self.maxpool = nn.MaxPool2d(kernel_size=hazy_stride, stride=hazy_stride)


    def forward(self, x, x_t, res, com):    # x[8,64,96,96] x_t[8,145,384]  res[8,64,96,96]
        x = self.cnn_block(x)        # [8,64,96,96]-->[8,64,96,96]     3：[8,64,96,96]-->[8,128,48,48]      5：[8,128,48,48]-->[8,256,24,24]      7: [8,256,24,24]-->[8,512,12,12]
        _, _, H, W = x.shape
        com = self.maxpool(com)
        com = self.hazy_conv(com)
        x = x + com
        x_st = self.squeeze_block(x, x_t)    # x_st[8,145,384]
        x_t = self.trans_block_mix(x_st + x_t, res)    # x_t[8,145,384]
        x_t_r = self.expand_block(x_t, H // self.dw_stride, W // self.dw_stride)       # x_t_r[8,64,96,96]    [8,128,48,48]    [8,256,24,24]
        x = self.fusion_block(x, x_t_r)        # [8,64,96,96]    [8,128,48,48]    [8,256,24,24]    [8,512,12,12]

        return x, x_t


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Conformer(nn.Module):

    def __init__(self, patch_size=32, in_chans=3, num_classes=1, base_channel=64,           # num_classes:1000——>1   embed_dim: 768——>384  depth: 12——>9
                 embed_dim=384, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):

        # Transformer
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        assert depth % 4 == 0

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.patch_embed = PatchEmbed(img_size=384, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches  # ...
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))  # ...
        self.pos_drop = nn.Dropout(p=drop_rate)  # ...
        self.mlp = Mlp(in_features=num_classes, hidden_features=1024, out_features=1)  # ...
        self.trans_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # Classifier head
        self.trans_norm = nn.LayerNorm(embed_dim)
        self.trans_cls_head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.conv_cls_head = nn.Linear(int(256 * 2), num_classes)

        # Stem stage: get the feature maps by conv block (copied form resnet.py)
        self.convx = conv_block(in_dim=in_chans, out_dim=64)
        self.convr = conv_block(in_dim=1, out_dim=64)
        self.down_block = Downsample(in_dim=64, out_dim=embed_dim, dw_stride=8)

        # 1 stage
        stage_1_channel = base_channel   # 64
        trans_dw_stride = patch_size // 4      # 4
        self.conv_1 = CNN_block(in_dim=64, out_dim=stage_1_channel)
        self.trans_patch_conv = nn.Conv2d(64, embed_dim, kernel_size=trans_dw_stride, stride=trans_dw_stride, padding=0)

        # 1~2 stage
        init_stage = 1
        fin_stage = depth // 4 + 1
        for i in range(init_stage, fin_stage):
            self.add_module('conv_trans_' + str(i),
                            ConvTransBlock(        # 64-->64
                                stage_1_channel, stage_1_channel, 1, dw_stride=trans_dw_stride, hazy_stride=1, embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=self.trans_dpr[i - 1],
                            )
                            )

        stage_2_channel = int(base_channel * 2)
        # 3~4 stage
        init_stage = fin_stage  # 5
        fin_stage = fin_stage + depth // 4  # 9
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_1_channel if i == init_stage else stage_2_channel
            self.add_module('conv_trans_' + str(i),
                            ConvTransBlock(      # 64-->128
                                in_channel, stage_2_channel, s, dw_stride=trans_dw_stride // 2, hazy_stride=2, embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=self.trans_dpr[i - 1],
                            )
                            )

        stage_3_channel = int(base_channel * 2 * 2)
        init_stage = fin_stage  # 5
        fin_stage = fin_stage + depth // 4  # 9
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_2_channel if i == init_stage else stage_3_channel
            self.add_module('conv_trans_' + str(i),
                            ConvTransBlock(      # 128-->256
                                in_channel, stage_3_channel, s, dw_stride=trans_dw_stride // 4, hazy_stride=4, embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=self.trans_dpr[i - 1],
                            )
                            )

        stage_4_channel = int(base_channel * 2 * 2 * 2)
        # 5~6 stage
        init_stage = fin_stage  # 9
        fin_stage = fin_stage + depth // 4  # 13
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_3_channel if i == init_stage else stage_4_channel
            self.add_module('conv_trans_' + str(i),
                            ConvTransBlock(       # 256-->512
                                in_channel, stage_4_channel, s, dw_stride=trans_dw_stride // 8, hazy_stride=8, embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=self.trans_dpr[i - 1],
                            )
                            )
        self.fin_stage = fin_stage

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)  # ...
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x, res):        # 模型输入residual + dehazed image
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)    # [8,1,384]

        # stem stage [N, 3, 224, 224] -> [N, 64, 56, 56]
        x_base = self.convx(x)     # [8,3,384,384]-->[8,64,96,96]
        res_base = self.convr(res)    # [8,1,384,384]-->[8,64,96,96]
        res_q = self.down_block(res_base)

        # res * dehaze image
        com = res_base * x_base

        # 1 stage
        x = x_base     # only x
        x_t = self.trans_patch_conv(x_base)       # [8,64,96,96]-->[8,384,12,12]
        x_t = x_t.flatten(2).transpose(1, 2)      # [8,384,144]-->[8,144,384]   flatten:[B,C,H,W]-->[B,C,HW]    transpose:[B,C,HW]-->[B,HW,C]
        x_t = torch.cat([cls_tokens, x_t], dim=1)
        x_t = x_t + self.pos_embed  # ...   [8,144,384]-->[8,145,384]
        x_t = self.pos_drop(x_t)  # ...

        # 2 ~ final
        for i in range(1, self.fin_stage):
            x, x_t = eval('self.conv_trans_' + str(i))(x, x_t, res_q, com)

        # conv classification
        x_p = self.pooling(x).flatten(1)
        conv_cls = self.conv_cls_head(x_p)
        # trans classification
        x_t = self.trans_norm(x_t)
        tran_cls = self.trans_cls_head(x_t[:, 0])

        result = conv_cls + tran_cls     # ...
        con = self.mlp(result)

        return con   # ...
