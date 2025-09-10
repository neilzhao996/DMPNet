import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch.nn as nn
import torch
from RGBD.toolbox.models.segformermodels.backbones.mix_transformer_ourprompt_proj import MITB5
from RGBD.toolbox.models.segformermodels.backbones.mix_transformer_ourprompt_proj import OverlapPatchEmbed
from functools import partial
from RGBD.toolbox.models.segformermodels.decode_heads.segformer_head import SegFormerHead
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif hasattr(m, 'initialize'):
            m.initialize()

class DirectionalGatedConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_h = nn.Conv2d(dim, dim, kernel_size=(1, 3), padding=(0, 1), groups=dim)
        self.conv_v = nn.Conv2d(dim, dim, kernel_size=(3, 1), padding=(1, 0), groups=dim)
        self.conv_d = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.conv_h(x)
        v = self.conv_v(x)
        d = self.conv_d(x)
        g = self.gate(x)
        return g * (h + v + d)

class DirectionalInteractionAttention(nn.Module):
    def __init__(self, dim, num_heads=8, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.q_conv = DirectionalGatedConv(dim)
        self.k_conv = DirectionalGatedConv(dim)
        self.v_conv = DirectionalGatedConv(dim)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        q = self.q_conv(self.q_proj(x))
        k = self.k_conv(self.k_proj(x))
        v = self.v_conv(self.v_proj(x))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

    def initialize(self):
        weight_init(self)




class ADA(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv0_0 = nn.Linear(dim, dim // 4)
        self.conv0_1 = nn.Linear(dim, dim // 4)
        self.conv = nn.Linear(dim // 4, dim)

    def forward(self, p, x):
        p = self.conv0_0(p)
        x = self.conv0_1(x)
        p1 = p + x
        p1 = self.conv(p1)
        return p1

class Fovea(nn.Module):

    def __init__(self, smooth=False):
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)

        self.smooth = smooth
        if smooth:
            self.smooth = nn.Parameter(torch.zeros(1) + 10.0)

    def forward(self, x):
        '''
            x: [batch_size, features, k]
        '''

        if x.dim() == 3:
            # 处理三维输入
            b, c, k = x.shape
            if self.smooth:
                mask = self.softmax(x * self.smooth)
            else:
                mask = self.softmax(x)
            output = mask * x
            return output

        elif x.dim() == 4:
            # 处理四维输入
            b, c, h, w = x.shape
            x = x.contiguous().view(b, c, h*w)

            if self.smooth:
                mask = self.softmax(x * self.smooth)
            else:
                mask = self.softmax(x)
            output = mask * x
            output = output.contiguous().view(b, c, h, w)

            return output


class DP(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.conv_bright = nn.Linear(dim, dim // 4)
        self.conv_dark = nn.Linear(dim, dim // 4)
        self.conv = nn.Linear(dim // 4, dim)
        self.fovea = Fovea()


        self.dia = DirectionalInteractionAttention(dim=dim // 4, num_heads=num_heads,bias= True)

        self.dia.initialize()
    def forward(self, depth, x1,bright,dark,H, W):

        bright_rgb = self.conv_bright(x1)
        dark_depth = self.conv_dark(depth)

        B, N, C = bright_rgb.shape

        bright_rgb_4d = bright_rgb.permute(0, 2, 1).view(B, C, H, W)  # [B, dim//4, H, W]


        enhanced_rgb = self.dia(bright_rgb_4d) + bright_rgb_4d # 方向感知增强RGB特征


        enhanced_rgb_3d = enhanced_rgb.view(B, C, N).permute(0, 2, 1)  # [B, N, dim//4]


        p1 = self.fovea(enhanced_rgb_3d + 0.05* bright * enhanced_rgb_3d) + \
             dark_depth * (0.95+ 0.05 * dark)


        p1 = self.conv(p1)

        return p1


class MEP(nn.Module):
    def __init__(self):
        super().__init__()
        self.ldm = LightDarkModule()

    def forward(self, rgb, depth):

        combined_input = torch.cat([rgb, depth], dim=1)
        ldm_input = F.interpolate(combined_input, size=[64, 64], mode='bilinear', align_corners=False)
        ldm_input = self.normImg(ldm_input)
        ldm_output = self.ldm(ldm_input)

        bright = ldm_output[:, 0].view(ldm_output.shape[0], 1, 1)
        dark = ldm_output[:, 1].view(ldm_output.shape[0], 1, 1)

        return bright, dark

    def normImg(self, img):
        # 找到张量的最小和最大值
        min_val = torch.min(img)
        max_val = torch.max(img)
        # 缩放张量的值到 [0, 1] 的范围
        img = (img - min_val) / (max_val - min_val)
        return img


class LightDarkModule(nn.Module):
    def __init__(self):
        super(LightDarkModule, self).__init__()

        self.conv1 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 64, 128)
        self.fc2 = nn.Linear(128, 2)  # 输出层神经元数量改为2
        # self.sigmoid = nn.Sigmoid()  # 添加 Sigmoid 激活函数

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 64)#    确定张量是连续的
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = self.sigmoid(x)  # 使用 Sigmoid 激活函数
        x = x / x.sum(dim=1, keepdim=True)
        return x






class DMPNet(nn.Module):
    def __init__(self, in_chans=3, img_size=224, patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1):
        super(DMPNet, self).__init__()

        self.rgb = MITB5(pretrained=True)
        self.head = SegFormerHead(4)

        self.learnable_prompt = nn.Parameter(torch.randn(1, 30, 32))

        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        self.dp1 = DP(dim=64)
        self.ada1 = ADA(dim=64)
        self.ada2 = ADA(dim=128)
        self.ada3 = ADA(dim=320)
        self.ada4 = ADA(dim=512)
        self.mep = MEP()
        self.conv1 = nn.Conv2d(150, 41, kernel_size=1)

    def forward(self, rgb, depth):
        B = rgb.shape[0]
        outs = []
        #   可学习的提示令牌（形状为[1, 30, 32]），通过expand操作适配批量大小，用于引导多模态特征融合。
        learnable_prompt = self.learnable_prompt.expand(rgb.shape[0], -1, -1)
        x1, H, W = self.rgb.patch_embed1(rgb)
        d1, Hd, Wd = self.patch_embed1(depth)
        bright,dark = self.mep(rgb,depth)
        prompted = self.dp1(d1, x1,bright,dark,H, W)
        x1 = x1 + prompted
        for i, blk in enumerate(self.rgb.block1):
            x1 = blk(x1, learnable_prompt, H, W)
        x1 = self.rgb.norm1(x1)
        x1 = x1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        prompted = prompted.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)


        x2, H, W = self.rgb.patch_embed2(x1)
        d2, Hd, Wd = self.patch_embed2(prompted)
        prompted = self.ada2(d2, x2)
        x2 = x2 + prompted
        for i, blk in enumerate(self.rgb.block2):
            x2 = blk(x2, learnable_prompt, H, W)
        x2 = self.rgb.norm2(x2)
        x2 = x2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        prompted = prompted.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x2)

        x3, H, W = self.rgb.patch_embed3(x2)
        d3, Hd, Wd = self.patch_embed3(prompted)
        prompted = self.ada3(d3, x3)
        x3 = x3 + prompted
        for i, blk in enumerate(self.rgb.block3):
            x3 = blk(x3, learnable_prompt, H, W)
        x3 = self.rgb.norm3(x3)
        x3 = x3.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        prompted = prompted.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x3)

        x4, H, W = self.rgb.patch_embed4(x3)
        d4, Hd, Wd = self.patch_embed4(prompted)
        prompted = self.ada4(d4, x4)
        x4 = x4 + prompted
        for i, blk in enumerate(self.rgb.block4):
            x4 = blk(x4, learnable_prompt, H, W)
        x4 = self.rgb.norm4(x4)
        x4 = x4.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x4)

        # x = self.rgb.head(outs)
        x = self.head(outs)
        x = torch.nn.functional.interpolate(x, scale_factor=4, mode='bilinear')
        x = self.conv1(x)

        return x

if __name__ == '__main__':
    from thop import profile
    x = torch.randn(1, 3, 480, 640)
    ir = torch.randn(1, 3, 480, 640)
    edge = torch.randn(1, 480, 640)
    net = DMPNet()

    # print Flops and learnable parameters
    # out = net(x, ir)
    # for i in out:
    #     print(i.shape)
    # flops, params = profile(net, (x, ir))
    # print('Flops: ', flops / 1e9, 'G')
    # print('Params: ', params / 1e6, 'M')

    # here is the number of learnable parameters.
    s = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(s)
    # print(p)

    # print which parameter are learnable
    # for k, v in net.state_dict().items():
    #     print(k, v.shape)

    x = net(x, ir)
    print(x.shape)
