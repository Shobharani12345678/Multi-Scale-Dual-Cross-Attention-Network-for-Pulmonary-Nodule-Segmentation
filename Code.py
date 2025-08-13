#!/usr/bin/env python
# coding: utf-8

# In[1]:


# MSDC-Net (PyTorch, brief skeleton)
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Utils ----------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

# ---------- MSSA (bottleneck): multi-scale + shared attention + MLP ----------
class MSSA(nn.Module):
    def __init__(self, ch, heads=4, mlp_ratio=4):
        super().__init__()
        self.ms1 = nn.Conv2d(ch, ch, 1, padding=0, bias=False)
        self.ms3 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.ms5 = nn.Conv2d(ch, ch, 5, padding=2, bias=False)

        self.q = nn.Conv2d(3*ch, ch, 1, bias=False)
        self.k = nn.Conv2d(3*ch, ch, 1, bias=False)
        self.v = nn.Conv2d(3*ch, ch, 1, bias=False)
        self.heads = heads
        self.scale = (ch // heads) ** -0.5
        self.proj = nn.Conv2d(ch, ch, 1, bias=False)

        self.norm1 = nn.LayerNorm(ch)
        self.norm2 = nn.LayerNorm(ch)

        hidden = ch * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Conv2d(ch, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, ch, 1),
        )

    def _attn(self, q, k, v):
        # q,k,v : (B, H, Chead, HW)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)  # (B, H, Chead, HW)
        return out

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        m1, m3, m5 = self.ms1(x), self.ms3(x), self.ms5(x)
        m = torch.cat([m1, m3, m5], dim=1)  # (B, 3C, H, W)

        q = self.q(m); k = self.k(m); v = self.v(m)  # (B,C,H,W)
        # split heads
        c_head = c // self.heads
        def reshape_heads(t):
            t = t.view(b, self.heads, c_head, h*w)
            return t
        qh, kh, vh = map(reshape_heads, (q, k, v))
        out = self._attn(qh, kh, vh)                # (B,H,Chead,HW)
        out = out.view(b, c, h, w)
        out = self.proj(out)

        # residual + norm (apply LN over channel dim)
        y = out + x
        y = y.permute(0,2,3,1)                      # (B,H,W,C) for LayerNorm
        y = self.norm1(y).permute(0,3,1,2)

        # MLP
        y2 = self.mlp(y)
        y = y2 + y
        y = y.permute(0,2,3,1)
        y = self.norm2(y).permute(0,3,1,2)
        return y

# ---------- CFDA (skip fusion): spatial + channel attention, cross-applied ----------
class SpatialAttn(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
    def forward(self, x_g, x_l):
        # x_g, x_l: (B,C,H,W)
        avg_g = torch.mean(x_g, dim=1, keepdim=True)
        max_l, _ = torch.max(x_l, dim=1, keepdim=True)
        s = torch.cat([avg_g, max_l], dim=1)        # (B,2,H,W)
        s = torch.sigmoid(self.conv(s))             # (B,1,H,W)
        return s

class ChannelAttn(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, ch//r, 1)
        self.fc2 = nn.Conv2d(ch//r, ch, 1)
    def forward(self, f_g, f_l):
        # global avg & max pooling
        def gp(x): return F.adaptive_avg_pool2d(x, 1)
        def mp(x): return F.adaptive_max_pool2d(x, 1)
        u = gp(f_g) + mp(f_l)                       # (B,C,1,1)
        u = torch.relu(self.fc1(u))
        u = torch.sigmoid(self.fc2(u))              # (B,C,1,1)
        return u

class CFDA(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.sattn = SpatialAttn(ch)
        self.cattn = ChannelAttn(ch)
        self.out = nn.Conv2d(2*ch, ch, 1)
    def forward(self, f_g, f_l):
        # Spatial refinement (cross)
        s = self.sattn(f_g, f_l)
        fg1 = s * f_l
        fl1 = s * f_g
        # Channel refinement (cross)
        c = self.cattn(fg1, fl1)
        fg2 = c * fl1
        fl2 = c * fg1
        f = torch.cat([fg2, fl2], dim=1)
        return self.out(f)                           # fused

# ---------- Decoder block with upsample + CFDA fusion ----------
class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.cfda = CFDA(out_ch)                    # fuse after aligning channels
        self.conv = ConvBlock(out_ch, out_ch)
        self.align = nn.Conv2d(skip_ch, out_ch, 1)
    def forward(self, x, skip):
        x = self.up(x)
        skip = self.align(skip)
        x = self.cfda(x, skip)
        x = self.conv(x)
        return x

# ---------- MSDC-Net ----------
class MSDCNet(nn.Module):
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        # Encoder
        self.e1 = ConvBlock(in_ch, base)            # 128 -> 128
        self.p1 = nn.MaxPool2d(2)                   # 128 -> 64
        self.e2 = ConvBlock(base, base*2)           # 64
        self.p2 = nn.MaxPool2d(2)                   # 64 -> 32
        self.e3 = ConvBlock(base*2, base*4)         # 32
        self.p3 = nn.MaxPool2d(2)                   # 32 -> 16

        # Bottleneck with MSSA (C=base*8 == 256 if base=32)
        self.b0 = ConvBlock(base*4, base*8)         # 16
        self.mssa = MSSA(base*8)

        # Decoder with CFDA in skips
        self.u3 = UpBlock(base*8, base*4, base*4)   # 16->32
        self.u2 = UpBlock(base*4, base*2, base*2)   # 32->64
        self.u1 = UpBlock(base*2, base, base)       # 64->128

        self.out = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x):
        s1 = self.e1(x)       # 128
        s2 = self.e2(self.p1(s1))  # 64
        s3 = self.e3(self.p2(s2))  # 32
        b  = self.b0(self.p3(s3))  # 16
        b  = self.mssa(b)          # 16

        x  = self.u3(b, s3)
        x  = self.u2(x, s2)
        x  = self.u1(x, s1)
        return self.out(x)     # logits

# ---------- Loss: BCE + Dice ----------
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        num = 2 * (probs*targets).sum(dim=(1,2,3)) + self.eps
        den = (probs+targets).sum(dim=(1,2,3)) + self.eps
        dice = 1 - (num/den)
        return dice.mean()

def bce_dice_loss(logits, targets, w_bce=0.5, w_dice=0.5):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = DiceLoss()(logits, targets)
    return w_bce*bce + w_dice*dice

# ---------- Minimal training loop (sketch) ----------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = MSDCNet(in_ch=1, base=32).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    # dummy batch: replace with real 128x128 CT slices and binary masks
    x = torch.randn(4, 1, 128, 128, device=device)
    y = (torch.rand(4, 1, 128, 128, device=device) > 0.5).float()

    net.train()
    for _ in range(10):
        opt.zero_grad()
        logits = net(x)
        loss = bce_dice_loss(logits, y)
        loss.backward()
        opt.step()
    print("OK, final loss:", float(loss))


# In[ ]:




