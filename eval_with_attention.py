import torch
import model_architectures.resnet_cifar 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm

# —— 下面两段就是你之前给的 Attention 模块 —— 
class SlotAttention(nn.Module):
    def __init__(self, num_slots, in_dim, slot_dim, iters=3):
        super().__init__()
        self.num_slots = num_slots
        self.iters      = iters
        self.slot_dim   = slot_dim
        self.scale      = slot_dim ** -0.5

        self.slots_mu   = nn.Parameter(torch.randn(1, num_slots, slot_dim))
        self.norm_inputs= nn.LayerNorm(in_dim)
        self.to_k       = nn.Linear(in_dim, slot_dim, bias=False)
        self.to_v       = nn.Linear(in_dim, slot_dim, bias=False)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.to_q       = nn.Linear(slot_dim, slot_dim, bias=False)

        self.gru        = nn.GRUCell(slot_dim, slot_dim)
        self.norm_mlp   = nn.LayerNorm(slot_dim)
        self.mlp        = nn.Sequential(
            nn.Linear(slot_dim, slot_dim),
            nn.ReLU(inplace=True),
            nn.Linear(slot_dim, slot_dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        x_flat = x.view(B, C, N).permute(0,2,1)    # [B, N, C]
        x_norm = self.norm_inputs(x_flat)

        slots = self.slots_mu.expand(B, -1, -1)    # [B, S, D]
        for _ in range(self.iters):
            q = self.to_q(self.norm_slots(slots))  # [B, S, D]
            k = self.to_k(x_norm)                  # [B, N, D]
            v = self.to_v(x_norm)                  # [B, N, D]
            attn_logits = torch.einsum('bnd,bsd->bns', k, q) * self.scale
            attn       = F.softmax(attn_logits, dim=-1)  # [B, N, S]
            updates    = torch.einsum('bns,bnd->bsd', attn, v)
            slots_flat   = slots.view(-1, self.slot_dim)
            updates_flat = updates.view(-1, self.slot_dim)
            slots = self.gru(updates_flat, slots_flat).view_as(slots)
            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots  # [B, S, D]

class CrossAttention(nn.Module):
    def __init__(self, feat_dim, slot_dim, num_heads):
        super().__init__()
        assert slot_dim % num_heads == 0, "slot_dim 必须可被 num_heads 整除"
        self.num_heads = num_heads
        self.head_dim   = slot_dim // num_heads
        self.scale      = self.head_dim ** -0.5

        self.to_q       = nn.Linear(feat_dim,   slot_dim, bias=False)
        self.to_k       = nn.Linear(slot_dim,   slot_dim, bias=False)
        self.to_v       = nn.Linear(slot_dim,   slot_dim, bias=False)
        self.proj       = nn.Linear(slot_dim,   feat_dim)

    def forward(self, feat, slots):
        B, C, H, W = feat.shape
        N = H * W
        S = slots.size(1)

        q = self.to_q(feat.view(B, C, N).permute(0,2,1))  # [B, N, D]
        k = self.to_k(slots)                              # [B, S, D]
        v = self.to_v(slots)                              # [B, S, D]

        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1,2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1,2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1,2)

        attn = torch.matmul(q, k.transpose(-2,-1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out  = torch.matmul(attn, v)
        out  = out.transpose(1,2).contiguous().view(B, N, -1)
        out  = self.proj(out)
        return out.permute(0,2,1).view(B, C, H, W)

# —— 原始的客户端/服务端模型定义 —— 
from model_architectures.resnet_cifar import ResNet20

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    # —— 1) CIFAR10 数据集 —— 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))
    ])
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

    # —— 2) 实例化 Split-Model —— 
    #    这里 cutting_layer=4（随意选），client 端就是 self.f = local_list[0]
    model = ResNet20(
        cutting_layer=4,
        logger=logger,
        num_client=1,
        num_class=10,
        initialize_different=False,
        adds_bottleneck=False,
        bottleneck_option="None",
        double_local_layer=False,
        upsize=False
    ).to(device)

    f       = model.local_list[0].to(device)
    f_tail  = model.cloud.to(device)
    cls_head= model.classifier.to(device)

    f.eval(); f_tail.eval(); cls_head.eval()

    # —— 3) Attention 模块 —— 
    # 先跑一批数据拿通道数 C, 高宽 H,W
    sample = next(iter(testloader))[0][:1].to(device)  # [1,3,32,32]
    with torch.no_grad():
        z = f(sample)  # e.g. [1, C, H, W]
    B, C, H, W = z.shape
    num_slots = 8
    slot_dim  = C     # 你也可以自定义不同维度
    num_heads = 4

    slot_attn  = SlotAttention(num_slots=num_slots, in_dim=C, slot_dim=slot_dim, iters=3).to(device)
    cross_attn = CrossAttention(feat_dim=C, slot_dim=slot_dim, num_heads=num_heads).to(device)
    slot_attn.eval(); cross_attn.eval()

    # —— 4) 基线：不加 Attention —— 
    correct_base = 0
    total = 0
    with torch.no_grad():
        for imgs, labs in tqdm(testloader, desc="Baseline"):
            imgs, labs = imgs.to(device), labs.to(device)
            feat = f(imgs)
            out  = f_tail(feat)
            # ResNet20 的 classifier 在 cloud 端之后需要 avgpool
            out  = F.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)
            logits = cls_head(out)
            preds = logits.argmax(1)
            correct_base += preds.eq(labs).sum().item()
            total += labs.size(0)

    acc_base = correct_base / total
    print(f"\nBaseline Acc (no attention): {acc_base*100:.2f}%")

    # —— 5) 加上 Slot + Cross Attention —— 
    correct_attn = 0
    total = 0
    with torch.no_grad():
        for imgs, labs in tqdm(testloader, desc="With Attention"):
            imgs, labs = imgs.to(device), labs.to(device)
            feat = f(imgs)                          # [B, C, H, W]
            slots= slot_attn(feat)                  # [B, S, D]
            feat2= cross_attn(feat, slots)          # [B, C, H, W]
            out  = f_tail(feat2)
            out  = F.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)
            logits = cls_head(out)
            preds = logits.argmax(1)
            correct_attn += preds.eq(labs).sum().item()
            total += labs.size(0)

    acc_attn = correct_attn / total
    print(f"\nWith Slot→Cross Attention: {acc_attn*100:.2f}%")

if __name__ == "__main__":
    main()