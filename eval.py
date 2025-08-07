# eval_with_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model_architectures.resnet_cifar import ResNet20
import logging
from tqdm import tqdm


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
        attn = None
        for _ in range(self.iters):
            q = self.to_q(self.norm_slots(slots))  # [B, S, D]
            k = self.to_k(x_norm)                  # [B, N, D]
            v = self.to_v(x_norm)                  # [B, N, D]
            attn_logits = torch.einsum('bnd,bsd->bns', k, q) * self.scale
            attn       = F.softmax(attn_logits, dim=-1)  # [B, N, S]
            updates    = torch.einsum('bns,bnd->bsd', attn, v)
            slots_flat   = slots.reshape(-1, self.slot_dim)
            updates_flat = updates.reshape(-1, self.slot_dim)
            slots = self.gru(updates_flat, slots_flat).view_as(slots)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots, attn  # slots: [B, S, D], attn: [B, N, S]


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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    # 1) CIFAR10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))
    ])
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    loader  = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

    # 2) Split ResNet20
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

    # 3) Slot+Cross Attention
    # sample to get C,H,W
    sample = next(iter(loader))[0][:1].to(device)
    with torch.no_grad():
        z = f(sample)  # [1,C,H,W]
    B, C, H, W = z.shape
    num_slots = 8
    slot_dim  = C
    num_heads = 4
    slot_attn  = SlotAttention(num_slots, in_dim=C, slot_dim=slot_dim, iters=3).to(device)
    cross_attn = CrossAttention(feat_dim=C, slot_dim=slot_dim, num_heads=num_heads).to(device)
    slot_attn.eval(); cross_attn.eval()

    # 4) 基线 & Attention 分类准确率
    correct_base = 0
    correct_attn = 0
    total = 0

    # 为后面统计 GMM 类似参数，收集所有样本的 slots & weights & labels
    all_slots   = []
    all_weights = []
    all_labels  = []

    with torch.no_grad():
        for imgs, labs in tqdm(loader, desc="Eval"):
            imgs, labs = imgs.to(device), labs.to(device)
            feat = f(imgs)                         # [B,C,H,W]

            # —— baseline —— 
            outb = f_tail(feat)
            outb = F.adaptive_avg_pool2d(outb,1).view(outb.size(0),-1)
            preds = cls_head(outb).argmax(1)
            correct_base += preds.eq(labs).sum().item()

            # —— attention 流 —— 
            slots, attn = slot_attn(feat)          # slots:[B,S,D], attn:[B,N,S]
            feat2 = cross_attn(feat, slots)        # [B,C,H,W]
            outa  = f_tail(feat2)
            outa  = F.adaptive_avg_pool2d(outa,1).view(outa.size(0),-1)
            preda = cls_head(outa).argmax(1)
            correct_attn += preda.eq(labs).sum().item()

            total += labs.size(0)

            # 收集参数
            # slots_flat: [B,S,D]
            # weights: 空间上对每个 slot 的平均注意力 => [B,S]
            weights = attn.mean(dim=1)  # [B,S]
            all_slots.append(slots.cpu())
            all_weights.append(weights.cpu())
            all_labels.append(labs.cpu())

    acc_base = correct_base/total
    acc_attn= correct_attn/total
    print(f"\nBaseline Acc (no attn): {acc_base*100:.2f}%")
    print(f"With Slot→Cross Attn: {acc_attn*100:.2f}%\n")

    # 5) 计算“GMM 参数”：means, covariances, weights
    slots_all   = torch.cat(all_slots, dim=0)    # [N, S, D]
    weights_all = torch.cat(all_weights, dim=0)  # [N, S]
    labels_all  = torch.cat(all_labels, dim=0)   # [N]

    num_classes = 10
    S, D = slots_all.shape[1], slots_all.shape[2]

    means      = torch.zeros(num_classes, S, D)
    covariances= torch.zeros(num_classes, S)
    cluster_ws = torch.zeros(num_classes, S)

    for c in range(num_classes):
        mask = (labels_all == c)
        if mask.sum()==0: continue
        slots_c   = slots_all[mask]       # [Nc, S, D]
        weights_c = weights_all[mask]     # [Nc, S]

        # 均值
        means[c] = slots_c.mean(dim=0)    # [S,D]
        # 方差: 每个 slot_index 上特征维度的 var，再对 D 求平均
        var_sd = slots_c.var(dim=0)       # [S,D]
        covariances[c] = var_sd.mean(dim=1)  # [S]
        # 权重: attention 的平均
        cluster_ws[c] = weights_c.mean(dim=0)  # [S]

    # 打印一下每个类的前三个 slot parameters
    for c in range(num_classes):
        print(f"Class {c:2d}:")
        print("  slot-weights[0:3]:", cluster_ws[c,:3].numpy())
        print("  slot-cov[0:3]:    ", covariances[c,:3].numpy())
        print("  slot-means[0:3]:  ", means[c,:3,:3].numpy(), "...")  # 只看前三维
    # 也可以选择把 means/cov/weights 保存到 .pt 文件：
    # torch.save(dict(means=means, cov=covariances, w=cluster_ws), "attn_params_cifar10.pt")


if __name__ == "__main__":
    main()