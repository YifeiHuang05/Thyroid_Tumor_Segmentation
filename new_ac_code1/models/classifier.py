"""
任务 C 多任务分类器：输入由任务 A 的 mask 裁剪出的结节 patch（patch_size×patch_size RGB），
共享同一 backbone，并行的多个分类头（dict 返回）：
  - head_bm     : 良性 vs 恶性  (2 类)
  - head_ptc    : 在“恶性”内部区分 PTC vs 非 PTC (2 类；仅用于恶性结节)
  - head_fnac   : FNAC 三分类 Bethesda II / III-IV / V-VI (3 类)
  - head_tirands: TI-RADS 五分类 (5 类)
forward(x) 返回 {name: logits}，各 logits 形状 (N, num_classes)。

backbone 优先用 timm 的 EfficientNet（ImageNet 预训练，num_classes=0 取特征）；
timm 不可用时自动回退到内置轻量 CNN。
"""
import torch
import torch.nn as nn


class NoduleTrunk(nn.Module):
    """内置轻量 CNN 骨干（timm 不可用时的回退）。输出 (B, base*8) 特征。"""

    def __init__(self, base=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, base, 3, padding=1), nn.BatchNorm2d(base), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(base, base * 2, 3, padding=1), nn.BatchNorm2d(base * 2), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(base * 2, base * 4, 3, padding=1), nn.BatchNorm2d(base * 4), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(base * 4, base * 8, 3, padding=1), nn.BatchNorm2d(base * 8), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        return self.features(x).flatten(1)


class HierClassifier(nn.Module):
    """多任务头分类器：共享 backbone，多个并行分类头（ModuleDict）。"""

    def __init__(self, trunk, num_features, heads):
        super().__init__()
        self.trunk = trunk
        self.heads = nn.ModuleDict({name: nn.Linear(num_features, n) for name, n, _ in heads})

    def forward(self, x):
        f = self.trunk(x)
        if isinstance(f, (tuple, list)):
            f = f[0]
        return {name: head(f) for name, head in self.heads.items()}


# ---------------- 头定义重建（训练 / 加载通用） ----------------
def get_heads(mcfg):
    """mcfg: Config 实例或 dict（如 checkpoint 里的 config）。返回 [(name, num_classes, names), ...]。"""
    if isinstance(mcfg, dict):
        def gid(k, default):
            return mcfg.get(k, default)
    else:
        def gid(k, default):
            return getattr(mcfg, k, default)
    return [
        ("bm", gid("bm_num_classes", 2), list(gid("bm_names", ("benign", "malignant")))),
        ("ptc", gid("ptc_num_classes", 2), list(gid("ptc_names", ("non_ptc", "ptc")))),
        ("fnac", gid("fnac_num_classes", 3), list(gid("fnac_names", ("Bethesda_II", "Bethesda_III-IV", "Bethesda_V-VI")))),
        ("tirands", gid("tirands_num_classes", 5), list(gid("tirands_names", ("TR1", "TR2", "TR3", "TR4", "TR5")))),
    ]


def build_classifier(patch_size=224, backbone="efficientnet_b3", pretrained=True,
                     use_timm=True, heads=None):
    if heads is None:
        heads = get_heads({})
    if use_timm:
        try:
            import timm
            # num_classes=0 -> 去掉分类头，forward 返回池化后的特征 (B, num_features)
            trunk = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
            num_features = trunk.num_features
            print(f"[classifier] 使用 timm backbone={backbone} num_features={num_features} "
                  f"pretrained={pretrained}  heads={[h[0] for h in heads]}")
            return HierClassifier(trunk, num_features, heads)
        except Exception as e:
            print(f"[classifier] timm 不可用（{e}），回退到内置 CNN")
    trunk = NoduleTrunk(base=32)
    num_features = 32 * 8
    print(f"[classifier] 使用内置 CNN trunk num_features={num_features} heads={[h[0] for h in heads]}")
    return HierClassifier(trunk, num_features, heads)
