"""
任务 C PTC 分类器：输入由任务 A 的 mask 裁剪出的结节 patch（patch_size×patch_size RGB），
输出单通道 logit（sigmoid 后为 PTC 概率）。
backbone 优先用 timm 的 EfficientNet（ImageNet 预训练）；timm 不可用时自动回退到内置轻量 CNN。
"""
import torch
import torch.nn as nn


class NoduleClassifier(nn.Module):
    """内置轻量 CNN（timm 不可用时的回退方案）。"""

    def __init__(self, base=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, base, 3, padding=1), nn.BatchNorm2d(base), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(base, base * 2, 3, padding=1), nn.BatchNorm2d(base * 2), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(base * 2, base * 4, 3, padding=1), nn.BatchNorm2d(base * 4), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(base * 4, base * 8, 3, padding=1), nn.BatchNorm2d(base * 8), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base * 8, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.head(self.features(x))


def build_classifier(patch_size=224, backbone="efficientnet_b3", pretrained=True, use_timm=True):
    if use_timm:
        try:
            import timm
            model = timm.create_model(backbone, pretrained=pretrained, num_classes=1)
            print(f"[classifier] 使用 timm backbone={backbone} pretrained={pretrained}")
            return model
        except Exception as e:
            print(f"[classifier] timm 不可用（{e}），回退到内置 CNN")
    return NoduleClassifier()
