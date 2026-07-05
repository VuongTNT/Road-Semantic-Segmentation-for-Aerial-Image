import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights

class ASPPConv(nn.Sequential):
    """Standard Conv block with dilation for ASPP"""
    def __init__(self, in_channels, out_channels, dilation):
        super(ASPPConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class ASPPPooling(nn.Sequential):
    """Global Average Pooling branch for ASPP"""
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[-2:]
        out = super(ASPPPooling, self).forward(x)
        return F.interpolate(out, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling Module adapted for ResNet-34 (512 input channels)"""
    def __init__(self, in_channels=512, out_channels=256, rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        modules = []
        # 1. 1x1 Convolution branch
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # 2. Three Dilated 3x3 Convolution branches
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
            
        # 3. Image Pooling branch
        modules.append(ASPPPooling(in_channels, out_channels))
        
        self.convs = nn.ModuleList(modules)
        
        # 4. Final Projection layer fusing all 5 branches
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabV3PlusResNet34(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(DeepLabV3PlusResNet34, self).__init__()
        
        # 1. ENCODER: Extract backbone from ResNet-34
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        base_resnet = resnet34(weights=weights)
        
        self.initial_blocks = nn.Sequential(
            base_resnet.conv1,
            base_resnet.bn1,
            base_resnet.relu
        )
        self.maxpool = base_resnet.maxpool
        
        self.layer1 = base_resnet.layer1  # Low-level features: 64 channels, 1/4 resolution
        self.layer2 = base_resnet.layer2  # 128 channels, 1/8 resolution
        self.layer3 = base_resnet.layer3  # 256 channels, 1/16 resolution
        self.layer4 = base_resnet.layer4  # High-level features: 512 channels, 1/32 resolution

        # 2. BOTTLENECK: Multi-scale context gathering via ASPP
        self.aspp = ASPP(in_channels=512, out_channels=256)

        # 3. DECODER: Feature refinement and fusion
        # Low-level projection (reduces channels from early encoder to avoid overpowering high-level features)
        self.low_level_project = nn.Sequential(
            nn.Conv2d(64, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Main Decoder Refining Layers (processes fused 256 + 48 = 304 channels)
        self.decoder_head = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        input_size = x.shape[-2:]  # Keep original H, W for final upsampling
        
        # --- Encoder Forward ---
        x = self.initial_blocks(x)
        low_level_features = self.layer1(x)  # Retain for decoder connection (1/4 size)
        
        x = self.maxpool(low_level_features)
        x = self.layer2(x)
        x = self.layer3(x)
        high_level_features = self.layer4(x) # 512 channels (1/32 size)

        # --- ASPP Bottleneck ---
        aspp_features = self.aspp(high_level_features) # 256 channels
        
        # --- Decoder Forward ---
        # 1. Upsample high-level ASPP features 8x to meet low-level size (1/4 size)
        aspp_features_upsampled = F.interpolate(
            aspp_features, size=low_level_features.shape[-2:], 
            mode='bilinear', align_corners=False
        )
        
        # 2. Process low-level features
        low_features_projected = self.low_level_project(low_level_features)
        
        # 3. Concatenate along channel dimension (U-Net style channel stacking)
        fused_features = torch.cat([aspp_features_upsampled, low_features_projected], dim=1)
        
        # 4. Refine via convolution blocks
        decoder_output = self.decoder_head(fused_features)
        
        # --- Final Upsampling to Match Input Resolution ---
        output = F.interpolate(decoder_output, size=input_size, mode='bilinear', align_corners=False)
        return output