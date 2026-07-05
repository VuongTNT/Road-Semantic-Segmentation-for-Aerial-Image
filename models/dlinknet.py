import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

class DBlock(nn.Module):
    def __init__(self, channels):
        super(DBlock, self).__init__()
        self.dilate1 = nn.Conv2d(channels, channels, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channels, channels, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channels, channels, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channels, channels, kernel_size=3, dilation=8, padding=8)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        d1 = self.relu(self.dilate1(x))
        d2 = self.relu(self.dilate2(d1))
        d3 = self.relu(self.dilate3(d2))
        d4 = self.relu(self.dilate4(d3))
        
        out = x + d1 + d2 + d3 + d4
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_channels // 4)  # norm1 -> bn1
        self.relu1 = nn.ReLU(inplace=True)

        self.deconv3d = nn.ConvTranspose3d(           # deconv2 -> deconv3d
            in_channels // 4, in_channels // 4, 
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), output_padding=(0, 1, 1)
        )
        self.bn2 = nn.BatchNorm2d(in_channels // 4)   # norm2 -> bn2
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels // 4, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)        # norm3 -> bn3
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = x.unsqueeze(2)
        x = self.deconv3d(x)
        x = x.squeeze(2)
        
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x


class DLinkNet34(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(DLinkNet34, self).__init__()
        
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        base_resnet = resnet34(weights=weights)
        
        self.firstconv = base_resnet.conv1
        self.firstbn = base_resnet.bn1
        self.firstrelu = base_resnet.relu
        self.firstmaxpool = base_resnet.maxpool
        
        self.encoder1 = base_resnet.layer1  
        self.encoder2 = base_resnet.layer2  
        self.encoder3 = base_resnet.layer3  
        self.encoder4 = base_resnet.layer4  

        self.dblock = DBlock(512)

        self.decoder4 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, 64)

        self.finaldeconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)

        e1 = self.encoder1(x)  
        e2 = self.encoder2(e1) 
        e3 = self.encoder3(e2) 
        e4 = self.encoder4(e3) 

        center = self.dblock(e4)

        d4 = self.decoder4(center) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return out