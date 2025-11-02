import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module"""
    def __init__(self, in_channels, out_channels=256, dilation_rates=[1, 6, 12, 18]):
        super(ASPP, self).__init__()
        
        # 1x1 convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Atrous convolutions with different dilation rates
        self.atrous_convs = nn.ModuleList()
        for dilation in dilation_rates[1:]:
            self.atrous_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=dilation, 
                             dilation=dilation, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Global average pooling
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Project concatenated features
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(dilation_rates) + 1), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        size = x.shape[2:]
        
        # Apply all parallel convolutions
        feat1 = self.conv1(x)
        atrous_feats = [conv(x) for conv in self.atrous_convs]
        
        # Global pooling and upsample
        global_feat = self.global_avg_pool(x)
        global_feat = F.interpolate(global_feat, size=size, mode='bilinear', align_corners=False)
        
        # Concatenate all features
        out = torch.cat([feat1] + atrous_feats + [global_feat], dim=1)
        
        # Project to output channels
        out = self.project(out)
        return out


class CompactSegmentationModel(nn.Module):
    """Compact segmentation model with MobileNetV3-Small encoder and ASPP decoder"""
    def __init__(self, num_classes=21, pretrained=True):
        super(CompactSegmentationModel, self).__init__()
        
        # Load pretrained MobileNetV3-Small as backbone
        mobilenet = mobilenet_v3_small(pretrained=pretrained)
        self.backbone = mobilenet.features
        
        # Feature extraction indices for skip connections
        # MobileNetV3-Small feature channels at different strides:
        # stride 4: layer 1 (16 channels)
        # stride 8: layer 3 (24 channels)  
        # stride 16: layer 8 (48 channels)
        # stride 16: layer 12 (576 channels) - final
        
        self.low_level_idx = 1   # stride ~4, 16 channels
        self.mid_level_idx = 3   # stride ~8, 24 channels
        self.high_level_idx = 12  # stride ~16, 576 channels
        
        # ASPP module on high-level features
        self.aspp = ASPP(in_channels=576, out_channels=256)
        
        # Low-level feature projection (reduce channels)
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(16, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Mid-level feature projection
        self.mid_level_conv = nn.Sequential(
            nn.Conv2d(24, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Final classifier
        self.classifier = nn.Conv2d(256, num_classes, 1)
        
    def forward(self, x, return_features=False):
        input_size = x.shape[2:]
        
        # Extract features at different levels
        features = {}
        feat = x
        
        for idx, layer in enumerate(self.backbone):
            feat = layer(feat)
            if idx == self.low_level_idx:
                low_level_feat = feat
                features['low'] = feat
            elif idx == self.mid_level_idx:
                mid_level_feat = feat
                features['mid'] = feat
            elif idx == self.high_level_idx:
                high_level_feat = feat
                features['high'] = feat
        
        # ASPP on high-level features
        aspp_out = self.aspp(high_level_feat)
        
        # Upsample ASPP output to match mid-level size
        aspp_up = F.interpolate(aspp_out, size=mid_level_feat.shape[2:], 
                               mode='bilinear', align_corners=False)
        
        # Process low and mid level features
        low_proj = self.low_level_conv(low_level_feat)
        low_proj = F.interpolate(low_proj, size=mid_level_feat.shape[2:],
                                mode='bilinear', align_corners=False)
        
        mid_proj = self.mid_level_conv(mid_level_feat)
        
        # Concatenate multi-scale features
        decoder_input = torch.cat([aspp_up, mid_proj, low_proj], dim=1)
        
        # Decode
        decoder_out = self.decoder(decoder_input)
        
        # Final classification
        logits = self.classifier(decoder_out)
        
        # Upsample to input size
        output = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
        
        if return_features:
            return output, features
        return output
    
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = CompactSegmentationModel(num_classes=21, pretrained=False)
    print(f"Number of parameters: {model.get_num_parameters():,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 256)
    output, features = model(x, return_features=True)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Low-level features shape: {features['low'].shape}")
    print(f"Mid-level features shape: {features['mid'].shape}")
    print(f"High-level features shape: {features['high'].shape}")