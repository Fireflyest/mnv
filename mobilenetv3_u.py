import torch
import torch.nn as nn
import torchvision.models as models

class MultiHeadMobileNetV3(nn.Module):
    def __init__(self, num_classes):
        super(MultiHeadMobileNetV3, self).__init__()
        # 加载预训练的 MobileNetV3 模型
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        mobilenet = models.mobilenet_v3_small(weights=weights)
        self.features = mobilenet.features
        # 分类系列拆出来用于输出特征
        self.avgpool = mobilenet.avgpool
        self.classifier = mobilenet.classifier
        mobilenet.classifier = nn.Identity()
        mobilenet.avgpool = nn.Identity()
        # 获取分类层的输入特征数，把分类层的线性层替换为一个空层用于连接自定义的头
        classifier_in_features = self.classifier[3].in_features
        self.classifier[3] = nn.Identity()
        # 添加多头输出
        self.head1 = nn.Sequential(
            nn.Linear(classifier_in_features, num_classes),
            nn.Softmax(dim=1)
        )
        self.head2 = nn.Sequential(
            nn.Linear(classifier_in_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.features[:2](x)       # [B, 16, H/2, W/2]
        x2 = self.features[2:4](x1)     # [B, 24, H/4, W/4]
        x3 = self.features[4:7](x2)     # [B, 40, H/8, W/8]
        x4 = self.features[7:10](x3)    # [B, 80, H/16, W/16]
        x5 = self.features[10:](x4)     # [B, 576, H/32, W/32]
        x6 = torch.flatten(self.avgpool(x5), 1)
        x7 = self.classifier(x6)
        out1 = self.head1(x7)
        out2 = self.head2(x7)

        pool = nn.AdaptiveAvgPool2d((1, 1))
        f1 = torch.flatten(pool(x1), 1)      # [B, 16]
        f2 = torch.flatten(pool(x2), 1)      # [B, 24]
        f3 = torch.flatten(pool(x3), 1)      # [B, 40]
        f4 = torch.flatten(pool(x4), 1)      # [B, 80]
        f5 = torch.flatten(pool(x5), 1)      # [B, 576]

        # Concatenate all features
        features = torch.cat([f1, f2, f3, f4, f5], dim=1)  # [B, 16+24+40+80+576=736]
        return out1, out2, features

# Encoder part with deep features
class MobileNetV3Encoder(nn.Module):
    def __init__(self, num_classes=5):
        super(MobileNetV3Encoder, self).__init__()
        # Load pretrained MobileNetV3
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        mobilenet = models.mobilenet_v3_small(weights=weights)
        self.features = mobilenet.features
        
        # Classification components
        self.avgpool = mobilenet.avgpool
        self.classifier = mobilenet.classifier
        mobilenet.classifier = nn.Identity()
        mobilenet.avgpool = nn.Identity()
        classifier_in_features = self.classifier[3].in_features
        self.classifier[3] = nn.Identity()
        
        # Multi-head outputs
        self.head1 = nn.Sequential(
            nn.Linear(classifier_in_features, num_classes),
            nn.Softmax(dim=1)
        )
        self.head2 = nn.Sequential(
            nn.Linear(classifier_in_features, 1),
            nn.Sigmoid()
        )
        
        # Channel reduction layer without changing spatial dimensions
        # Changed from downsampling to channel reduction only
        self.extra_downsample = nn.Sequential(
            nn.Conv2d(576, 96, kernel_size=1, stride=1, padding=0),  # 1x1 conv to change channels only
            nn.BatchNorm2d(96),
            nn.Hardswish(inplace=True)
        )

    def forward(self, x):
        # Encoder path - correcting the slicing indices based on model printout
        x1 = self.features[0](x)         # [B, 16, H/2, W/2]
        x2 = self.features[1:4](x1)      # [B, 24, H/4, W/4]
        x3 = self.features[4:9](x2)      # [B, 48, H/8, W/8]
        x4 = self.features[9:12](x3)     # [B, 96, H/16, W/16]
        x5 = self.features[12:](x4)      # [B, 576, H/32, W/32]
        
        # Apply extra_downsample - now it's only changing channels, not spatial dimensions
        x6 = self.extra_downsample(x5)  # [B, 512, H/32, W/32] - same spatial size as x5
        
        # Classification path (using x5 as in the original model)
        x_cls = torch.flatten(self.avgpool(x5), 1)
        x_cls = self.classifier(x_cls)
        out1 = self.head1(x_cls)
        out2 = self.head2(x_cls)
        
        # Feature extraction for original multi-task output
        pool = nn.AdaptiveAvgPool2d((1, 1))
        f1 = torch.flatten(pool(x1), 1)      # [B, 16]
        f2 = torch.flatten(pool(x2), 1)      # [B, 24]
        f3 = torch.flatten(pool(x3), 1)      # [B, 40]
        f4 = torch.flatten(pool(x4), 1)      # [B, 80]
        f5 = torch.flatten(pool(x5), 1)      # [B, 576]
        f6 = torch.flatten(pool(x6), 1)      # [B, 96]
        features = torch.cat([f1, f2, f3, f4, f5, f6], dim=1)  # [B, 16+24+40+80+576+96=832]
        
        # Return classification outputs, feature vector, and all feature maps for decoder
        return out1, out2, features, x1, x2, x3, x4, x5, x6

# Separate Decoder part
class MobileNetV3Decoder(nn.Module):
    def __init__(self, output_channels=3):
        super(MobileNetV3Decoder, self).__init__()
        
        # Since x6 is now the same spatial dimensions as x5, we don't need upsampling
        # Changed from ConvTranspose2d to 1x1 Conv to match dimensions only
        self.decoder0 = nn.Sequential(
            nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0),  # Changed from ConvTranspose2d
            nn.BatchNorm2d(576),
            nn.ReLU(inplace=True)
        )
        
        # Other decoder blocks remain transposed convs for upsampling
        # Decoder block 1: [B, 576+576=1152, H/32, W/32] -> [B, 96, H/16, W/16]
        # Corrected: x5 has 576 channels based on the model printout
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(1152, 96, kernel_size=2, stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        
        # Decoder block 2: [B, 96+96=192, H/16, W/16] -> [B, 40, H/8, W/8]
        # Corrected: x4 has 96 channels based on the model printout
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(192, 48, kernel_size=2, stride=2),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Decoder block 3: [B, 48+48=96, H/8, W/8] -> [B, 24, H/4, W/4]
        # Corrected: x3 has 48 channels based on the model printout
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(96, 24, kernel_size=2, stride=2),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        
        # Decoder block 4: [B, 24+24=48, H/4, W/4] -> [B, 16, H/2, W/2]
        # Corrected: x2 has 24 channels based on the model printout
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(48, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # Final upsampling: [B, 16+16=32, H/2, W/2] -> [B, output_channels, H, W]
        # Corrected: x1 has 16 channels based on the model printout
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose2d(32, output_channels, kernel_size=2, stride=2),
            nn.Sigmoid()  # Sigmoid for output in range [0,1]
        )
        
        # Add debug flag
        self.debug = False
        
    def forward(self, x1, x2, x3, x4, x5, x6):
        if self.debug:
            print(f"Input shapes: x1: {x1.shape}, x2: {x2.shape}, x3: {x3.shape}, x4: {x4.shape}, x5: {x5.shape}, x6: {x6.shape}")
        
        # Decoder path with skip connections
        d0 = self.decoder0(x6)  # [B, 576, H/32, W/32]
        if self.debug:
            print(f"d0 shape: {d0.shape}")
        
        # Ensure x5 and d0 have the same spatial dimensions
        if x5.shape[2:] != d0.shape[2:]:
            d0 = nn.functional.interpolate(d0, size=x5.shape[2:], mode='bilinear', align_corners=False)
            if self.debug:
                print(f"d0 after interpolation: {d0.shape}")
        
        # Cat d0 and x5
        d0 = torch.cat([d0, x5], dim=1)  # [B, 576+576=1152, H/32, W/32]
        if self.debug:
            print(f"d0 after concat: {d0.shape}")
        
        d1 = self.decoder1(d0)  # [B, 96, H/16, W/16]
        if self.debug:
            print(f"d1 shape: {d1.shape}")
        
        # Ensure x4 and d1 have the same spatial dimensions
        if x4.shape[2:] != d1.shape[2:]:
            d1 = nn.functional.interpolate(d1, size=x4.shape[2:], mode='bilinear', align_corners=False)
            if self.debug:
                print(f"d1 after interpolation: {d1.shape}")
        
        # Cat d1 and x4
        d1 = torch.cat([d1, x4], dim=1)  # [B, 96+96=192, H/16, W/16]
        if self.debug:
            print(f"d1 after concat: {d1.shape}")
        
        d2 = self.decoder2(d1)  # [B, 48, H/8, W/8]
        if self.debug:
            print(f"d2 shape: {d2.shape}")
        
        # Ensure x3 and d2 have the same spatial dimensions
        if x3.shape[2:] != d2.shape[2:]:
            d2 = nn.functional.interpolate(d2, size=x3.shape[2:], mode='bilinear', align_corners=False)
            if self.debug:
                print(f"d2 after interpolation: {d2.shape}")
        
        # Cat d2 and x3
        d2 = torch.cat([d2, x3], dim=1)  # [B, 48+48=96, H/8, W/8]
        if self.debug:
            print(f"d2 after concat: {d2.shape}")
        
        d3 = self.decoder3(d2)  # [B, 24, H/4, W/4]
        if self.debug:
            print(f"d3 shape: {d3.shape}")
        
        # Ensure x2 and d3 have the same spatial dimensions
        if x2.shape[2:] != d3.shape[2:]:
            d3 = nn.functional.interpolate(d3, size=x2.shape[2:], mode='bilinear', align_corners=False)
            if self.debug:
                print(f"d3 after interpolation: {d3.shape}")
        
        # Cat d3 and x2
        d3 = torch.cat([d3, x2], dim=1)  # [B, 24+24=48, H/4, W/4]
        if self.debug:
            print(f"d3 after concat: {d3.shape}")
        
        d4 = self.decoder4(d3)  # [B, 16, H/2, W/2]
        if self.debug:
            print(f"d4 shape: {d4.shape}")
        
        # Ensure x1 and d4 have the same spatial dimensions
        if x1.shape[2:] != d4.shape[2:]:
            d4 = nn.functional.interpolate(d4, size=x1.shape[2:], mode='bilinear', align_corners=False)
            if self.debug:
                print(f"d4 after interpolation: {d4.shape}")
        
        # Cat d4 and x1
        d4 = torch.cat([d4, x1], dim=1)  # [B, 16+16=32, H/2, W/2]
        if self.debug:
            print(f"d4 after concat: {d4.shape}")
        
        reconstruction = self.decoder5(d4)  # [B, output_channels, H, W]
        if self.debug:
            print(f"reconstruction shape: {reconstruction.shape}")
        
        return reconstruction

# Create a new decoder class without skip connections
class MobileNetV3DecoderNoSkip(nn.Module):
    def __init__(self, output_channels=3):
        super(MobileNetV3DecoderNoSkip, self).__init__()
        
        # Since x6 is now the same spatial dimensions as x5, we don't need upsampling
        # But we need to start with a richer feature representation since we don't have skip connections
        self.decoder0 = nn.Sequential(
            nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(576),
            nn.ReLU(inplace=True)
        )
        
        # Decoder block 1: [B, 576, H/32, W/32] -> [B, 96, H/16, W/16]
        # No concatenation, so input channels reduced
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(576, 96, kernel_size=2, stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        
        # Decoder block 2: [B, 96, H/16, W/16] -> [B, 48, H/8, W/8]
        # No concatenation, so input channels reduced
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(96, 48, kernel_size=2, stride=2),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Decoder block 3: [B, 48, H/8, W/8] -> [B, 24, H/4, W/4]
        # No concatenation, so input channels reduced
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(48, 24, kernel_size=2, stride=2),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        
        # Decoder block 4: [B, 24, H/4, W/4] -> [B, 16, H/2, W/2]
        # No concatenation, so input channels reduced
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(24, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # Final upsampling: [B, 16, H/2, W/2] -> [B, output_channels, H, W]
        # No concatenation, so input channels reduced
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose2d(16, output_channels, kernel_size=2, stride=2),
            nn.Sigmoid()  # Sigmoid for output in range [0,1]
        )
        
        # Add debug flag
        self.debug = False
        
    def forward(self, x1, x2, x3, x4, x5, x6):
        # Only using x6, ignoring x1-x5 (no skip connections)
        if self.debug:
            print(f"Input shape x6: {x6.shape}")
        
        # Decoder path without skip connections
        d0 = self.decoder0(x6)  # [B, 576, H/32, W/32]
        if self.debug:
            print(f"d0 shape: {d0.shape}")
        
        d1 = self.decoder1(d0)  # [B, 96, H/16, W/16]
        if self.debug:
            print(f"d1 shape: {d1.shape}")
        
        d2 = self.decoder2(d1)  # [B, 48, H/8, W/8]
        if self.debug:
            print(f"d2 shape: {d2.shape}")
        
        d3 = self.decoder3(d2)  # [B, 24, H/4, W/4]
        if self.debug:
            print(f"d3 shape: {d3.shape}")
        
        d4 = self.decoder4(d3)  # [B, 16, H/2, W/2]
        if self.debug:
            print(f"d4 shape: {d4.shape}")
        
        reconstruction = self.decoder5(d4)  # [B, output_channels, H, W]
        if self.debug:
            print(f"reconstruction shape: {reconstruction.shape}")
        
        return reconstruction

# Create a Feature-Guided decoder without skip connections but with feature similarity losses
class MobileNetV3DecoderFeatureGuided(nn.Module):
    def __init__(self, output_channels=3):
        super(MobileNetV3DecoderFeatureGuided, self).__init__()
        
        # First layer to process bottleneck features
        self.decoder0 = nn.Sequential(
            nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(576),
            nn.ReLU(inplace=True)
        )
        
        # Upsampling layers without skip connections
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(576, 96, kernel_size=2, stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(96, 48, kernel_size=2, stride=2),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(48, 24, kernel_size=2, stride=2),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(24, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose2d(16, output_channels, kernel_size=2, stride=2),
            nn.Sigmoid()
        )
        
        # Intermediate feature projection layers for loss calculation
        self.project1 = nn.Conv2d(96, 96, kernel_size=1)    # Project d1 to match x4's channel count
        self.project2 = nn.Conv2d(48, 48, kernel_size=1)    # Project d2 to match x3's channel count
        self.project3 = nn.Conv2d(24, 24, kernel_size=1)    # Project d3 to match x2's channel count
        self.project4 = nn.Conv2d(16, 16, kernel_size=1)    # Project d4 to match x1's channel count
        
        self.debug = False
    
    def forward(self, x1, x2, x3, x4, x5, x6):
        # Store intermediate feature maps for loss calculation
        decoder_features = {}
        
        # Only use x6 as input, other encoder features are only used for loss calculation
        if self.debug:
            print(f"Input shape x6: {x6.shape}")
        
        # Decoder path
        d0 = self.decoder0(x6)
        if self.debug:
            print(f"d0 shape: {d0.shape}")
        
        d1 = self.decoder1(d0)
        if self.debug:
            print(f"d1 shape: {d1.shape}")
        decoder_features['d1'] = self.project1(d1)  # Store for loss
        
        d2 = self.decoder2(d1)
        if self.debug:
            print(f"d2 shape: {d2.shape}")
        decoder_features['d2'] = self.project2(d2)  # Store for loss
        
        d3 = self.decoder3(d2)
        if self.debug:
            print(f"d3 shape: {d3.shape}")
        decoder_features['d3'] = self.project3(d3)  # Store for loss
        
        d4 = self.decoder4(d3)
        if self.debug:
            print(f"d4 shape: {d4.shape}")
        decoder_features['d4'] = self.project4(d4)  # Store for loss
        
        reconstruction = self.decoder5(d4)
        if self.debug:
            print(f"reconstruction shape: {reconstruction.shape}")
        
        return reconstruction, decoder_features

# Wrapper class to maintain backward compatibility
class MobileNetV3UNetDeep(nn.Module):
    def __init__(self, num_classes=5, output_channels=3):
        super(MobileNetV3UNetDeep, self).__init__()
        self.encoder = MobileNetV3Encoder(num_classes=num_classes)
        self.decoder = MobileNetV3Decoder(output_channels=output_channels)
        
    def forward(self, x):
        out1, out2, features, x1, x2, x3, x4, x5, x6 = self.encoder(x)
        reconstruction = self.decoder(x1, x2, x3, x4, x5, x6)
        return out1, out2, features, reconstruction

# Create an alternative UNet model without skip connections
class MobileNetV3UNetNoSkip(nn.Module):
    def __init__(self, num_classes=5, output_channels=3):
        super(MobileNetV3UNetNoSkip, self).__init__()
        self.encoder = MobileNetV3Encoder(num_classes=num_classes)
        self.decoder = MobileNetV3DecoderNoSkip(output_channels=output_channels)
        
    def forward(self, x):
        out1, out2, features, x1, x2, x3, x4, x5, x6 = self.encoder(x)
        reconstruction = self.decoder(x1, x2, x3, x4, x5, x6)
        return out1, out2, features, reconstruction

# Create a UNet model with feature similarity loss instead of skip connections
class MobileNetV3UNetFeatureLoss(nn.Module):
    def __init__(self, num_classes=5, output_channels=3):
        super(MobileNetV3UNetFeatureLoss, self).__init__()
        self.encoder = MobileNetV3Encoder(num_classes=num_classes)
        self.decoder = MobileNetV3DecoderFeatureGuided(output_channels=output_channels)
        
    def forward(self, x):
        # Run encoder
        out1, out2, features, x1, x2, x3, x4, x5, x6 = self.encoder(x)
        
        # Run decoder (returns reconstruction and intermediate features)
        reconstruction, decoder_features = self.decoder(x1, x2, x3, x4, x5, x6)
        
        # Store encoder features for loss calculation
        encoder_features = {
            'x4': x4,  # To compare with d1
            'x3': x3,  # To compare with d2
            'x2': x2,  # To compare with d3
            'x1': x1   # To compare with d4
        }
        
        return out1, out2, features, reconstruction, encoder_features, decoder_features

# Feature similarity loss function
class FeatureSimilarityLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(FeatureSimilarityLoss, self).__init__()
        self.alpha = alpha  # Weight for MSE vs feature similarity loss
        self.mse = nn.MSELoss()
    
    def forward(self, reconstruction, target, encoder_features, decoder_features):
        # Reconstruction loss (MSE)
        rec_loss = self.mse(reconstruction, target)
        
        # Feature similarity losses
        feature_loss = 0.0
        
        # For each level, calculate feature similarity loss
        # We need to handle potential size mismatches
        for d_key, e_key in [('d1', 'x4'), ('d2', 'x3'), ('d3', 'x2'), ('d4', 'x1')]:
            d_feat = decoder_features[d_key]
            e_feat = encoder_features[e_key]
            
            # Ensure spatial dimensions match
            if d_feat.shape[2:] != e_feat.shape[2:]:
                d_feat = nn.functional.interpolate(
                    d_feat, size=e_feat.shape[2:], 
                    mode='bilinear', align_corners=False
                )
            
            # Calculate L2 loss between feature maps
            level_loss = self.mse(d_feat, e_feat)
            feature_loss += level_loss
        
        # Combine losses
        total_loss = self.alpha * rec_loss + (1 - self.alpha) * feature_loss
        
        return total_loss, {
            'rec_loss': rec_loss.item(),
            'feature_loss': feature_loss.item(),
            'total_loss': total_loss.item()
        }

# Create and display the model
model = MobileNetV3UNetDeep(num_classes=5, output_channels=3)
print(model)

# 输出网络结构
model = MultiHeadMobileNetV3(num_classes=5)
print(model)

# Improved testing function to capture the actual dimensions
def test_model_dimensions():
    # Create a small test input
    x = torch.randn(1, 3, 224, 224)
    model = MobileNetV3UNetDeep(num_classes=5, output_channels=3)
    
    # Enable debug mode to print shapes
    model.decoder.debug = True
    
    # Get encoder features to analyze directly
    with torch.no_grad():
        # Run forward pass on encoder
        out1, out2, features, x1, x2, x3, x4, x5, x6 = model.encoder(x)
        print("==== Encoder Output Dimensions ====")
        print(f"x1 channels: {x1.size(1)}, shape: {x1.shape}")
        print(f"x2 channels: {x2.size(1)}, shape: {x2.shape}")
        print(f"x3 channels: {x3.size(1)}, shape: {x3.shape}")
        print(f"x4 channels: {x4.size(1)}, shape: {x4.shape}")
        print(f"x5 channels: {x5.size(1)}, shape: {x5.shape}")
        print(f"x6 channels: {x6.size(1)}, shape: {x6.shape}")
        print("==================================")
        
        # Run full model
        out1, out2, features, reconstruction = model(x)
        print(f"Reconstruction shape: {reconstruction.shape}")
    
    print("Model executed successfully")
    return model

# Add a debug function to specifically test the encoder-decoder dimensions
def debug_spatial_dimensions():
    x = torch.randn(1, 3, 224, 224)
    model = MobileNetV3UNetDeep(num_classes=5, output_channels=3)
    
    # Enable debug mode in decoder
    model.decoder.debug = True
    
    with torch.no_grad():
        try:
            # First extract encoder features to check them
            out1, out2, features, x1, x2, x3, x4, x5, x6 = model.encoder(x)
            
            print("\nEncoder Feature Map Shapes:")
            print(f"x1: {x1.shape}")
            print(f"x2: {x2.shape}")
            print(f"x3: {x3.shape}")
            print(f"x4: {x4.shape}")
            print(f"x5: {x5.shape}")
            print(f"x6: {x6.shape}")
            
            # Check if the spatial dimensions are factors of 2
            for i, feat in enumerate([x1, x2, x3, x4, x5, x6]):
                h, w = feat.shape[2], feat.shape[3]
                print(f"x{i+1} spatial dim: {h}×{w} - Symmetric: {h==w}")
            
            # Run full model with debug output
            print("\nRunning full forward pass with debug:")
            out = model(x)
            print("\n✓ Model forward pass successful")
            
        except Exception as e:
            print(f"\n✗ Error during execution: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Disable debug mode after running
    model.decoder.debug = False
    return model

# Function to compare models with and without skip connections
def compare_skip_vs_noskip():
    # Test sample
    x = torch.randn(1, 3, 224, 224)
    
    # Create both models
    model_skip = MobileNetV3UNetDeep(num_classes=5, output_channels=3)
    model_noskip = MobileNetV3UNetNoSkip(num_classes=5, output_channels=3)
    
    # Set to evaluation mode
    model_skip.eval()
    model_noskip.eval()
    
    print("\n====== Comparing UNet with and without skip connections ======")
    
    with torch.no_grad():
        # Forward pass for both models
        out1_skip, out2_skip, feat_skip, recon_skip = model_skip(x)
        out1_noskip, out2_noskip, feat_noskip, recon_noskip = model_noskip(x)
        
        print("\nOutput shapes:")
        print(f"With skip connections: {recon_skip.shape}")
        print(f"Without skip connections: {recon_noskip.shape}")
        
        # Compare reconstructions (just to demonstrate difference)
        print("\nReconstruction statistics:")
        print(f"With skip - min: {recon_skip.min().item():.4f}, max: {recon_skip.max().item():.4f}, mean: {recon_skip.mean().item():.4f}")
        print(f"No skip - min: {recon_noskip.min().item():.4f}, max: {recon_noskip.max().item():.4f}, mean: {recon_noskip.mean().item():.4f}")
        
        print("\nExpected differences:")
        print("1. 特征保留: 带跳跃连接的模型可以更好地保留多尺度特征")
        print("2. 高频信息: 没有跳跃连接的模型会丢失更多的高频细节")
        print("3. 训练难度: 没有跳跃连接的模型训练难度更大，容易出现梯度消失问题")
        print("4. 重构准确性: 没有跳跃连接的模型在精细纹理重建上表现会更差")
        print("5. 优化收敛: 没有跳跃连接的模型优化收敛速度更慢")
    
    return model_skip, model_noskip

# Function to compare the original UNet with the feature-guided UNet
def compare_unet_variants():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create a sample input and target
    x = torch.randn(1, 3, 224, 224)
    target = torch.rand(1, 3, 224, 224)  # Random target image
    
    # Create models
    model_skip = MobileNetV3UNetDeep(num_classes=5, output_channels=3)
    model_noskip = MobileNetV3UNetNoSkip(num_classes=5, output_channels=3)
    model_featureloss = MobileNetV3UNetFeatureLoss(num_classes=5, output_channels=3)
    
    # Set to evaluation mode
    model_skip.eval()
    model_noskip.eval()
    model_featureloss.eval()
    
    # Create loss function for feature-guided model
    feature_loss_fn = FeatureSimilarityLoss(alpha=0.5)
    
    print("\n======== Comparing UNet Variants ========")
    
    with torch.no_grad():
        # Forward pass for all models
        out1_skip, out2_skip, feat_skip, recon_skip = model_skip(x)
        out1_noskip, out2_noskip, feat_noskip, recon_noskip = model_noskip(x)
        out1_fl, out2_fl, feat_fl, recon_fl, encoder_features, decoder_features = model_featureloss(x)
        
        # Calculate feature similarity loss
        total_loss, loss_components = feature_loss_fn(recon_fl, target, encoder_features, decoder_features)
        
        print("\n模型输出形状:")
        print(f"传统UNet (有跳跃连接): {recon_skip.shape}")
        print(f"无跳跃连接UNet: {recon_noskip.shape}")
        print(f"特征引导UNet (无跳跃连接但有特征相似度损失): {recon_fl.shape}")
        
        print("\n特征引导UNet损失组成:")
        print(f"重建损失 (MSE): {loss_components['rec_loss']:.4f}")
        print(f"特征相似度损失: {loss_components['feature_loss']:.4f}")
        print(f"总损失: {loss_components['total_loss']:.4f}")
        
        print("\n三种方法的比较分析:")
        print("1. 传统UNet (有跳跃连接):")
        print("   - 直接在解码器中使用编码器特征")
        print("   - 可以更好地保留空间细节")
        print("   - 结构简单，训练稳定")
        
        print("\n2. 无跳跃连接UNet:")
        print("   - 解码器完全依赖于瓶颈特征")
        print("   - 难以恢复细节信息")
        print("   - 在细节重建上表现较差")
        
        print("\n3. 特征引导UNet (特征相似度损失):")
        print("   - 无直接跳跃连接，但通过损失函数引导解码器学习相似特征")
        print("   - 结构上与无跳跃连接UNet相同，但训练目标不同")
        print("   - 可能比无跳跃连接模型表现更好，但仍可能不如直接跳跃连接")
        print("   - 特点: 解码器在结构上独立，但训练过程中会学习到与编码器相似的特征表示")
        print("   - 优势: 更灵活的架构设计，可以单独调整特征相似度的权重影响")
        print("   - 适用场景: 当需要解码器有更多独立性但又不想完全丢失细节时")
    
    return model_skip, model_noskip, model_featureloss

# Run the debug function
debug_model = debug_spatial_dimensions()

# Run the comparative analysis
# skip_model, noskip_model = compare_skip_vs_noskip()

# Uncomment to test on different image sizes to verify size adaptability
# test_sizes = [(224, 224), (256, 256), (300, 400), (512, 512)]
# for size in test_sizes:
#     print(f"\nTesting size: {size}")
#     x = torch.randn(1, 3, *size)
#     try:
#         with torch.no_grad():
#             out = model(x)
#         print(f"✓ Success with input size {size}")
#     except Exception as e:
#         print(f"✗ Failed with input size {size}: {str(e)}")

# 改进版的特征引导解码器
class MobileNetV3DecoderFeatureGuidedPlus(nn.Module):
    def __init__(self, output_channels=3):
        super(MobileNetV3DecoderFeatureGuidedPlus, self).__init__()
        
        # 增加通道后再处理 - 给解码器更强的表达能力
        self.decoder0 = nn.Sequential(
            nn.Conv2d(96, 768, kernel_size=1, stride=1, padding=0),  # 增大通道数
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            nn.Conv2d(768, 576, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(576),
            nn.ReLU(inplace=True)
        )
        
        # 增强每个解码器块的能力 - 使用残差连接
        # Decoder block 1: [B, 576, H/32, W/32] -> [B, 96, H/16, W/16]
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(576, 192, kernel_size=2, stride=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        
        # Decoder block 2: [B, 96, H/16, W/16] -> [B, 48, H/8, W/8]
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(96, 96, kernel_size=2, stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 48, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Decoder block 3: [B, 48, H/8, W/8] -> [B, 24, H/4, W/4]
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(48, 48, kernel_size=2, stride=2),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        
        # Decoder block 4: [B, 24, H/4, W/4] -> [B, 16, H/2, W/2]
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(24, 24, kernel_size=2, stride=2),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # Final upsampling with extra refinement
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, output_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
        # 投影层 - 更复杂的投影层以更好地对齐特征
        self.project1 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=1)
        )
        
        self.project2 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=1)
        )
        
        self.project3 = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, kernel_size=1)
        )
        
        self.project4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1)
        )
        
        # 添加注意力机制 - 帮助解码器关注重要特征
        self.attention1 = ChannelAttention(96)
        self.attention2 = ChannelAttention(48)
        self.attention3 = ChannelAttention(24)
        self.attention4 = ChannelAttention(16)
        
        self.debug = False
    
    def forward(self, x1, x2, x3, x4, x5, x6):
        decoder_features = {}
        
        # 开始解码
        d0 = self.decoder0(x6)
        
        d1 = self.decoder1(d0)
        d1 = self.attention1(d1)  # 应用通道注意力
        decoder_features['d1'] = self.project1(d1)
        
        d2 = self.decoder2(d1)
        d2 = self.attention2(d2)
        decoder_features['d2'] = self.project2(d2)
        
        d3 = self.decoder3(d2)
        d3 = self.attention3(d3)
        decoder_features['d3'] = self.project3(d3)
        
        d4 = self.decoder4(d3)
        d4 = self.attention4(d4)
        decoder_features['d4'] = self.project4(d4)
        
        reconstruction = self.decoder5(d4)
        
        return reconstruction, decoder_features

# 通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 共享MLP
        reduced_channels = max(8, in_channels // reduction_ratio)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out

# 改进的UNet模型
class MobileNetV3UNetFeatureLossPlus(nn.Module):
    def __init__(self, num_classes=5, output_channels=3):
        super(MobileNetV3UNetFeatureLossPlus, self).__init__()
        self.encoder = MobileNetV3Encoder(num_classes=num_classes)
        self.decoder = MobileNetV3DecoderFeatureGuidedPlus(output_channels=output_channels)
        
    def forward(self, x):
        out1, out2, features, x1, x2, x3, x4, x5, x6 = self.encoder(x)
        reconstruction, decoder_features = self.decoder(x1, x2, x3, x4, x5, x6)
        
        encoder_features = {
            'x4': x4,
            'x3': x3,
            'x2': x2,
            'x1': x1
        }
        
        return out1, out2, features, reconstruction, encoder_features, decoder_features

# 感知损失的特征提取器
class PerceptualFeatureExtractor(nn.Module):
    def __init__(self):
        super(PerceptualFeatureExtractor, self).__init__()
        # 加载预训练的VGG模型
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        # 只使用前面几个卷积块
        self.features = nn.Sequential(*list(vgg16.features.children())[:23])
        
        # 冻结所有参数
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        # 收集不同层次的特征
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            # 保存一些中间特征
            if i in [3, 8, 15, 22]:  # conv1_2, conv2_2, conv3_3, conv4_3
                features.append(x)
        return features

# 感知损失 + 风格损失 + 特征相似度损失
class EnhancedFeatureLoss(nn.Module):
    def __init__(self, content_weight=1.0, style_weight=0.05, feature_weight=0.2, 
                consistency_weight=0.3, alpha=0.5):
        super(EnhancedFeatureLoss, self).__init__()
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.feature_weight = feature_weight
        self.consistency_weight = consistency_weight
        self.alpha = alpha  # MSE vs 特征损失的权重
        
        self.perceptual = PerceptualFeatureExtractor()
        self.mse = nn.MSELoss()
        
    def gram_matrix(self, x):
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram
        
    def forward(self, reconstruction, target, encoder_features, decoder_features):
        # 基础MSE重建损失
        rec_loss = self.mse(reconstruction, target)
        
        # 编码器-解码器特征对齐损失
        feature_loss = 0.0
        for d_key, e_key in [('d1', 'x4'), ('d2', 'x3'), ('d3', 'x2'), ('d4', 'x1')]:
            d_feat = decoder_features[d_key]
            e_feat = encoder_features[e_key]
            
            if d_feat.shape[2:] != e_feat.shape[2:]:
                d_feat = nn.functional.interpolate(
                    d_feat, size=e_feat.shape[2:], 
                    mode='bilinear', align_corners=False
                )
                
            # L2损失 + 余弦相似度损失（确保方向一致性）
            l2_loss = self.mse(d_feat, e_feat)
            cos_sim = torch.nn.functional.cosine_similarity(
                d_feat.flatten(2), e_feat.flatten(2), dim=2
            ).mean()
            feature_loss += l2_loss - 0.1 * cos_sim  # 希望最大化余弦相似度
        
        # 感知损失 (VGG特征)
        # 归一化到VGG期望的输入范围
        target_vgg = target * 2.0 - 1.0
        recon_vgg = reconstruction * 2.0 - 1.0
        
        if target_vgg.shape[1] == 1:  # 如果是灰度图
            target_vgg = target_vgg.repeat(1, 3, 1, 1)
            recon_vgg = recon_vgg.repeat(1, 3, 1, 1)
            
        target_features = self.perceptual(target_vgg)
        recon_features = self.perceptual(recon_vgg)
        
        content_loss = 0
        style_loss = 0
        
        # 内容损失和风格损失
        for tf, rf in zip(target_features, recon_features):
            # 内容损失 - 特征图相似度
            content_loss += self.mse(rf, tf)
            
            # 风格损失 - Gram矩阵相似度
            style_loss += self.mse(self.gram_matrix(rf), self.gram_matrix(tf))
        
        # 总损失
        total_loss = self.alpha * rec_loss + \
                    (1 - self.alpha) * (
                        self.feature_weight * feature_loss + 
                        self.content_weight * content_loss + 
                        self.style_weight * style_loss
                    )
        
        loss_details = {
            'rec_loss': rec_loss.item(),
            'feature_loss': feature_loss.item(),
            'content_loss': content_loss.item(),
            'style_loss': style_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_details