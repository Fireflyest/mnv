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

        # 反注意力，加强背景特征
        spatial_attention = torch.mean(x5, dim=1, keepdim=True)  # [B, 1, H/32, W/32]
        spatial_attention = torch.sigmoid(spatial_attention)
        x5 = x5 * (1 - spatial_attention)  # 逐元素相乘，广播机制会自动扩展

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

# Run the debug function
debug_model = debug_spatial_dimensions()

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