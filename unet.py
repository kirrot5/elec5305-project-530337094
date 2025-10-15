"""
U-Net model for vocal separation
Author: Kiro Chen
Reference: Jansson et al., 2017
"""

import torch
import torch.nn as nn

class UNet(nn.Module):
    """U-Net architecture for source separation"""

    def __init__(self, input_channels=1, output_channels=1, base_filters=16):
        """
        Args:
            input_channels: Number of input channels
            output_channels: Number of output channels
            base_filters: Base number of filters
        """
        super(UNet, self).__init__()

        # Encoder (downsampling path)
        self.enc1 = self._conv_block(input_channels, base_filters)
        self.enc2 = self._conv_block(base_filters, base_filters * 2)
        self.enc3 = self._conv_block(base_filters * 2, base_filters * 4)
        self.enc4 = self._conv_block(base_filters * 4, base_filters * 8)

        # Bottleneck
        self.bottleneck = self._conv_block(base_filters * 8, base_filters * 16)

        # Decoder (upsampling path)
        self.dec4 = self._upconv_block(base_filters * 16, base_filters * 8)
        self.dec3 = self._upconv_block(base_filters * 8, base_filters * 4)
        self.dec2 = self._upconv_block(base_filters * 4, base_filters * 2)
        self.dec1 = self._upconv_block(base_filters * 2, base_filters)

        # Output layer
        self.output = nn.Conv2d(base_filters, output_channels, kernel_size=1)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def _conv_block(self, in_channels, out_channels):
        """Convolutional block: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _upconv_block(self, in_channels, out_channels):
        """Upsampling block: ConvTranspose -> Conv -> BatchNorm -> ReLU"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor (batch, channels, height, width)

        Returns:
            output: Output tensor (same shape as input)
        """
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoder with skip connections
        dec4 = self.dec4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)  # Skip connection

        dec3 = self.dec3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)

        dec2 = self.dec2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)

        dec1 = self.dec1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)

        # Output
        output = self.output(dec1)

        return output


# Test model
if __name__ == "__main__":
    model = UNet(input_channels=1, output_channels=1, base_filters=16)

    # Test with random input
    x = torch.randn(1, 1, 256, 256)  # (batch, channels, height, width)
    output = model(x)

    print("U-Net Model Test:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
```

---

## 项目结构

创建这个文件结构：
```
elec5305-project-530337094/
│
├── README.md
├── proposal.pdf
├── requirements.txt
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── baseline_separator.py
│   ├── evaluation.py
│   ├── visualize.py
│   └── models/
│       ├── __init__.py
│       └── unet.py
│
├── experiments/
│   ├── baseline_experiment.py
│   └── results/
│       └── .gitkeep
│
├── notebooks/
│   └── exploration.ipynb  (可选)
│
└── data/
    └── README.md  (说明如何获取MUSDB18)