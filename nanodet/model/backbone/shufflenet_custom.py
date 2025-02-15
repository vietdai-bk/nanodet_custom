import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from ..module.activation import act_layers

model_urls = {
    "shufflenetv2_0.25x": None,  # Tự train từ đầu
    "shufflenetv2_0.5x": "https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth",  # noqa: E501
    "shufflenetv2_1.0x": "https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth",  # noqa: E501
    "shufflenetv2_1.5x": "https://download.pytorch.org/models/shufflenetv2_x1_5-3c479a10.pth",  # noqa: E501
    "shufflenetv2_2.0x": "https://download.pytorch.org/models/shufflenetv2_x2_0-8be3c8ee.pth",  # noqa: E501
}

def channel_shuffle(x, groups):
    # Tối ưu hóa bằng reshape + transpose thay vì view + transpose
    batchsize, num_channels, height, width = x.size()
    x = x.reshape(batchsize, groups, num_channels // groups, height, width)
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    return x.reshape(batchsize, num_channels, height, width)

class SpeedV2Block(nn.Module):
    """Phiên bản tối giản của ShuffleV2Block, tập trung vào tốc độ"""
    def __init__(self, inp, oup, stride):
        super().__init__()
        self.stride = stride
        branch_features = oup // 2

        if self.stride > 1:
            # Nhánh 1: Depthwise -> Pointwise
            self.branch1 = nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, 1, 1, 0, bias=False),
                nn.BatchNorm2d(branch_features),
            )
        else:
            self.branch1 = nn.Identity()

        # Nhánh 2 tối giản
        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if stride > 1 else branch_features, branch_features, 1, 1, 0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_features, branch_features, 3, stride, 1, groups=branch_features, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat([x1, self.branch2(x2)], dim=1)
        else:
            out = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        return channel_shuffle(out, 2)

class UltraFastShuffleNetV2(nn.Module):
    def __init__(
        self,
        model_size="0.25x",
        out_stages=(2, 3, 4),
        with_last_conv=False,
        activation="ReLU",
        pretrain=False,
    ):
        super().__init__()
        assert set(out_stages).issubset((2, 3, 4))
        self.out_stages = out_stages
        
        stage_configs = {
            "0.25x": {
                "repeats": [2, 4, 2],
                "channels": [16, 32, 64, 128, 512]
            },
            "0.5x": {
                "repeats": [4, 8, 4],
                "channels": [24, 48, 96, 192, 1024]
            },
            "1.0x": {
                "repeats": [4, 8, 4],
                "channels": [24, 116, 232, 464, 1024]
            },
            "1.5x": {
                "repeats": [4, 8, 4],
                "channels": [24, 176, 352, 704, 1024]
            },
            "2.0x": {
                "repeats": [4, 8, 4],
                "channels": [24, 244, 488, 976, 2048]
            }
        }
        
        config = stage_configs[model_size]
        self.stage_repeats = config["repeats"]
        self._stage_out_channels = config["channels"]
        
        # Tối ưu lớp đầu vào
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self._stage_out_channels[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(self._stage_out_channels[0]),
            nn.ReLU(inplace=True),
        )
        
        # Thay thế MaxPool bằng Conv stride=2
        self.conv2 = nn.Sequential(
            nn.Conv2d(self._stage_out_channels[0], self._stage_out_channels[0], 
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self._stage_out_channels[0]),
            nn.ReLU(inplace=True),
        )

        # Xây dựng các stage với SpeedV2Block
        input_channels = self._stage_out_channels[0]
        for stage_idx in [2, 3, 4]:
            seq = []
            output_channels = self._stage_out_channels[stage_idx-1]
            for i in range(self.stage_repeats[stage_idx-2]):
                stride = 2 if i == 0 else 1
                seq.append(SpeedV2Block(input_channels, output_channels, stride))
                input_channels = output_channels
            setattr(self, f"stage{stage_idx}", nn.Sequential(*seq))
        
        # Tối ưu hóa lớp cuối
        if with_last_conv:
            self.final_conv = nn.Conv2d(input_channels, self._stage_out_channels[-1], 1)
        else:
            self.final_conv = nn.Identity()
        
        self._initialize_weights(pretrain and model_size != "0.25x")

    def _initialize_weights(self, pretrain):
        # Khởi tạo đơn giản hóa
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        if pretrain:
            # Tải pretrain chỉ cho các model size được hỗ trợ
            if self.model_size in model_urls:
                state_dict = model_zoo.load_url(model_urls[f"shufflenetv2_{self.model_size}"])
                self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)  # Thay thế maxpool
        outputs = []
        for stage_idx in [2, 3, 4]:
            x = getattr(self, f"stage{stage_idx}")(x)
            if stage_idx in self.out_stages:
                outputs.append(x)
        x = self.final_conv(x)
        return tuple(outputs)