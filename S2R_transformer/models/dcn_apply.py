import torch
import torch.nn as nn
from mmdetection.mmdet.ops.dcn.deform_conv import DeformConv

def normal_init(module: nn.Module,
                mean: float = 0,
                std: float = 1,
                bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class Dcn_apply(nn.Module):

    def __init__(self,in_channels, out_channels, kernel_size=3, deformable_groups=4):
        super(Dcn_apply, self).__init__()
        offset_channels = kernel_size * kernel_size * 2
        self.conv_offset = nn.Conv2d(4, deformable_groups * offset_channels, 1, bias=False)
        self.conv_adaption = DeformConv(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2, deformable_groups=deformable_groups)
        self.relu = nn.ReLU(inplace = True)
    
    def init_weights(self):
        normal_init(self.conv_offset, std=0.1)
        normal_init(self.conv_adaption, std=0.01)
    
    def forward(self, x, shape):
        offset = self.conv_offset(shape)
        x = self.relu(self.conv_adaption(x, offset))
        return x