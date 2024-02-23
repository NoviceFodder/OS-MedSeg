from typing import Tuple, Any, List
import os
import torch
import torch.nn as nn
# MONAI imports
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from torch import Tensor

class UNet_distill(nn.Module):
    def __init__(
            self,
            in_channel: int,
            num_class: int,
            channel_list: List[int],
            residual: bool,
            device: str
    ):
        super().__init__()
        valid_devices = {'cuda', 'cpu'}
        if device not in valid_devices:
            raise ValueError(f"Invalid value for device. Expected one of {valid_devices}, but got {device}")

        self.residual = residual
        self.encode_blocks = []
        self.decode_blocks = []
        self.encode_outputs = []
        self.down_sample = nn.MaxPool3d(kernel_size=2, stride=2, padding=0).to(device)
        self.up_sample = nn.Upsample(scale_factor=2, mode='trilinear').to(device)

        self.add_module('down_sample_layer', self.down_sample)
        self.add_module('up_sample_layer', self.up_sample)

        # Build encoder
        self.encode_blocks.append(
            nn.Sequential(
                Convolution(
                    spatial_dims=3,
                    in_channels=in_channel,
                    out_channels=channel_list[0],
                    kernel_size=3,
                    strides=1
                ).to(device),

                Convolution(
                    spatial_dims=3,
                    in_channels=channel_list[0],
                    out_channels=channel_list[0],
                    kernel_size=3,
                    strides=1
                ).to(device),

            )
        )

        for i in range(0, len(channel_list) - 1):
            if i < len(channel_list) - 2:
                self.encode_blocks.append(
                    nn.Sequential(
                        Convolution(
                            spatial_dims=3,
                            in_channels=channel_list[i],
                            out_channels=channel_list[i + 1],
                            kernel_size=3,
                            strides=1
                        ).to(device),

                        Convolution(
                            spatial_dims=3,
                            in_channels=channel_list[i + 1],
                            out_channels=channel_list[i + 1],
                            kernel_size=3,
                            strides=1
                        ).to(device)
                    )
                )
            else:
                # Last layer(bottle neck): channel: 256->512->256
                self.encode_blocks.append(
                    nn.Sequential(
                        Convolution(
                            spatial_dims=3,
                            in_channels=channel_list[i],
                            out_channels=channel_list[i + 1],
                            kernel_size=3,
                            strides=1
                        ).to(device)
                    )
                )

        for i, block in enumerate(self.encode_blocks):
            self.add_module('encode_block{0}'.format(i), block)

        # Build decoder
        self.decode_blocks.append(
            nn.Sequential(
                Convolution(
                    spatial_dims=3,
                    in_channels=channel_list[-1],
                    out_channels=channel_list[-2],
                    kernel_size=3,
                    strides=2,
                    is_transposed=True
                ).to(device)
            )
        )
        for i in range(len(channel_list) - 1, 0, -1):
            self.decode_blocks.append(
                nn.Sequential(
                    Convolution(
                        spatial_dims=3,
                        in_channels=channel_list[i],
                        out_channels=channel_list[i - 1],
                        kernel_size=3,
                        strides=1,
                    ).to(device),
                    # Final block do not up-sample, if i - 1 == 0, then reach the final block
                    Convolution(
                        spatial_dims=3,
                        in_channels=channel_list[i - 1],
                        out_channels=channel_list[i - 2] if i - 1 > 0 else channel_list[0],
                        kernel_size=3,
                        strides=2 if i - 1 > 0 else 1,
                        is_transposed=True if i - 1 > 0 else False
                    ).to(device)
                )
            )

        for i, block in enumerate(self.decode_blocks):
            self.add_module('decode_block{0}'.format(i), block)

        # Segmentation head
        self.seg_head = nn.Sequential(
            Convolution(
                spatial_dims=3,
                in_channels=channel_list[0],
                out_channels=channel_list[0],
                kernel_size=3,
                strides=1
            ).to(device),

            nn.Conv3d(
                in_channels=channel_list[0],
                out_channels=num_class,
                kernel_size=1,
                stride=1,
                padding=0

            ).to(device)
        )

    def forward(self, x) -> Tensor:
        self.encode_outputs = []

        for i, block in enumerate(self.encode_blocks):
            if i < len(self.encode_blocks) - 1:
                x = block(x)
                self.encode_outputs.append(x)
                # print(x.shape)
                x = self.down_sample(x)
            else:
                # Last layer in encoder, do not downsample
                x = block(x)
                # print(x.shape)

        self.encode_outputs.reverse()
        last_two_layer_outputs = []
        for i, block in enumerate(self.decode_blocks):
            if i == 0:
                x = block(x)
            else:
                # 保存倒数第二层的特征
                if i == len(self.decode_blocks) - 2:
                    last_two_layer_outputs.append(x.clone().detach())
                
                # print(x.shape, self.encode_outputs[i - 1].shape)
                x = torch.concat([self.encode_outputs[i - 1], x], dim=1)
                x = block(x)

        # 保存最后一层的特征
        last_two_layer_outputs.append(x.clone().detach())
        pre_seg = self.seg_head(x)

        return pre_seg,last_two_layer_outputs

class teacher_student_model(nn.Module):
    def __init__(
            self,
            in_channel: int,
            num_class: int,
            channel_list: List[int],
            residual: bool,
            vae: bool,
            device: str
    ):
        super().__init__()
        '''
        self.teacher_model = recon_vae_unet(
            in_channel=in_channel,
            num_class=num_class,
            channel_list=channel_list,
            residual=residual,
            vae = vae,
            device=device
        )
        '''
        self.student_model = UNet_distill(
            in_channel=in_channel,
            num_class=num_class,
            channel_list=channel_list,
            residual=residual,
            device=device
        )
        
    def forward(self, x, y, is_Training=False) -> Tensor:
        '''
        Parameters:
            x: Image for student model -> tensor.
            y: Image for teacher model -> tensor.
        '''   
        '''
        if is_Training:
            recon,teacher_features = self.teacher_model(y)
            student_pred_label,student_features = self.student_model(x) 
            return teacher_features,recon,student_pred_label,student_features  
        '''
        
        student_pred_label,student_features = self.student_model(x) 
        return student_pred_label,student_features  






