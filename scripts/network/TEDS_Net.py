import numpy as np
import torch.nn as nn 
import torch.nn.functional as F
from torch.distributions.normal import Normal
import torchvision.transforms.functional as FF

from network.UNet import ConvBlock,EncoderBranch,DecoderBranch,BottleNeck
from network.utils_teds import WholeDiffeoUnit

class TEDS_Net(nn.Module):
    '''
    TEDS-Net:

    Input is the parameter describing dictionary.

    '''

    def __init__(self,params):
        
        
        super(TEDS_Net, self).__init__()


        # Parameters settings - arch:
        in_channels = params.network_params.in_chan
        out_channels =params.network_params.out_chan
        features = params.network_params.fi
        net_depth=params.network_params.net_depth
        dropout = params.network_params.dropout
        dec_depth = params.network.dec_depth
        self.no_branches = len(dec_depth)
        
        # Parameters settings - diffeomorphic:
        int_steps = params.network.diffeo_int
        GSmooth =params.network.guas_smooth
        Guas_kernel = params.network.Guas_kernel
        Guas_P = params.network.sigma
        act = params.network.act
        self.mega_P = params.network.mega_P

        # Dataset dependant parameters
        self.ndims = params.dataset.ndims
        inshape = params.dataset.inshape
        

        # -------------------------------------------------------------------
        # --------- 1. Enc:
        self.enc = EncoderBranch(in_channels,features,self.ndims,net_depth,dropout)
        
        # --------- 2. Bottleneck:
        self.bottleneck = BottleNeck(features,self.ndims,net_depth,dropout)

        # --------- 3. Decoder + Diffeo Units:
        if self.no_branches ==1:
            self.STN = WholeDiffeoUnit(params,branch=0)
        elif self.no_branches ==2:
            self.STN_bulk = WholeDiffeoUnit(params,branch=0)
            self.STN_ft = WholeDiffeoUnit(params,branch=1)

        # --------------------------------------------------------------------
        # --------- 4. Downsample to up-sampled fields (visualisation):
        if self.mega_P>1:
            if self.ndims ==2:
                from torch.nn import MaxPool2d as MaxPool
            elif self.ndims ==3:
                from torch.nn import MaxPool3d as MaxPool
            self.downsample =MaxPool(kernel_size= 3,stride =self.mega_P,padding=1) # downsample the final results

        # --------- 5. Gate mechanism for enhancing prior shape:
        # 添加门控网络，用于计算门控值
        bottleneck_features = features * (2 ** (net_depth - 1))
        self.gate_network = nn.Sequential(
            nn.AdaptiveAvgPool2d(1) if self.ndims == 2 else nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(bottleneck_features, bottleneck_features // 4),
            nn.ReLU(),
            nn.Linear(bottleneck_features // 4, 1),
            nn.Sigmoid()
        )
        
        # 添加注意力模块，用于生成注意力权重
        self.attention_module = nn.Sequential(
            nn.Conv2d(bottleneck_features, bottleneck_features // 8, kernel_size=1) if self.ndims == 2 
            else nn.Conv3d(bottleneck_features, bottleneck_features // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(bottleneck_features // 8, 1, kernel_size=1) if self.ndims == 2
            else nn.Conv3d(bottleneck_features // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 上采样模块，将注意力权重上采样到输出尺寸
        self.Mega_inshape = [s * self.mega_P for s in inshape]
        modes = {2: 'bilinear', 3: 'trilinear'}
        self.attention_upsample = nn.Upsample(
            size=self.Mega_inshape, 
            mode=modes[self.ndims], 
            align_corners=False
        ) if self.mega_P > 1 else None


    def forward(self, x, prior_shape):
        '''
        Inputs [ tensor] - dims [Batch,2,Chan,X,Y,Z] where the second dimension has both the prior shape and the image
        在此方法中，UNet网络生成的特征被用来创建位移场，该位移场应用于先验形状以生成最终分割
        '''
        
        # -------- 0. Get inputs
        #x = inputs[:,0,...] # first channel of batch
        #prior_shape =inputs[:,1:] # second channel of batch
 
        # -------- 1. Enc + Bottleneck:
        enc_outputs = self.enc(x)
        BottleNeck =self.bottleneck(enc_outputs[-1])

        # --------- 2. Dec + Diffeo:
        # 在这里，UNet网络的输出与先验形状结合
        # UNet生成位移场，然后将该位移场应用于先验形状
        if self.no_branches ==1:
            flow_field,flow_upsamp,sampled=self.STN(BottleNeck,enc_outputs,prior_shape)
            
            # 门控增强机制
            # 计算门控值
            gate_value = self.gate_network(BottleNeck)
            # 生成注意力权重
            attention_weights = self.attention_module(BottleNeck)
            # 上采样注意力权重到输出尺寸
            if self.attention_upsample is not None:
                attention_weights = self.attention_upsample(attention_weights)
            # 应用门控增强机制: enhanced_prior = prior_shape * (1 + attention_weights * gate_value)
            # 需要确保gate_value能够广播到sampled的维度
            gate_value_expanded = gate_value.view(gate_value.size(0), 1, 1, 1) if len(sampled.shape) == 4 else gate_value.view(gate_value.size(0), 1, 1, 1, 1)
            enhanced_output = sampled * (1 + attention_weights * gate_value_expanded)
            
            # DOWNSAMPLE
            if self.mega_P>1:
                enhanced_output = self.downsample(enhanced_output)

            # 返回增强后的形变先验形状作为分割结果
            return enhanced_output,flow_upsamp

        elif self.no_branches ==2:
            # 双分支结构，先用bulk分支处理，再用ft分支细化
            flow_bulk_field,flow_bulk_upsamp,bulk_sampled=self.STN_bulk(BottleNeck,enc_outputs,prior_shape)
            
            # 对bulk分支应用门控增强机制
            gate_value_bulk = self.gate_network(BottleNeck)
            attention_weights_bulk = self.attention_module(BottleNeck)
            # 上采样注意力权重到输出尺寸
            if self.attention_upsample is not None:
                attention_weights_bulk = self.attention_upsample(attention_weights_bulk)
            # 应用门控增强机制
            gate_value_bulk_expanded = gate_value_bulk.view(gate_value_bulk.size(0), 1, 1, 1) if len(bulk_sampled.shape) == 4 else gate_value_bulk.view(gate_value_bulk.size(0), 1, 1, 1, 1)
            enhanced_bulk = bulk_sampled * (1 + attention_weights_bulk * gate_value_bulk_expanded)
            
            flow_ft_field,flow_ft_upsamp,ft_sampled=self.STN_ft(BottleNeck,enc_outputs,enhanced_bulk)

            # 对ft分支也可以应用门控机制（可选）
            gate_value_ft = self.gate_network(BottleNeck)
            attention_weights_ft = self.attention_module(BottleNeck)
            # 上采样注意力权重到输出尺寸
            if self.attention_upsample is not None:
                attention_weights_ft = self.attention_upsample(attention_weights_ft)
            # 应用门控增强机制
            gate_value_ft_expanded = gate_value_ft.view(gate_value_ft.size(0), 1, 1, 1) if len(ft_sampled.shape) == 4 else gate_value_ft.view(gate_value_ft.size(0), 1, 1, 1, 1)
            enhanced_output = ft_sampled * (1 + attention_weights_ft * gate_value_ft_expanded)

            if self.mega_P>1:
                enhanced_bulk = self.downsample(enhanced_bulk)
                enhanced_output = self.downsample(enhanced_output)
            
            # 返回最终的增强形变先验形状作为分割结果
            return enhanced_output, flow_bulk_upsamp,flow_ft_upsamp