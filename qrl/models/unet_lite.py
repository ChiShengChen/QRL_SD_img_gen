"""輕量級 UNet 模型，作為 diffusers UNet 的替代選擇。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import math


class ResBlock(nn.Module):
    """殘差塊。"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.1
    ):
        """
        初始化殘差塊。
        
        Args:
            in_channels: 輸入通道數
            out_channels: 輸出通道數
            time_emb_dim: 時間嵌入維度
            dropout: Dropout 比率
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 時間嵌入投影
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        # 第一個卷積塊
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        # 第二個卷積塊
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        # 殘差連接
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        前向傳播。
        
        Args:
            x: 輸入特徵 [B, C, H, W]
            time_emb: 時間嵌入 [B, time_emb_dim]
            
        Returns:
            torch.Tensor: 輸出特徵
        """
        residual = self.residual_conv(x)
        
        # 第一個卷積塊
        h = self.block1(x)
        
        # 添加時間嵌入
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb.unsqueeze(-1).unsqueeze(-1)
        
        # 第二個卷積塊
        h = self.block2(h)
        
        # 殘差連接
        return h + residual


class CrossAttention(nn.Module):
    """交叉注意力模組。"""
    
    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        heads: int = 8,
        dim_head: int = 64
    ):
        """
        初始化交叉注意力。
        
        Args:
            query_dim: 查詢維度
            context_dim: 上下文維度
            heads: 注意力頭數
            dim_head: 每個頭的維度
        """
        super().__init__()
        
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        
        inner_dim = dim_head * heads
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(0.1)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向傳播。
        
        Args:
            x: 查詢張量 [B, N, D]
            context: 上下文張量 [B, M, D]
            
        Returns:
            torch.Tensor: 注意力輸出
        """
        if context is None:
            context = x
        
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        # 重塑為多頭注意力
        q = q.view(q.shape[0], q.shape[1], self.heads, self.dim_head).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], self.heads, self.dim_head).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], self.heads, self.dim_head).transpose(1, 2)
        
        # 計算注意力分數
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        
        # 應用注意力
        out = torch.matmul(attn, v)
        
        # 重塑回原始形狀
        out = out.transpose(1, 2).contiguous().view(
            out.shape[0], out.shape[2], self.heads * self.dim_head
        )
        
        return self.to_out(out)


class UNetLite(nn.Module):
    """輕量級 UNet 模型。"""
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        model_channels: int = 128,
        num_res_blocks: int = 2,
        attention_resolutions: tuple = (8, 16),
        dropout: float = 0.1,
        channel_mult: tuple = (1, 2, 4, 8),
        conv_resample: bool = True,
        num_heads: int = 8,
        use_spatial_transformer: bool = True,
        transformer_depth: int = 1,
        context_dim: Optional[int] = None,
        time_embed_dim: Optional[int] = None
    ):
        """
        初始化輕量級 UNet。
        
        Args:
            in_channels: 輸入通道數
            out_channels: 輸出通道數
            model_channels: 模型通道數
            num_res_blocks: 每個層級的殘差塊數
            attention_resolutions: 注意力分辨率
            dropout: Dropout 比率
            channel_mult: 通道倍數
            conv_resample: 是否使用卷積重採樣
            num_heads: 注意力頭數
            use_spatial_transformer: 是否使用空間變換器
            transformer_depth: 變換器深度
            context_dim: 上下文維度
            time_embed_dim: 時間嵌入維度
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        self.use_spatial_transformer = use_spatial_transformer
        self.transformer_depth = transformer_depth
        self.context_dim = context_dim
        self.time_embed_dim = time_embed_dim or model_channels * 4
        
        # 時間嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim)
        )
        
        # 輸入投影
        self.input_blocks = nn.ModuleList([
            nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        ])
        
        # 下採樣路徑
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(ch, mult * model_channels, self.time_embed_dim, dropout)
                ]
                ch = mult * model_channels
                
                if ds in attention_resolutions:
                    if use_spatial_transformer:
                        layers.append(
                            CrossAttention(
                                ch, context_dim or ch, num_heads, ch // num_heads
                            )
                        )
                
                self.input_blocks.append(nn.ModuleList(layers))
                input_block_chans.append(ch)
            
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    nn.ModuleList([nn.Conv2d(ch, ch, 3, stride=2, padding=1)])
                )
                input_block_chans.append(ch)
                ds *= 2
        
        # 中間塊
        self.middle_block = nn.ModuleList([
            ResBlock(ch, ch, self.time_embed_dim, dropout),
            CrossAttention(ch, context_dim or ch, num_heads, ch // num_heads) if use_spatial_transformer else nn.Identity(),
            ResBlock(ch, ch, self.time_embed_dim, dropout)
        ])
        
        # 上採樣路徑
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(ch + ich, mult * model_channels, self.time_embed_dim, dropout)
                ]
                ch = mult * model_channels
                
                if ds in attention_resolutions:
                    if use_spatial_transformer:
                        layers.append(
                            CrossAttention(
                                ch, context_dim or ch, num_heads, ch // num_heads
                            )
                        )
                
                if level and i == num_res_blocks:
                    layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
                    ds //= 2
                
                self.output_blocks.append(nn.ModuleList(layers))
        
        # 輸出投影
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1)
        )
        
        # 初始化權重
        self._init_weights()
    
    def _init_weights(self):
        """初始化權重。"""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向傳播。
        
        Args:
            x: 輸入張量 [B, C, H, W]
            timesteps: 時間步 [B]
            context: 上下文張量 [B, N, D]
            
        Returns:
            torch.Tensor: 輸出張量
        """
        # 時間嵌入
        t_emb = timestep_embedding(timesteps, self.model_channels)
        t_emb = self.time_embed(t_emb)
        
        # 輸入塊
        h = x
        hs = []
        for module_list in self.input_blocks:
            if isinstance(module_list, nn.Conv2d):
                h = module_list(h)
            else:
                for module in module_list:
                    if isinstance(module, ResBlock):
                        h = module(h, t_emb)
                    elif isinstance(module, CrossAttention):
                        # 重塑為序列
                        b, c, h_, w_ = h.shape
                        h_reshaped = h.view(b, c, h_ * w_).transpose(1, 2)
                        h_attn = module(h_reshaped, context)
                        h = h_attn.transpose(1, 2).view(b, c, h_, w_)
                    else:
                        h = module(h)
            hs.append(h)
        
        # 中間塊
        for module in self.middle_block:
            if isinstance(module, ResBlock):
                h = module(h, t_emb)
            elif isinstance(module, CrossAttention):
                b, c, h_, w_ = h.shape
                h_reshaped = h.view(b, c, h_ * w_).transpose(1, 2)
                h_attn = module(h_reshaped, context)
                h = h_attn.transpose(1, 2).view(b, c, h_, w_)
            else:
                h = module(h)
        
        # 輸出塊
        for module_list in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            for module in module_list:
                if isinstance(module, ResBlock):
                    h = module(h, t_emb)
                elif isinstance(module, CrossAttention):
                    b, c, h_, w_ = h.shape
                    h_reshaped = h.view(b, c, h_ * w_).transpose(1, 2)
                    h_attn = module(h_reshaped, context)
                    h = h_attn.transpose(1, 2).view(b, c, h_, w_)
                else:
                    h = module(h)
        
        # 輸出投影
        return self.out(h)


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    創建時間步嵌入。
    
    Args:
        timesteps: 時間步張量
        dim: 嵌入維度
        max_period: 最大週期
        
    Returns:
        torch.Tensor: 時間步嵌入
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    
    args = timesteps[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


# 便捷函數
def create_unet_lite(
    in_channels: int = 4,
    out_channels: int = 4,
    model_channels: int = 128,
    **kwargs
) -> UNetLite:
    """創建輕量級 UNet 的便捷函數。"""
    return UNetLite(
        in_channels=in_channels,
        out_channels=out_channels,
        model_channels=model_channels,
        **kwargs
    )
