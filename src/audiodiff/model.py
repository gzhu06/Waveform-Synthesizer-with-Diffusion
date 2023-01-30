from typing import Optional, Sequence

from torch import Tensor, nn

from diffusion import (
    ADPM2Sampler,
    Diffusion,
    DiffusionSampler,
    Distribution,
    KarrasSchedule,
    LogNormalDistribution,
    Sampler,
    Schedule,
)
from modules import UNet1d

""" Diffusion Classes (generic for 1d data) """

class Model1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        patch_size: int,
        kernel_sizes_init: Sequence[int],
        multipliers: Sequence[int],
        factors: Sequence[int],
        num_blocks: Sequence[int],
        attentions: Sequence[bool],
        attention_heads: int,
        attention_features: int,
        attention_multiplier: int,
        use_attention_bottleneck: bool,
        resnet_groups: int,
        kernel_multiplier_downsample: int,
        use_nearest_upsample: bool,
        use_skip_scale: bool,
        diffusion_sigma_distribution: Distribution,
        diffusion_sigma_data: int,
        diffusion_dynamic_threshold: float,
        out_channels: Optional[int] = None,
        context_channels: Optional[Sequence[int]] = None,
        **kwargs
    ):
        super().__init__()

        self.unet = UNet1d(
            in_channels=in_channels,
            channels=channels,
            patch_size=patch_size,
            kernel_sizes_init=kernel_sizes_init,
            multipliers=multipliers,
            factors=factors,
            num_blocks=num_blocks,
            attentions=attentions,
            attention_heads=attention_heads,
            attention_features=attention_features,
            attention_multiplier=attention_multiplier,
            use_attention_bottleneck=use_attention_bottleneck,
            resnet_groups=resnet_groups,
            kernel_multiplier_downsample=kernel_multiplier_downsample,
            use_nearest_upsample=use_nearest_upsample,
            use_skip_scale=use_skip_scale,
            out_channels=out_channels,
            context_channels=context_channels,
            **kwargs
        )

        self.diffusion = Diffusion(
            net=self.unet,
            sigma_distribution=diffusion_sigma_distribution,
            sigma_data=diffusion_sigma_data,
            dynamic_threshold=diffusion_dynamic_threshold,
        )

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.diffusion(x, **kwargs)

    def sample(
        self,
        noise: Tensor,
        num_steps: int,
        sigma_schedule: Schedule,
        sampler: Sampler,
        **kwargs
    ) -> Tensor:
        diffusion_sampler = DiffusionSampler(
            diffusion=self.diffusion,
            sampler=sampler,
            sigma_schedule=sigma_schedule,
            num_steps=num_steps,
        )
        return diffusion_sampler(noise, **kwargs)
    
class AudioDiffusionModel(Model1d):
    def __init__(self, params, *args, **kwargs):
        default_kwargs = dict(
            in_channels=params.in_channels,
            channels=params.channels,
            patch_size=params.patch_size,
            kernel_sizes_init=params.kernel_sizes_init,
            multipliers=params.multipliers,
            factors=params.factors,
            num_blocks=params.num_blocks,
            attentions=params.attentions,
            attention_heads=params.attention_heads,
            attention_features=params.attention_features,
            attention_multiplier=params.attention_multiplier,
            use_attention_bottleneck=params.use_attention_bottleneck,
            resnet_groups=params.resnet_groups,
            kernel_multiplier_downsample=params.kernel_multiplier_downsample,
            use_nearest_upsample=params.use_nearest_upsample,
            use_skip_scale=params.use_skip_scale,
            diffusion_sigma_distribution=LogNormalDistribution(mean=-3.0, std=1.0),
            diffusion_sigma_data=params.diffusion_sigma_data,
            diffusion_dynamic_threshold=params.diffusion_dynamic_threshold,
        )

        super().__init__(*args, **{**default_kwargs, **kwargs})

    def sample(self, *args, **kwargs):
        default_kwargs = dict(
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
            sampler=ADPM2Sampler(rho=1.0),
        )
        return super().sample(*args, **{**default_kwargs, **kwargs})
