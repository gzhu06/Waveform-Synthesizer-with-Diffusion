# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import numpy as np

class AttrDict(dict):
  def __init__(self, *args, **kwargs):
      super(AttrDict, self).__init__(*args, **kwargs)
      self.__dict__ = self

  def override(self, attrs):
    if isinstance(attrs, dict):
      self.__dict__.update(**attrs)
    elif isinstance(attrs, (list, tuple, set)):
      for attr in attrs:
        self.override(attr)
    elif attrs is not None:
      raise NotImplementedError
    return self

params = AttrDict(
    # Training params
    max_grad_norm=None,
    batch_size=48,
    learning_rate = 1e-4,
    beta1 = 0.9,
    beta2 = 0.99,

    # Data params
    sample_rate=16000,

    # Model params
    in_channels = 1,
    channels = 128,
    patch_size = 16,
    resnet_groups = 8,
    kernel_multiplier_downsample = 2,
    kernel_sizes_init = [1, 3, 7],
    multipliers = [1, 2, 4, 4, 4, 4, 4],
    factors = [4, 4, 4, 2, 2, 2],
    num_blocks = [2, 2, 2, 2, 2, 2],
    attentions = [False, False, False, True, True, True],
    attention_heads = 8,
    attention_features = 64,
    attention_multiplier = 2,
    use_nearest_upsample = False,
    use_skip_scale = True,
    use_attention_bottleneck = True,
    diffusion_sigma_data = 0.2,
    diffusion_dynamic_threshold = 0.0,
    noise_schedule = np.linspace(1e-4, 0.02, 200).tolist(),

    # unconditional sample len
    audio_len = 2**14, # unconditional_synthesis_samples
)
