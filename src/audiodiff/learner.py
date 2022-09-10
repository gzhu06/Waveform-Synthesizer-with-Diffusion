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
import os
import torch
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import from_path
from model import AudioDiffusionModel

def _nested_map(struct, map_fn):
    if isinstance(struct, tuple):
        return tuple(_nested_map(x, map_fn) for x in struct)
    if isinstance(struct, list):
        return [_nested_map(x, map_fn) for x in struct]
    if isinstance(struct, dict):
        return { k: _nested_map(v, map_fn) for k, v in struct.items() }
    return map_fn(struct)

class DiffWaveLearner:
    def __init__(self, model_dir, model, dataset, 
                 optimizer, params, *args, **kwargs):
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.params = params
        self.autocast = torch.cuda.amp.autocast(enabled=kwargs.get('fp16', False))
        self.scaler = torch.cuda.amp.GradScaler(enabled=kwargs.get('fp16', False))
        self.step = 0
        self.is_master = True

        self.summary_writer = None

    def state_dict(self):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        return {'step': self.step,
                'model': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items() },
                'optimizer': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items() },
                'params': dict(self.params), 'scaler': self.scaler.state_dict()}

    def load_state_dict(self, state_dict):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            self.model.module.load_state_dict(state_dict['model'])
        else:
            self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scaler.load_state_dict(state_dict['scaler'])
        self.step = state_dict['step']

    def save_to_checkpoint(self, filename='weights'):
        save_basename = f'{filename}-{self.step}.pt'
        save_name = f'{self.model_dir}/{save_basename}'
        link_name = f'{self.model_dir}/{filename}.pt'
        torch.save(self.state_dict(), save_name)
        if os.name == 'nt':
            torch.save(self.state_dict(), link_name)
        else:
            if os.path.islink(link_name):
                os.unlink(link_name)
            os.symlink(save_basename, link_name)
    
    def restore_from_checkpoint(self, filename='weights'):
        try:
            checkpoint = torch.load(f'{self.model_dir}/{filename}.pt')
            self.load_state_dict(checkpoint)
            return True
        except FileNotFoundError:
            return False

    def train(self, max_steps=None):
        device = next(self.model.parameters()).device
        while True:
            for features in tqdm(self.dataset, desc=f'Epoch {self.step//len(self.dataset)}') if self.is_master else self.dataset:
                if max_steps is not None and self.step >= max_steps:
                    return
                features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
                loss = self.train_step(features)
                if torch.isnan(loss).any():
                    raise RuntimeError(f'Detected NaN loss at step {self.step}.')
                if self.is_master:
                    if self.step % 50 == 0:
                        self._write_summary(self.step, features, loss)
                    if self.step % 50000 == 0:
                        self.save_to_checkpoint()
                self.step += 1

    def train_step(self, features):
        for param in self.model.parameters():
            param.grad = None

        audio = features['audio'].unsqueeze(1)
        N, _, T = audio.shape
        device = audio.device
        with self.autocast:
            loss = self.model(audio)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                  self.params.max_grad_norm or 1e9)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss
    
    def inference(self):
        device = torch.device('cuda', 0)
        with torch.no_grad():
            # audio initialization
            audio = torch.randn(1, 1, self.params.audio_len).to(device)
            sampled = self.model.sample(noise=audio, num_steps=50)
        return sampled

    def _write_summary(self, step, features, loss):
        writer = self.summary_writer or SummaryWriter(self.model_dir, purge_step=step)
        writer.add_audio('feature/audio_gt', features['audio'][0], step,
                         sample_rate=self.params.sample_rate)
        if step % 1000 == 0:
            print('generating audio:')
            audio_pre = self.inference()
            writer.add_audio('feature/audio_pred', audio_pre[0], step,
                             sample_rate=self.params.sample_rate)    
        writer.add_scalar('train/loss', np.log(loss.item()), step)
        writer.add_scalar('train/grad_norm', np.log(self.grad_norm.item()), step)
        writer.flush()
        self.summary_writer = writer

def _train_impl(replica_id, model, dataset, args, params):
    torch.backends.cudnn.benchmark = True
    opt = torch.optim.Adam(model.parameters(), lr=params.learning_rate, 
                           betas=(params.beta1, params.beta2),)
    
    learner = DiffWaveLearner(args.model_dir, model, dataset, opt, params, fp16=args.fp16)
    learner.is_master = (replica_id == 0)
    learner.restore_from_checkpoint()
    learner.train(max_steps=args.max_steps)

def train(args, params):
    dataset = from_path(args.data_dirs, params)
    model = AudioDiffusionModel(params).cuda()
    _train_impl(0, model, dataset, args, params)

def train_distributed(replica_id, replica_count, port, args, params):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    torch.distributed.init_process_group('nccl', rank=replica_id, 
                                         world_size=replica_count)
    
    dataset = from_path(args.data_dirs, params, is_distributed=True)
    device = torch.device('cuda', replica_id)
    torch.cuda.set_device(device)
    model = AudioDiffusionModel(params).to(device)
    model = DistributedDataParallel(model, device_ids=[replica_id])
    _train_impl(replica_id, model, dataset, args, params)
