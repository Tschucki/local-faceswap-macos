# Copyright 2024-2025 Bytedance Ltd. and/or its affiliates. All rights reserved.
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modified for Local FaceSwap macOS: Apple Silicon (MPS) support and pipeline changes.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import hashlib
import logging
import math
import os
import random
import sys
import time
import types
from contextlib import contextmanager
from functools import partial

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm
import torchvision.transforms.functional as TF

from .distributed.fsdp import shard_model
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .utils.na_resize import NaResize, DivisibleCrop, Rearrange
from torchvision.transforms import ToTensor,Normalize,Compose
import math
from PIL import Image, ImageOps

def _trace(name, t, tensor):
    return # Disable trace logging to prevent MPS CPU-GPU blocking synchronizations

class DreamIDV:

    def __init__(
        self,
        config,
        checkpoint_dir,
        dreamidv_ckpt,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        compute_dtype=None,
        t5_cache_dir=None,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            dreamidv_ckpt (`str`):
                Path of DreamID-V dit checkpoint
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            compute_dtype (`torch.dtype`, *optional*):
                Autocast dtype for the DiT on MPS. Weights stay float32; matmuls
                and attention run in this dtype (bf16 is ~4x faster than float32
                on Apple GPUs). None or torch.float32 disables autocast.
            t5_cache_dir (`str`, *optional*):
                Directory for cached T5 prompt embeddings. When the embedding for
                a prompt is cached, the ~20 GB T5 encoder is never loaded.
        """
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = torch.float32 if self.device.type == 'mps' else config.param_dtype
        if compute_dtype is None:
            compute_dtype = torch.float32 if self.device.type == 'mps' else config.param_dtype
        self.compute_dtype = compute_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self._t5_kwargs = dict(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)
        self.t5_cache_dir = t5_cache_dir
        self.text_encoder = None  # loaded lazily, only when a prompt is not cached

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        logging.info("Instantiating WanVAE...")
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel (DiT Model instantiation start) from {dreamidv_ckpt}")
        self.model = WanModel(
                             model_type=config.model_type,
                             dim=config.dim, 
                             ffn_dim=config.ffn_dim,
                             freq_dim=config.freq_dim,
                             in_dim=config.in_dim,
                             num_heads=config.num_heads,
                             num_layers=config.num_layers,
                             window_size=config.window_size,
                             qk_norm=config.qk_norm,
                             cross_attn_norm=config.cross_attn_norm,
                             eps=config.eps)
    
        logging.info(f"loading ckpt.")
        state = torch.load(dreamidv_ckpt, map_location=self.device)
        logging.info(f"loading state dict.")
        missing_keys, unexpected_keys = self.model.load_state_dict(state, strict=False)
        logging.info(f"len missing_keys: {len(missing_keys)}")
        logging.info(f"len unexpected_keys: {len(unexpected_keys)}")
        print(missing_keys)
        self.model.eval().requires_grad_(False)

        if use_usp:
            from xfuser.core.distributed import \
                get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (usp_attn_forward,
                                                            usp_dit_forward)
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)


    def _t5_cache_path(self, prompt):
        if not self.t5_cache_dir:
            return None
        key = hashlib.sha1(
            f"{prompt}|{self._t5_kwargs['checkpoint_path']}|{self.config.t5_dtype}".encode()
        ).hexdigest()[:16]
        return os.path.join(self.t5_cache_dir, f"t5_context_{key}.pt")

    def encode_prompt(self, prompt, offload_model=True):
        """Return the T5 context for `prompt`, from disk cache when available."""
        cache_path = self._t5_cache_path(prompt)
        if cache_path and os.path.exists(cache_path):
            logging.info(f"Using cached T5 embedding: {cache_path}")
            context = torch.load(cache_path, map_location='cpu', weights_only=True)
            return [t.to(self.device) for t in context]

        if self.text_encoder is None:
            logging.info("Instantiating T5EncoderModel (T5 standup start)...")
            self.text_encoder = T5EncoderModel(**self._t5_kwargs)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([prompt], self.device)
            if offload_model:
                # Not used again during denoising: free the memory right away.
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save([t.detach().cpu() for t in context], cache_path)
            logging.info(f"Saved T5 embedding cache: {cache_path}")
            # Cached from now on: drop the encoder to free ~20 GB.
            self.text_encoder = None
            gc.collect()
        return context

    @staticmethod
    def _read_frames(path, frame_num):
        """Read at most `frame_num` frames (cut to 4n+1) instead of the whole file."""
        from decord import VideoReader
        vr = VideoReader(path)
        n = min(len(vr), frame_num)
        n = (n - 1) // 4 * 4 + 1
        batch = vr.get_batch(list(range(n))).asnumpy()
        return [Image.fromarray(f) for f in batch]

    def load_image_latent_ref_ip_video(self,paths: str, size, device, frame_num):
        # Load size.
        patch_size = self.patch_size
        vae_stride = self.vae_stride

        def is_image_or_video_by_extension(file_path):
            image_exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
            video_exts = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm"}
            
            ext = os.path.splitext(file_path)[1].lower()
            if ext in image_exts:
                return "image"
            elif ext in video_exts:
                return "video"
            else:
                return "unknown"

        # import pdb; pdb.set_trace()
        # Load image and video.
        ref_vae_latents = {
            "image": [],
            "video": [],
            "mask": [],
            'pose_embedding':[]
        }
        video_h = 0
        video_w = 0
        from decord import VideoReader
        for i, path in enumerate(paths):
            
            if is_image_or_video_by_extension(path) == "video" and i == 0:
                print('ori_path', path)
                frames = self._read_frames(path, frame_num)
                video_w = frames[0].size[0]
                video_h = frames[0].size[1]

                video_transform=Compose(
                    [
                        NaResize(
                            resolution=math.sqrt(size[0] * size[1]), 
                            downsample_only=True,
                        ),
                        DivisibleCrop((vae_stride[1] * patch_size[1], vae_stride[2] * patch_size[2])),
                        Normalize(0.5, 0.5),
                        Rearrange("t c h w -> c t h w"),
                    ]
                )
                
                video_frames = video_transform(frames)
                video_vae_latent = self.vae.encode([video_frames], device)[0]
            
                ref_vae_latents["video"].append(video_vae_latent)

                

            elif is_image_or_video_by_extension(path) == "video" and i == 1:
                print('mask_path', path)
                frames = self._read_frames(path, frame_num)
                video_w = frames[0].size[0]
                video_h = frames[0].size[1]
                video_mask_transform=Compose(
                    [
                        NaResize(
                            resolution=math.sqrt(size[0] * size[1]), # 256*448, 480*832
                            downsample_only=True,
                        ),
                        DivisibleCrop((vae_stride[1] * patch_size[1], vae_stride[2] * patch_size[2])),
                        # Normalize(0.5, 0.5),
                        Rearrange("t c h w -> c t h w"),
                    ]
                )
                
                video_frames = video_mask_transform(frames) 
                video_vae_latent = self.vae.encode([video_frames], device)[0]
                video_vae_latent = video_vae_latent 
                ref_vae_latents["mask"].append(video_vae_latent)

            elif is_image_or_video_by_extension(path) == "video" and i == 3:
                print('pose_path', path)
                frames = self._read_frames(path, frame_num)
                video_mask_transform=Compose(
                    [
                        NaResize(
                            resolution=math.sqrt(size[0] * size[1]), 
                            downsample_only=True,   
                        ),
                        DivisibleCrop((vae_stride[1] * patch_size[1], vae_stride[2] * patch_size[2])),
                        Rearrange("t c h w -> c t h w"),
                    ]
                )
                
                video_frames = video_mask_transform(frames) 
                ref_vae_latents["pose_embedding"].append(video_frames)

            elif is_image_or_video_by_extension(path) == "image" and i == 2:

                with Image.open(path) as img:
                    img = img.convert("RGB")
                    img_ratio = img.width / img.height
                    target_ratio = video_w / video_h
                    
                    if img_ratio > target_ratio: 
                        new_width = video_w
                        new_height = int(new_width / img_ratio)
                    else:  
                        new_height = video_h
                        new_width = int(new_height * img_ratio)

                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    delta_w = video_w - img.size[0]
                    delta_h = video_h - img.size[1]
                    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
                    new_img = ImageOps.expand(img, padding, fill=(255, 255, 255))

                # Transform to tensor and normalize.
                image_transform=Compose(
                    [
                        NaResize(
                            resolution=math.sqrt(size[0] * size[1]), 
                            downsample_only=True,
                        ),
                        DivisibleCrop((vae_stride[1] * patch_size[1], vae_stride[2] * patch_size[2])),
                        Normalize(0.5, 0.5),
                        Rearrange("t c h w -> c t h w"),
                    ]
                )
                new_img = image_transform([new_img])
                img_vae_latent = self.vae.encode([new_img], device)[0]
                ref_vae_latents["image"].append(img_vae_latent)
            else:
                print("Unknown file type.")
       
        ref_vae_latents["image"] = torch.cat(ref_vae_latents["image"], dim=0)
        ref_vae_latents["video"] = torch.cat(ref_vae_latents["video"], dim=0)
        ref_vae_latents["mask"] = torch.cat(ref_vae_latents["mask"], dim=0)
        ref_vae_latents["pose_embedding"] = torch.cat(ref_vae_latents["pose_embedding"], dim=0)
        
        return ref_vae_latents


        
    def generate(self,
                 input_prompt,
                 paths,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='euler',
                 sampling_steps=50,
                 guide_scale_img=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
             paths ([`str`])`:
                Reference paths for swap face.
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        device = self.device   
        dtype = self.param_dtype
        
        latents_ref = self.load_image_latent_ref_ip_video(paths, 
                                                          size, 
                                                          device,
                                                          frame_num,
                                                        )

        latents_ref_video = latents_ref["video"].to(device,dtype)

        latents_ref_image = latents_ref["image"].to(device,dtype)
        pose_embedding = latents_ref["pose_embedding"].to(device,dtype) 
   
        pose_embedding = pose_embedding.unsqueeze(0)
        
        msk = latents_ref["mask"].to(device,dtype)

        F = frame_num
        target_shape = (self.vae.model.z_dim, latents_ref_video.shape[1],
                        latents_ref_video.shape[2],
                        latents_ref_video.shape[3])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        from dreamidv_wan.utils.speed_tracer import tracer
        tracer.start("Prompt Encoding")
        context = self.encode_prompt(input_prompt, offload_model=offload_model)
        tracer.stop("Prompt Encoding")
        
        y_i_v = latents_ref_video 
        img_ref = latents_ref_image

        arg_tiv = {
            'context': context, 
            'seq_len': seq_len,
            'y': [torch.concat([y_i_v, msk])],
            'pose_embedding': pose_embedding,
            'img_ref': [img_ref]
            }

        
        arg_tv = {
            'context': context,
            'seq_len': seq_len, 
            'y': [torch.concat([y_i_v, msk])],
            'pose_embedding': pose_embedding,
            'img_ref': [torch.zeros_like(img_ref)]
            }


        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # MPS: weights stay float32, matmuls/attention run in compute_dtype
        # (bf16 by default from the CLI). The model keeps its own float32
        # islands (time embedding, modulation, norms, RoPE, head).
        if self.device.type == 'mps':
            use_amp = self.compute_dtype in (torch.bfloat16, torch.float16)
            amp_ctx = torch.autocast(device_type='mps', dtype=self.compute_dtype, enabled=use_amp)
        else:
            amp_ctx = torch.amp.autocast(device_type=self.device.type, dtype=self.param_dtype)
        with amp_ctx, torch.no_grad(), no_sync():

            if sample_solver in ('unipc', 'euler'):
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise
            self.model.to(device=self.device, dtype=self.param_dtype)

            from dreamidv_wan.utils.speed_tracer import tracer

            for idx, t in enumerate(tqdm(timesteps)):
                tracer.start_step(idx)
                
                timestep = [t]
                timestep = torch.stack(timestep)

                if idx == 0:
                    logging.info(f"[DEBUG DTYPES] latents[0]: {latents[0].dtype}")
                    logging.info(f"[DEBUG DTYPES] timestep: {timestep.dtype}")
                    logging.info(f"[DEBUG DTYPES] context[0]: {arg_tiv['context'][0].dtype}")
                    logging.info(f"[DEBUG DTYPES] y[0]: {arg_tiv['y'][0].dtype}")
                    logging.info(f"[DEBUG DTYPES] pose_embedding: {arg_tiv['pose_embedding'].dtype}")
                    logging.info(f"[DEBUG DTYPES] img_ref[0]: {arg_tiv['img_ref'][0].dtype}")
                    logging.info(f"[DEBUG DTYPES] model param: {next(self.model.parameters()).dtype}")
                    if hasattr(self.model, 'project_pose'):
                        logging.info(f"[DEBUG DTYPES] project_pose conv1d weight: {self.model.project_pose.time_down[0].weight.dtype}")
                        if self.model.project_pose.time_down[0].bias is not None:
                            logging.info(f"[DEBUG DTYPES] project_pose conv1d bias: {self.model.project_pose.time_down[0].bias.dtype}")

                _trace("LATENTS_PRE_STEP", t, latents[0])
                
                t_dit_start = time.perf_counter()
                pos_tiv = self.model(latents, t=timestep, **arg_tiv)[0]
                pos_tv = self.model(latents, t=timestep, **arg_tv)[0]
                t_dit_end = time.perf_counter()
                tracer.log_step_metric(idx, 'dit', t_dit_end - t_dit_start)
                
                _trace("DIT_POS_TIV", t, pos_tiv)
                _trace("DIT_POS_TV", t, pos_tv)

                noise_pred = pos_tiv
                noise_pred += guide_scale_img * (pos_tiv - pos_tv) 
                _trace("CFG_NOISE_PRED", t, noise_pred)
                
                t_sched_start = time.perf_counter()
                if sample_solver == 'euler':
                    sigma_s0 = sample_scheduler.sigmas[idx]
                    sigma_t = sample_scheduler.sigmas[idx + 1]
                    temp_x0 = latents[0] - (sigma_s0 - sigma_t).to(self.device) * noise_pred
                else:
                    temp_x0 = sample_scheduler.step(
                        noise_pred.unsqueeze(0),
                        t,
                        latents[0].unsqueeze(0),
                        return_dict=False,
                        generator=seed_g)[0].squeeze(0)
                t_sched_end = time.perf_counter()
                tracer.log_step_metric(idx, 'scheduler', t_sched_end - t_sched_start)
                
                latents = [temp_x0]
                _trace("LATENTS_POST_STEP", t, latents[0])
                
                # Tracing optional mid-step offloading time (if offload_model were enabled)
                t_off_start = time.perf_counter()
                if offload_model:
                    # Note: currently self.model is NOT offloaded inside the loop,
                    # but if it were, this records the paging cost.
                    pass
                t_off_end = time.perf_counter()
                tracer.log_step_metric(idx, 'offload', t_off_end - t_off_start)

                tracer.stop_step(idx)

            x0 = latents

            if offload_model:
                t_off_start = time.perf_counter()
                self.model.cpu()
                torch.mps.empty_cache() if torch.backends.mps.is_available() else None
                t_off_end = time.perf_counter()
                logging.info(f"[SPEED] Final DiT offload & cache empty: {t_off_end - t_off_start:.3f}s")
            if self.rank == 0:
                logging.info("[VAE_DECODE_START] Starting VAE decode...")
                tracer.start("VAE Decoding")
                # Decode in full precision: bf16 VAE decode costs fine detail.
                with torch.autocast(device_type=self.device.type, enabled=False):
                    videos = self.vae.decode(x0)
                duration = tracer.stop("VAE Decoding")
                logging.info(f"[VAE_DECODE_END] VAE decode completed in {duration:.3f}s")

        del noise, latents
        del sample_scheduler

        if offload_model:
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
            elif torch.cuda.is_available():
                torch.cuda.synchronize()

        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
