# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modified for Local FaceSwap macOS: Apple Silicon (MPS) support and pipeline changes.
import logging
import math
import os
import cv2
import types
from copy import deepcopy
from functools import partial
from einops import rearrange
import numpy as np
import torch

import hashlib
import time

import torch.distributed as dist
from decord import VideoReader
from tqdm import tqdm
import torch.nn.functional as F
from .distributed.fsdp import shard_model
from .distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from .distributed.util import get_world_size

from .modules.animate import WanAnimateModel
from .modules.animate import CLIPModel
from .modules.t5 import T5EncoderModel
from .modules.vae2_1 import Wan2_1_VAE
from .modules.animate.animate_utils import TensorList
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler



class WanAnimate:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
        use_relighting_lora=False,
        t5_cache_dir=None,
        extra_loras=(),
    ):
        r"""
        Initializes the generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_sp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of sequence parallel.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.
            use_relighting_lora (`bool`, *optional*, defaults to False):
               Whether to use relighting lora for character replacement.
            t5_cache_dir (`str`, *optional*):
               MPS port: cache encoded prompts here; T5 (11 GB) is then only
               loaded when a prompt is not cached yet.
            extra_loras (list of (path, strength), *optional*):
               Further LoRAs merged into the DiT, e.g. the lightx2v step-distill
               LoRA for 4-8 step sampling.
        """
        self.extra_loras = list(extra_loras)
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{device_id}")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        shard_fn = partial(shard_model, device_id=device_id)
        # Lazy T5: built on first cache miss, dropped right after encoding.
        self._t5_kwargs = dict(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )
        self.text_encoder = None
        self.t5_cache_dir = t5_cache_dir

        self.clip = CLIPModel(
            dtype=torch.float16,
            device=self.device,
            checkpoint_path=os.path.join(checkpoint_dir,
                                         config.clip_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.clip_tokenizer))

        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanAnimate from {checkpoint_dir}")

        if not dit_fsdp and self.device.type == 'mps':
            # MPS port: device_map='mps' copies every tensor straight out of the
            # mmap'd safetensors, which page-faults at ~30 MB/s (20+ min for
            # 33 GB). Reading the shards into RAM first runs at SSD speed.
            self.noise_model = self._load_dit_mps(checkpoint_dir)
        elif not dit_fsdp:
            self.noise_model = WanAnimateModel.from_pretrained(
                checkpoint_dir,
                torch_dtype=self.param_dtype,
                device_map=self.device)
        else:
            self.noise_model = WanAnimateModel.from_pretrained(
                checkpoint_dir, torch_dtype=self.param_dtype)

        self.noise_model = self._configure_model(
            model=self.noise_model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype,
            use_lora=use_relighting_lora,
            checkpoint_dir=checkpoint_dir,
            config=config
            )

        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt
        self.sample_prompt = config.prompt


    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
                         convert_model_dtype, use_lora, checkpoint_dir, config):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.

        Args:
            model (torch.nn.Module):
                The model instance to configure.
            use_sp (`bool`):
                Enable distribution strategy of sequence parallel.
            dit_fsdp (`bool`):
                Enable FSDP sharding for DiT model.
            shard_fn (callable):
                The function to apply FSDP sharding.
            convert_model_dtype (`bool`):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.

        Returns:
            torch.nn.Module:
                The configured model.
        """
        model.eval().requires_grad_(False)

        if use_sp:
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(
                    sp_attn_forward, block.self_attn)

            model.use_context_parallel = True

        if dist.is_initialized():
            dist.barrier()

        if use_lora:
            logging.info("Loading Relighting Lora. ")
            lora_path = os.path.join(checkpoint_dir, config.lora_checkpoint)
            self._merge_lora(model, lora_path, alpha=128)
        for path, strength in self.extra_loras:
            self._merge_lora(model, path, strength=strength)

        if dit_fsdp:
            model = shard_fn(model, use_lora=use_lora)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            if not self.init_on_cpu:
                model.to(self.device)

        if self.device.type != 'cuda':
            self._upcast_fp32_islands(model)

        return model

    def _load_dit_mps(self, checkpoint_dir):
        """MPS port: from_pretrained(device_map='mps') hands the GPU tensors that
        live in mmap'd (file-backed) pages; the Metal copies then stall for many
        minutes. Here each shard is read into ordinary RAM, deserialized and
        copied to the GPU, parameters are assigned into an empty (meta) model."""
        import json
        from accelerate import init_empty_weights
        from safetensors.torch import load as load_safetensors

        t0 = time.time()
        config = WanAnimateModel.load_config(checkpoint_dir)
        with init_empty_weights():
            model = WanAnimateModel.from_config(config)
        index_path = os.path.join(checkpoint_dir, 'diffusion_pytorch_model.safetensors.index.json')
        if os.path.exists(index_path):
            with open(index_path) as f:
                shards = sorted(set(json.load(f)['weight_map'].values()))
        else:
            shards = ['diffusion_pytorch_model.safetensors']
        total = 0
        for shard in shards:
            with open(os.path.join(checkpoint_dir, shard), 'rb') as f:
                state = load_safetensors(f.read())
            state = {k: v.to(device=self.device, dtype=self.param_dtype) for k, v in state.items()}
            total += sum(v.numel() for v in state.values())
            model.load_state_dict(state, strict=False, assign=True)
            del state
        missing = [n for n, p in model.named_parameters() if p.device.type == 'meta']
        if missing:
            raise RuntimeError(f'{len(missing)} DiT weights missing in checkpoint, e.g. {missing[:3]}')
        model.to(self.device)  # buffers created on the CPU
        logging.info(f'DiT loaded ({total / 1e9:.1f}B params) in {time.time() - t0:.0f}s')
        return model

    def _merge_lora(self, model, lora_path, alpha=None, strength=1.0):
        """MPS port: merge a LoRA into the base weights instead of wrapping the
        Linear layers with peft adapters (same math, no per-step overhead, no
        peft dependency). Understands peft (lora_A/lora_B) and ComfyUI/Kijai
        (lora_down/lora_up, optional .alpha, .diff_b bias and .diff weight
        deltas) layouts, .pt/.pth or .safetensors.
            W += strength * B @ A * alpha / rank
        alpha=None: per-layer .alpha key if present, else rank (scale 1)."""
        if lora_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            sd = load_file(lora_path)
        else:
            sd = torch.load(lora_path, map_location='cpu', weights_only=True)
            sd = sd.get('state_dict', sd)
        modules = dict(model.named_modules())

        def module_of(prefix):
            name = prefix
            for p in ('base_model.model.', 'transformer.', 'diffusion_model.'):
                name = name[len(p):] if name.startswith(p) else name
            return modules.get(name.replace('.base_layer', ''))

        merged, skipped = 0, []
        for key, a in sd.items():
            for down, up in (('lora_A.weight', 'lora_B.weight'), ('lora_down.weight', 'lora_up.weight')):
                if key.endswith(down):
                    break
            else:
                continue
            prefix = key[:-len(down) - 1]
            mod = module_of(prefix)
            if mod is None:
                skipped.append(prefix)
                continue
            b = sd[f'{prefix}.{up}']
            rank = a.shape[0]
            al = alpha if alpha is not None else float(sd.get(f'{prefix}.alpha', rank))
            w = mod.weight
            delta = b.to(w.device, torch.float32) @ a.to(w.device, torch.float32)
            w.data = (w.data.float() + delta * (strength * al / rank)).to(w.dtype)
            merged += 1
        for key, d in sd.items():  # full-tensor deltas (biases, norm weights)
            for suffix, attr in (('.diff_b', 'bias'), ('.diff', 'weight')):
                if key.endswith(suffix):
                    mod = module_of(key[:-len(suffix)])
                    p = getattr(mod, attr, None) if mod is not None else None
                    if p is None or p.shape != d.shape:
                        skipped.append(key)
                        break
                    p.data = (p.data.float() + strength * d.to(p.device, torch.float32)).to(p.dtype)
                    merged += 1
                    break
        if merged == 0:
            raise RuntimeError(f'no LoRA weights found in {lora_path}')
        msg = f'merged {os.path.basename(lora_path)} (x{strength:g}) into {merged} tensors'
        if skipped:
            msg += f', {len(skipped)} not in this model (e.g. {skipped[0]})'
        logging.info(msg)

    @staticmethod
    def _upcast_fp32_islands(model):
        """Upstream runs time embedding, modulation, head and the motion
        direction under CUDA autocast(float32), which upcasts the bf16 weights
        on the fly. MPS autocast has no float32 mode, so these small modules are
        kept in float32 instead (exact: bf16 -> fp32 is lossless)."""
        for m in (model.time_embedding, model.time_projection, model.head,
                  model.motion_encoder.dec):
            m.float()
        for blk in model.blocks:
            blk.modulation.data = blk.modulation.data.float()

    def _t5_cache_path(self, prompt):
        key = f"{prompt}|{self._t5_kwargs['checkpoint_path']}|{self._t5_kwargs['dtype']}"
        digest = hashlib.sha256(key.encode('utf-8')).hexdigest()[:16]
        return os.path.join(self.t5_cache_dir, f't5_context_{digest}.pt')

    def encode_prompt(self, prompt):
        path = self._t5_cache_path(prompt) if self.t5_cache_dir else None
        if path and os.path.exists(path):
            context = torch.load(path, map_location='cpu', weights_only=True)
        else:
            if self.text_encoder is None:
                self.text_encoder = T5EncoderModel(**self._t5_kwargs)
            device = torch.device('cpu') if self.t5_cpu else self.device
            self.text_encoder.model.to(device)
            context = [t.cpu() for t in self.text_encoder([prompt], device)]
            self.text_encoder.model.cpu()
            if path:
                os.makedirs(self.t5_cache_dir, exist_ok=True)
                torch.save(context, path)
            if self.t5_cache_dir:
                self.text_encoder = None  # free 11 GB; cache covers next runs
        return [t.to(self.device) for t in context]

    def inputs_padding(self, array, target_len):
        idx = 0
        flip = False
        target_array = []
        while len(target_array) < target_len:
            target_array.append(deepcopy(array[idx]))
            if flip:
                idx -= 1
            else:
                idx += 1
            if idx == 0 or idx == len(array) - 1:
                flip = not flip
        return target_array[:target_len]

    def get_valid_len(self, real_len, clip_len=81, overlap=1):
        real_clip_len = clip_len - overlap
        last_clip_num = (real_len - overlap) % real_clip_len
        if last_clip_num == 0:
            extra = 0
        else:
            extra = real_clip_len - last_clip_num
        target_len = real_len + extra
        return target_len


    def get_i2v_mask(self, lat_t, lat_h, lat_w, mask_len=1, mask_pixel_values=None, device=None):
        device = device if device is not None else self.device
        if mask_pixel_values is None:
            msk = torch.zeros(1, (lat_t-1) * 4 + 1, lat_h, lat_w, device=device)
        else:
            msk = mask_pixel_values.clone()
        msk[:, :mask_len] = 1
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]
        return msk

    def padding_resize(self, img_ori, height=512, width=512, padding_color=(0, 0, 0), interpolation=cv2.INTER_LINEAR):
        ori_height = img_ori.shape[0]
        ori_width = img_ori.shape[1]
        channel = img_ori.shape[2]

        img_pad = np.zeros((height, width, channel))
        if channel == 1:
            img_pad[:, :, 0] = padding_color[0]
        else:
            img_pad[:, :, 0] = padding_color[0]
            img_pad[:, :, 1] = padding_color[1]
            img_pad[:, :, 2] = padding_color[2]

        if (ori_height / ori_width) > (height / width):
            new_width = int(height / ori_height * ori_width)
            img = cv2.resize(img_ori, (new_width, height), interpolation=interpolation)
            padding = int((width - new_width) / 2)
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]  
            img_pad[:, padding: padding + new_width, :] = img
        else:
            new_height = int(width / ori_width * ori_height)
            img = cv2.resize(img_ori, (width, new_height), interpolation=interpolation)
            padding = int((height - new_height) / 2)
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]  
            img_pad[padding: padding + new_height, :, :] = img

        img_pad = np.uint8(img_pad)

        return img_pad

    def prepare_source(self, src_pose_path, src_face_path, src_ref_path):
        pose_video_reader = VideoReader(src_pose_path)
        pose_len = len(pose_video_reader)
        pose_idxs = list(range(pose_len))
        cond_images = pose_video_reader.get_batch(pose_idxs).asnumpy()

        face_video_reader = VideoReader(src_face_path)
        face_len = len(face_video_reader)
        face_idxs = list(range(face_len))
        face_images = face_video_reader.get_batch(face_idxs).asnumpy()
        height, width = cond_images[0].shape[:2]
        refer_images = cv2.imread(src_ref_path)[..., ::-1]
        refer_images = self.padding_resize(refer_images, height=height, width=width)
        return cond_images, face_images, refer_images
    
    def prepare_source_for_replace(self, src_bg_path, src_mask_path):
        bg_video_reader = VideoReader(src_bg_path)
        bg_len = len(bg_video_reader)
        bg_idxs = list(range(bg_len))
        bg_images = bg_video_reader.get_batch(bg_idxs).asnumpy()

        mask_video_reader = VideoReader(src_mask_path)
        mask_len = len(mask_video_reader)
        mask_idxs = list(range(mask_len))
        mask_images = mask_video_reader.get_batch(mask_idxs).asnumpy()
        mask_images = mask_images[:, :, :, 0] / 255
        return bg_images, mask_images

    def generate(
        self,
        src_root_path,
        replace_flag=False,
        clip_len=77,
        refert_num=1,
        shift=5.0,
        sample_solver='dpm++',
        sampling_steps=20,
        guide_scale=1,
        input_prompt="",
        n_prompt="",
        seed=-1,
        offload_model=True,
        inputs=None,
        progress=None,
    ):
        r"""
        Generates video frames from input image using diffusion process.

        Args:
            src_root_path ('str'):
                Process output path
            replace_flag (`bool`, *optional*, defaults to False):
                Whether to use character replace.
            clip_len (`int`, *optional*, defaults to 77):
                How many frames to generate per clips. The number should be 4n+1
            refert_num (`int`, *optional*, defaults to 1):
                How many frames used for temporal guidance. Recommended to be 1 or 5.
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. 
            sample_solver (`str`, *optional*, defaults to 'dpm++'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 20):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float` or tuple[`float`], *optional*, defaults 1.0):
                Classifier-free guidance scale. We only use it for expression control. 
                In most cases, it's not necessary and faster generation can be achieved without it. 
                When expression adjustments are needed, you may consider using this feature.
            input_prompt (`str`):
                Text prompt for content generation. We don't recommend custom prompts (although they work)
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
            inputs (`dict`, *optional*):
                MPS port: in-memory inputs instead of the src_*.mp4 files in
                src_root_path (avoids lossy mp4 round trips, e.g. of the mask).
                Keys: cond_images, face_images (T,512,512,3), refer_image (H,W,3),
                and for replacement bg_images and mask_images (T,H,W) in [0,1]. RGB uint8.
            progress (`callable`, *optional*):
                Called as progress(step, total_steps, clip_idx, n_clips) after each step.

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N, H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames
                - H: Frame height 
                - W: Frame width 
        """
        assert refert_num == 1 or refert_num == 5, "refert_num should be 1 or 5."

        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        if input_prompt == "":
            input_prompt = self.sample_prompt

        if inputs is not None:
            cond_images = list(inputs['cond_images'])
            face_images = list(inputs['face_images'])
            h0, w0 = cond_images[0].shape[:2]
            refer_images = self.padding_resize(inputs['refer_image'], height=h0, width=w0)
        else:
            src_pose_path = os.path.join(src_root_path, "src_pose.mp4")
            src_face_path = os.path.join(src_root_path, "src_face.mp4")
            src_ref_path = os.path.join(src_root_path, "src_ref.png")
            cond_images, face_images, refer_images = self.prepare_source(src_pose_path=src_pose_path, src_face_path=src_face_path, src_ref_path=src_ref_path)

        # MPS port: cached prompt encoding; the negative prompt is only needed with CFG.
        context = self.encode_prompt(input_prompt)
        context_null = self.encode_prompt(n_prompt) if guide_scale > 1 else None

        real_frame_len = len(cond_images)
        target_len = self.get_valid_len(real_frame_len, clip_len, overlap=refert_num)
        logging.info('real frames: {} target frames: {}'.format(real_frame_len, target_len))
        cond_images = self.inputs_padding(cond_images, target_len)
        face_images = self.inputs_padding(face_images, target_len)

        if replace_flag:
            if inputs is not None:
                bg_images, mask_images = list(inputs['bg_images']), list(inputs['mask_images'])
            else:
                src_bg_path = os.path.join(src_root_path, "src_bg.mp4")
                src_mask_path = os.path.join(src_root_path, "src_mask.mp4")
                bg_images, mask_images = self.prepare_source_for_replace(src_bg_path, src_mask_path)
            bg_images = self.inputs_padding(bg_images, target_len)
            mask_images = self.inputs_padding(mask_images, target_len)
        n_clips = max(1, (target_len - refert_num) // (clip_len - refert_num))
        clip_idx = 0

        height, width = refer_images.shape[:2]
        start = 0
        end = clip_len
        all_out_frames = []
        while True:
            if start + refert_num >= len(cond_images):
                break

            if start == 0:
                mask_reft_len = 0
            else:
                mask_reft_len = refert_num

            batch = {
                        "conditioning_pixel_values": torch.zeros(1, 3, clip_len, height, width),
                        "bg_pixel_values": torch.zeros(1, 3, clip_len, height, width),
                        "mask_pixel_values": torch.zeros(1, 1, clip_len, height, width),
                        "face_pixel_values": torch.zeros(1, 3, clip_len, 512, 512),
                        "refer_pixel_values": torch.zeros(1, 3, height, width),
                        "refer_t_pixel_values": torch.zeros(refert_num, 3, height, width)
                    }   

            batch["conditioning_pixel_values"] = rearrange(
                torch.tensor(np.stack(cond_images[start:end]) / 127.5 - 1),
                "t h w c -> 1 c t h w",
            )
            batch["face_pixel_values"] = rearrange(
                torch.tensor(np.stack(face_images[start:end]) / 127.5 - 1),
                "t h w c -> 1 c t h w",
            )

            batch["refer_pixel_values"] = rearrange(
                torch.tensor(refer_images / 127.5 - 1), "h w c -> 1 c h w"
            )

            if start > 0:
                batch["refer_t_pixel_values"] = rearrange(
                    out_frames[0, :, -refert_num:].clone().detach(),
                    "c t h w -> t c h w",
                )

            batch["refer_t_pixel_values"] = rearrange(batch["refer_t_pixel_values"],
                                            "t c h w -> 1 c t h w",
                                            )

            if replace_flag:
                batch["bg_pixel_values"] = rearrange(
                    torch.tensor(np.stack(bg_images[start:end]) / 127.5 - 1),
                    "t h w c -> 1 c t h w",
                )

                batch["mask_pixel_values"] = rearrange(
                    torch.tensor(np.stack(mask_images[start:end])[:, :, :, None]),
                    "t h w c -> 1 t c h w",
                )
                

            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device=self.device, dtype=torch.bfloat16)

            ref_pixel_values = batch["refer_pixel_values"]
            refer_t_pixel_values = batch["refer_t_pixel_values"]
            conditioning_pixel_values = batch["conditioning_pixel_values"]
            face_pixel_values = batch["face_pixel_values"]

            B, _, H, W = ref_pixel_values.shape
            T = clip_len
            lat_h = H // 8
            lat_w = W // 8
            lat_t = T // 4 + 1
            target_shape = [lat_t + 1, lat_h, lat_w]
            noise = [
                torch.randn(
                    16,
                    target_shape[0],
                    target_shape[1],
                    target_shape[2],
                    dtype=torch.float32,
                    device=self.device,
                    generator=seed_g,
                )
            ]
        
            max_seq_len = int(math.ceil(np.prod(target_shape) // 4 / self.sp_size)) * self.sp_size
            if max_seq_len % self.sp_size != 0:
                raise ValueError(f"max_seq_len {max_seq_len} is not divisible by sp_size {self.sp_size}")

            with (
                torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=True),
                torch.no_grad()
            ):
                euler_sigmas = None
                if sample_solver == 'euler':
                    # MPS port: plain flow-matching Euler (used by the distilled
                    # Wan models; stable on MPS where multistep solvers drifted).
                    euler_sigmas = list(get_sampling_sigmas(sampling_steps, shift)) + [0.0]
                    timesteps = torch.tensor(euler_sigmas[:-1], dtype=torch.float32,
                                             device=self.device) * self.num_train_timesteps
                elif sample_solver == 'unipc':
                    sample_scheduler = FlowUniPCMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        use_dynamic_shifting=False)
                    sample_scheduler.set_timesteps(
                        sampling_steps, device=self.device, shift=shift)
                    timesteps = sample_scheduler.timesteps
                elif sample_solver == 'dpm++':
                    sample_scheduler = FlowDPMSolverMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        use_dynamic_shifting=False)
                    sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                    timesteps, _ = retrieve_timesteps(
                        sample_scheduler,
                        device=self.device,
                        sigmas=sampling_sigmas)
                else:
                    raise NotImplementedError("Unsupported solver.")

                latents = noise

                pose_latents_no_ref =  self.vae.encode(conditioning_pixel_values.to(torch.bfloat16))
                pose_latents_no_ref = torch.stack(pose_latents_no_ref)
                pose_latents = torch.cat([pose_latents_no_ref], dim=2)

                ref_pixel_values = rearrange(ref_pixel_values, "t c h w -> 1 c t h w")
                ref_latents =  self.vae.encode(ref_pixel_values.to(torch.bfloat16))
                ref_latents = torch.stack(ref_latents)

                mask_ref = self.get_i2v_mask(1, lat_h, lat_w, 1, device=self.device)
                y_ref = torch.concat([mask_ref, ref_latents[0]]).to(dtype=torch.bfloat16, device=self.device)

                img = ref_pixel_values[0, :, 0]
                clip_context = self.clip.visual([img[:, None, :, :]]).to(dtype=torch.bfloat16, device=self.device)

                if mask_reft_len > 0:
                    if replace_flag:
                        bg_pixel_values = batch["bg_pixel_values"]
                        y_reft = self.vae.encode(
                            [
                                torch.concat([refer_t_pixel_values[0, :, :mask_reft_len], bg_pixel_values[0, :, mask_reft_len:]], dim=1).to(self.device)
                            ]
                        )[0]
                        mask_pixel_values = 1 - batch["mask_pixel_values"]
                        mask_pixel_values = rearrange(mask_pixel_values, "b t c h w -> (b t) c h w")
                        mask_pixel_values = F.interpolate(mask_pixel_values, size=(H//8, W//8), mode='nearest')
                        mask_pixel_values = rearrange(mask_pixel_values, "(b t) c h w -> b t c h w", b=1)[:,:,0]
                        msk_reft = self.get_i2v_mask(lat_t, lat_h, lat_w, mask_reft_len, mask_pixel_values=mask_pixel_values, device=self.device)
                    else:
                        y_reft = self.vae.encode(
                            [
                                torch.concat(
                                    [
                                        torch.nn.functional.interpolate(refer_t_pixel_values[0, :, :mask_reft_len].cpu(),
                                                                        size=(H, W), mode="bicubic"),
                                        torch.zeros(3, T - mask_reft_len, H, W),
                                    ],
                                    dim=1,
                                ).to(self.device)
                            ]
                        )[0]
                        msk_reft = self.get_i2v_mask(lat_t, lat_h, lat_w, mask_reft_len, device=self.device)
                else:
                    if replace_flag:
                        bg_pixel_values = batch["bg_pixel_values"]
                        mask_pixel_values = 1 - batch["mask_pixel_values"]
                        mask_pixel_values = rearrange(mask_pixel_values, "b t c h w -> (b t) c h w")
                        mask_pixel_values = F.interpolate(mask_pixel_values, size=(H//8, W//8), mode='nearest')
                        mask_pixel_values = rearrange(mask_pixel_values, "(b t) c h w -> b t c h w", b=1)[:,:,0]
                        y_reft = self.vae.encode(
                            [
                                torch.concat(
                                    [
                                        bg_pixel_values[0],
                                    ],
                                    dim=1,
                                ).to(self.device)
                            ]
                        )[0]
                        msk_reft = self.get_i2v_mask(lat_t, lat_h, lat_w, mask_reft_len, mask_pixel_values=mask_pixel_values, device=self.device)
                    else:
                        y_reft = self.vae.encode(
                            [
                                torch.concat(
                                    [
                                        torch.zeros(3, T - mask_reft_len, H, W),
                                    ],
                                    dim=1,
                                ).to(self.device)
                            ]
                        )[0]
                        msk_reft = self.get_i2v_mask(lat_t, lat_h, lat_w, mask_reft_len, device=self.device)

                y_reft = torch.concat([msk_reft, y_reft]).to(dtype=torch.bfloat16, device=self.device)
                y = torch.concat([y_ref, y_reft], dim=1)

                arg_c = {
                    "context": context, 
                    "seq_len": max_seq_len,
                    "clip_fea": clip_context.to(dtype=torch.bfloat16, device=self.device),
                    "y": [y],
                    "pose_latents": pose_latents,
                    "face_pixel_values": face_pixel_values,
                }

                if guide_scale > 1:
                    face_pixel_values_uncond = face_pixel_values * 0 - 1
                    arg_null = {
                        "context": context_null,
                        "seq_len": max_seq_len,
                        "clip_fea": clip_context.to(dtype=torch.bfloat16, device=self.device),
                        "y": [y],
                        "pose_latents": pose_latents,
                        "face_pixel_values": face_pixel_values_uncond,
                    }

                for i, t in enumerate(tqdm(timesteps)):
                    latent_model_input = latents
                    timestep = [t]

                    timestep = torch.stack(timestep)

                    noise_pred_cond = TensorList(
                         self.noise_model(TensorList(latent_model_input), t=timestep, **arg_c)
                    )

                    if guide_scale > 1:
                        noise_pred_uncond = TensorList(
                             self.noise_model(
                                TensorList(latent_model_input), t=timestep, **arg_null
                            )
                        )
                        noise_pred = noise_pred_uncond + guide_scale * (
                            noise_pred_cond - noise_pred_uncond
                        )
                    else:
                        noise_pred = noise_pred_cond

                    if euler_sigmas is not None:
                        dt = euler_sigmas[i] - euler_sigmas[i + 1]
                        latents[0] = (latents[0].float() - dt * noise_pred[0].float())
                    else:
                        temp_x0 = sample_scheduler.step(
                            noise_pred[0].unsqueeze(0),
                            t,
                            latents[0].unsqueeze(0),
                            return_dict=False,
                            generator=seed_g,
                        )[0]
                        latents[0] = temp_x0.squeeze(0)

                    x0 = latents
                    if progress is not None:
                        progress(i + 1, len(timesteps), clip_idx, n_clips)

                x0 = [x.to(dtype=torch.float32) for x in x0]
                out_frames = torch.stack(self.vae.decode([x0[0][:, 1:]]))

                if start != 0:
                    out_frames = out_frames[:, :, refert_num:]

                all_out_frames.append(out_frames.cpu())

                start += clip_len - refert_num
                end += clip_len - refert_num
                clip_idx += 1
                if self.device.type == 'mps':
                    torch.mps.empty_cache()

        videos = torch.cat(all_out_frames, dim=2)[:, :, :real_frame_len]
        return videos[0] if self.rank == 0 else None
