# Adpated from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/pipeline_stable_video_diffusion.py
import inspect
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union
import copy
import numpy as np
import PIL.Image
import torch
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from diffusers import AutoencoderKLTemporalDecoder,  UNetSpatioTemporalConditionModel
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.utils import logging
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import (
        _append_dims,
        tensor2vid,
        _resize_with_antialiasing,
        StableVideoDiffusionPipelineOutput
)
from einops import rearrange
from ..schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
import cv2

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class EVSFrameInterpolationWithNoiseInjectionPipeline(DiffusionPipeline):
    r"""
    Pipeline to generate video from an input image using Stable Video Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKLTemporalDecoder`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        image_encoder ([`~transformers.CLIPVisionModelWithProjection`]):
            Frozen CLIP image-encoder ([laion/CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)).
        unet ([`UNetSpatioTemporalConditionModel`]):
            A `UNetSpatioTemporalConditionModel` to denoise the encoded image latents.
        scheduler ([`EulerDiscreteScheduler`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images.
    """

    model_cpu_offload_seq = "image_encoder->unet->vae"
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        vae: AutoencoderKLTemporalDecoder,
        image_encoder: CLIPVisionModelWithProjection,
        unet: UNetSpatioTemporalConditionModel,
        scheduler: EulerDiscreteScheduler,
        feature_extractor: CLIPImageProcessor,
    ):
        super().__init__()
        
        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        # self.ori_unet = copy.deepcopy(unet)
       
    def _encode_image(
        self,
        image: PipelineImageInput,
        device: Union[str, torch.device],
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
    ) -> torch.FloatTensor:
        dtype = next(self.image_encoder.parameters()).dtype

        # if not isinstance(image, torch.Tensor):
        #     image = self.image_processor.pil_to_numpy(image)
        #     image = self.image_processor.numpy_to_pt(image)

        #     # We normalize the image before resizing to match with the original implementation.
        #     # Then we unnormalize it after resizing.
        #     image = image * 2.0 - 1.0
        
        image = _resize_with_antialiasing(image, (224, 224))
        image = (image + 1.0) / 2.0

        # Normalize the image with for CLIP input
        image = self.feature_extractor(
            images=image,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        ).pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeddings = self.image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            negative_image_embeddings = torch.zeros_like(image_embeddings)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])

        return image_embeddings

    def _encode_vae_image(
        self,
        image: torch.Tensor,
        device: Union[str, torch.device],
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
    ):
        image = image.to(device=device)
        image_latents = self.vae.encode(image).latent_dist.mode()

        if do_classifier_free_guidance:
            negative_image_latents = torch.zeros_like(image_latents)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_latents = torch.cat([negative_image_latents, image_latents])

        # duplicate image_latents for each generation per prompt, using mps friendly method
        image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1)

        return image_latents

    def _get_add_time_ids(
        self,
        fps: int,
        motion_bucket_id: int,
        noise_aug_strength: float,
        dtype: torch.dtype,
        batch_size: int,
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

        passed_add_embed_dim = self.unet.config.addition_time_embed_dim * len(add_time_ids)
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)

        if do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids])

        return add_time_ids

    def decode_latents(self, latents: torch.FloatTensor, num_frames: int, decode_chunk_size: int = 14):
        # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
        latents = latents.flatten(0, 1)

        latents = 1 / self.vae.config.scaling_factor * latents

        forward_vae_fn = self.vae._orig_mod.forward if is_compiled_module(self.vae) else self.vae.forward
        accepts_num_frames = "num_frames" in set(inspect.signature(forward_vae_fn).parameters.keys())

        # decode decode_chunk_size frames at a time to avoid OOM
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i : i + decode_chunk_size].shape[0]
            decode_kwargs = {}
            if accepts_num_frames:
                # we only pass num_frames_in if it's expected
                decode_kwargs["num_frames"] = num_frames_in

            frame = self.vae.decode(latents[i : i + decode_chunk_size], **decode_kwargs).sample
            frames.append(frame)
        frames = torch.cat(frames, dim=0)

        # [batch*frames, channels, height, width] -> [batch, channels, frames, height, width]
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        frames = frames.float()
        return frames

    def check_inputs(self, image, height, width):
        if (
            not isinstance(image, torch.Tensor)
            and not isinstance(image, PIL.Image.Image)
            and not isinstance(image, list)
        ):
            raise ValueError(
                "`image` has to be of type `torch.FloatTensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(image)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    def prepare_latents(
        self,
        batch_size: int,
        num_frames: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: Union[str, torch.device],
        generator: torch.Generator,
        latents: Optional[torch.FloatTensor] = None,
    ):
        shape = (
            batch_size,
            num_frames,
            num_channels_latents // 2,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        if isinstance(self.guidance_scale, (int, float)):
            return self.guidance_scale > 1
        return self.guidance_scale.max() > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps
    

    @torch.no_grad()
    def multidiffusion_step(self, latents, t, 
                    image1_embeddings, 
                    image2_embeddings, 
                    image1_latents,
                    image2_latents,
                    added_time_ids, 
                    avg_weight
    ):
        # expand the latents if we are doing classifier free guidance
        # latents1 = latents
        # latents2 = torch.flip(latents, (1,))
        latents1 = torch.flip(latents, (1,))
        latents2 = latents

        # latent_model_input1 = torch.cat([latents1] * 2) if self.do_classifier_free_guidance else latents1
        # latent_model_input1 = self.scheduler.scale_model_input(latent_model_input1, t)

        latent_model_input2 = torch.cat([latents2] * 2) if self.do_classifier_free_guidance else latents2
        latent_model_input2= self.scheduler.scale_model_input(latent_model_input2, t)


        # Concatenate image_latents over channels dimention
        # latent_model_input1 = torch.cat([latent_model_input1, image1_latents], dim=2)
        latent_model_input2 = torch.cat([latent_model_input2, image2_latents], dim=2)

        # predict the noise residual
        # noise_pred1 = self.ori_unet(
        #     latent_model_input1,
        #     t,
        #     encoder_hidden_states=image1_embeddings,
        #     added_time_ids=added_time_ids,
        #     return_dict=False,
        # )[0]
        
        
        image1_embeddings = image1_embeddings.to(image2_embeddings.dtype)
        encoder_hidden_states = ((image2_embeddings + image1_embeddings) / 2).to(image2_embeddings.dtype)
        
        # encoder_hidden_states = self.prepare_clip_feature(image2_embeddings,image1_embeddings)

        noise_pred2 = self.unet(
            latent_model_input2,
            t,
            encoder_hidden_states= encoder_hidden_states,
            added_time_ids=added_time_ids,
            return_dict=False,
        )[0]
        # perform guidance
        if self.do_classifier_free_guidance:
            # noise_pred_uncond1, noise_pred_cond1 = noise_pred1.chunk(2)
            # noise_pred1 = noise_pred_uncond1 + self.guidance_scale * (noise_pred_cond1 - noise_pred_uncond1)

            noise_pred_uncond2, noise_pred_cond2 = noise_pred2.chunk(2)
            noise_pred2 = noise_pred_uncond2 + self.guidance_scale * (noise_pred_cond2 - noise_pred_uncond2)

        # noise_pred2 = torch.flip(noise_pred2, (1,))
        # noise_pred = avg_weight*noise_pred1+ (1-avg_weight)*noise_pred2

        # noise_pred = noise_pred1
        noise_pred = noise_pred2

        return noise_pred

    def save_image_debug(self, new_condition, name="None"):
        B, T, C, H, W = new_condition.shape
        print("Before:", torch.mean(new_condition),B)
        new_condition = new_condition.cpu().detach().numpy().astype(np.float32)
        print("After:", np.mean(new_condition))
        for t in range(T):
            condition_t = new_condition[1, t, :, :, :]
            print(f"Time step {t}: Min: {np.min(condition_t)}, Max: {np.max(condition_t)}, Mean: {np.mean(condition_t)}")
            min_val = np.min(condition_t)
            max_val = np.max(condition_t)
            if max_val != min_val:
                condition_t = (condition_t - min_val) / (max_val - min_val)  # 归一化到 [0, 1]
            else:
                print(f"Warning: All values are the same at time step {t}, skipping normalization.")
                condition_t = np.zeros_like(condition_t)  # 如果所有值相同，设为零
            condition_t = np.clip(condition_t, 0, 1)
            condition_t = np.transpose(condition_t, (1, 2, 0))
            img = (condition_t ** 0.1) * 255
            img = np.nan_to_num(img[:,:,0:3], nan=0.0)
            print(f"Image mean after processing: {np.mean(img)}")
            img_name = f"/mnt/workspace/zhangziran/DiffEVS/svd_keyframe_interpolation-main/my_results/check_debug/{name}_condition_t_{t}.png"
            cv2.imwrite(img_name, img.astype(np.uint8))  # 保存图像，确保值是 uint8 类型
            print(f"Saved {img_name}")
            

    def prepare_latent_feature(self, conditions_latent, evs_latents, conditions_latent_ref):
        data_dtype = conditions_latent.dtype
        evs_latents = evs_latents.to(data_dtype)
        B, T, C, H, W = conditions_latent.shape
        w_cond = torch.arange(T).float() / (T - 1) 
        w_ref = 1 - torch.arange(T).float() / (T - 1)  
        w_evs = torch.ones(T)

        w_evs[-1] = 0
        w_evs[0] = 0
        w_ref[0] = 2
        w_cond[-1] = 2

        device = conditions_latent.device
        w_cond = w_cond.view(1, T, 1, 1, 1).expand(B, T, C, H, W).to(data_dtype).to(device)
        w_ref = w_ref.view(1, T, 1, 1, 1).expand(B, T, C, H, W).to(data_dtype).to(device)
        w_evs = w_evs.view(1, T, 1, 1, 1).expand(B, T, C, H, W).to(data_dtype).to(device)

        assert conditions_latent.shape == (B, T, C, H, W)
        assert evs_latents.shape == (B, T, C, H, W)
        assert conditions_latent_ref.shape == (B, T, C, H, W)

        alpha = (1.0 / (w_cond + w_evs + w_ref)).to(device)

        new_condition = alpha * (w_cond * conditions_latent + w_evs * evs_latents + w_ref * conditions_latent_ref)
        print("The latent space features are prepared!")

        if False:
            self.save_image_debug(new_condition,"new_condition")
            self.save_image_debug(conditions_latent,"conditions_latent")
            self.save_image_debug(conditions_latent_ref,"conditions_latent_ref")

        return new_condition

    def prepare_clip_feature(self, encoder_hidden_states, encoder_hidden_states_ref, lambda_value = 1.0):
        A_img1 = torch.exp(lambda_value * encoder_hidden_states) / (torch.exp(lambda_value * encoder_hidden_states) + torch.exp(lambda_value * encoder_hidden_states_ref))
        A_img2 = torch.exp(lambda_value * encoder_hidden_states_ref) / (torch.exp(lambda_value * encoder_hidden_states) + torch.exp(lambda_value * encoder_hidden_states_ref))
        encoder_hidden_states_fused = A_img1 * encoder_hidden_states + A_img2 * encoder_hidden_states_ref
        return encoder_hidden_states_fused.to(encoder_hidden_states.dtype)



    @torch.no_grad()
    def __call__(
        self,
        image1: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
        image2: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
        evs: torch.FloatTensor,
        height: int = 576,
        width: int = 1024,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 25,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        weighted_average: bool = False,
        noise_injection_steps: int = 0,
        noise_injection_ratio: float=0.0,
        return_dict: bool = True,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.FloatTensor`):
                Image or images to guide image generation. If you provide a tensor, it needs to be compatible with
                [`CLIPImageProcessor`](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/blob/main/feature_extractor/preprocessor_config.json).
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_frames (`int`, *optional*):
                The number of video frames to generate. Defaults to 14 for `stable-video-diffusion-img2vid` and to 25 for `stable-video-diffusion-img2vid-xt`
            num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter is modulated by `strength`.
            min_guidance_scale (`float`, *optional*, defaults to 1.0):
                The minimum guidance scale. Used for the classifier free guidance with first frame.
            max_guidance_scale (`float`, *optional*, defaults to 3.0):
                The maximum guidance scale. Used for the classifier free guidance with last frame.
            fps (`int`, *optional*, defaults to 7):
                Frames per second. The rate at which the generated images shall be exported to a video after generation.
                Note that Stable Diffusion Video's UNet was micro-conditioned on fps-1 during training.
            motion_bucket_id (`int`, *optional*, defaults to 127):
                The motion bucket ID. Used as conditioning for the generation. The higher the number the more motion will be in the video.
            noise_aug_strength (`float`, *optional*, defaults to 0.02):
                The amount of noise added to the init image, the higher it is the less the video will look like the init image. Increase it for more motion.
            decode_chunk_size (`int`, *optional*):
                The number of frames to decode at a time. The higher the chunk size, the higher the temporal consistency
                between frames, but also the higher the memory consumption. By default, the decoder will decode all frames at once
                for maximal quality. Reduce `decode_chunk_size` to reduce memory usage.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.

        Returns:
            [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list of list with the generated frames.

        Examples:

        ```py
        from diffusers import StableVideoDiffusionPipeline
        from diffusers.utils import load_image, export_to_video

        pipe = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
        pipe.to("cuda")

        image = load_image("https://lh3.googleusercontent.com/y-iFOHfLTwkuQSUegpwDdgKmOjRSTvPxat63dQLB25xkTs4lhIbRUFeNBWZzYf370g=s1200")
        image = image.resize((1024, 576))

        frames = pipe(image, num_frames=25, decode_chunk_size=8).frames[0]
        export_to_video(frames, "generated.mp4", fps=7)
        ```
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames
        
        # if isinstance(image1, torch.Tensor):
        #     print("image1 缩放前范围:", image1.min().item(), image1.max().item())
        #     image1 = (image1 + 1.0) / 2.0  # 假设原始范围是 [-1, 1]
        #     print("image1 缩放后范围:", image1.min().item(), image1.max().item())
        # if isinstance(image2, torch.Tensor):
        #     print("image2 缩放前范围:", image2.min().item(), image2.max().item())
        #     image2 = (image2 + 1.0) / 2.0  # 假设原始范围是 [-1, 1]
        #     print("image2 缩放后范围:", image2.min().item(), image2.max().item())

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(image1, height, width)
        self.check_inputs(image2, height, width)

        # 2. Define call parameters
        if isinstance(image1, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image1, list):
            batch_size = len(image1)
        else:
            batch_size = image1.shape[0]
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        self._guidance_scale = max_guidance_scale

        # 3. Encode input image
        image1_embeddings = self._encode_image(image1, device, num_videos_per_prompt, self.do_classifier_free_guidance)
        image2_embeddings = self._encode_image(image2, device, num_videos_per_prompt, self.do_classifier_free_guidance)

        # NOTE: Stable Diffusion Video was conditioned on fps - 1, which
        # is why it is reduced here.
        # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
        fps = fps - 1

        # 4. Encode input image using VAE
        image1 = self.image_processor.preprocess(image1, height=height, width=width).to(device)
        image2 = self.image_processor.preprocess(image2, height=height, width=width).to(device)
        noise = randn_tensor(image1.shape, generator=generator, device=image1.device, dtype=image1.dtype)
        image1 = image1 + noise_aug_strength * noise
        image2 = image2 + noise_aug_strength * noise

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)


        # Repeat the image latents for each frame so we can concatenate them with the noise
        # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
        image1_latent = self._encode_vae_image(image1, device, num_videos_per_prompt, self.do_classifier_free_guidance)
        image1_latent = image1_latent.to(image1_embeddings.dtype)
        image1_latents = image1_latent.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

        image2_latent = self._encode_vae_image(image2, device, num_videos_per_prompt, self.do_classifier_free_guidance)
        image2_latent = image2_latent.to(image2_embeddings.dtype)
        image2_latents = image2_latent.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
        
        evs_values = rearrange(evs, "b f c h w -> (b f) c h w")
        self.check_inputs(evs_values, height, width)
        evs_latents = self._encode_vae_image(evs_values, device, num_videos_per_prompt, self.do_classifier_free_guidance)

        evs_latents = rearrange(evs_latents, "(b f) c h w -> b f c h w", f=num_frames)

        # image2_latents = image2_latents + evs_latents.to(image2_latents.dtype)
        
        image2_latents = self.prepare_latent_feature(image2_latents,evs_latents,image1_latents)
        

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # 5. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            image1_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            self.do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(device)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents,
            height,
            width,
            image1_embeddings.dtype,
            device,
            generator,
            latents,
        )

        # 7. Prepare guidance scale
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)

        if weighted_average:
            self._guidance_scale = guidance_scale
            w = torch.linspace(1, 0, num_frames).unsqueeze(0).to(device, latents.dtype)
            w = w.repeat(batch_size*num_videos_per_prompt, 1)
            w = _append_dims(w, latents.ndim)
        else:
            self._guidance_scale = (guidance_scale+torch.flip(guidance_scale, (1,)))*0.5
            w = 0.5

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        # self.ori_unet = self.ori_unet.to(device)

        noise_injection_step_threshold = int(num_inference_steps*noise_injection_ratio)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                noise_pred = self.multidiffusion_step(latents, t, 
                    image1_embeddings, image2_embeddings, 
                    image1_latents, image2_latents, added_time_ids, w
                )
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                if i < noise_injection_step_threshold and noise_injection_steps > 0:
                    sigma_t = self.scheduler.sigmas[self.scheduler.step_index]
                    sigma_tm1 = self.scheduler.sigmas[self.scheduler.step_index+1]
                    sigma = torch.sqrt(sigma_t**2-sigma_tm1**2)
                    for j in range(noise_injection_steps):
                        noise = randn_tensor(latents.shape, device=latents.device, dtype=latents.dtype)
                        noise = noise * sigma
                        latents = latents + noise
                        noise_pred = self.multidiffusion_step(latents, t, 
                            image1_embeddings, image2_embeddings, 
                            image1_latents, image2_latents, added_time_ids, w
                        )
                        # compute the previous noisy sample x_t -> x_t-1
                        latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                self.scheduler._step_index += 1

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if not output_type == "latent":
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            frames = self.decode_latents(latents, num_frames, decode_chunk_size)
            frames = tensor2vid(frames, self.image_processor, output_type=output_type)
        else:
            frames = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(frames=frames)
