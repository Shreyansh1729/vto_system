import torch
import torch.nn as nn
from diffusers import DDPMScheduler, UNet2DConditionModel, VaeImageProcessor, AutoencoderKL
from transformers import CLIPVisionModelWithProjection
from diffusers.models.controlnet import ControlNetModel


class VTOModel(nn.Module):
    """
    The main Virtual Try-On model. It orchestrates the VAE, Garment Encoder,
    ControlNets, and the main UNet to generate a try-on image.
    """
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        controlnet_pose_id: str = "lllyasviel/sd-controlnet-openpose",
        clip_model_id: str = "openai/clip-vit-large-patch14"
    ):
        super().__init__()

        # --- 1. Load Pre-trained Components ---
        
        # Main UNet for diffusion
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")

        # VAE for encoding/decoding images to/from latent space
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        
        # Garment encoder (CLIP Vision Model)
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_model_id)
        
        # ControlNets for pose and canny edge guidance
        self.controlnet_pose = ControlNetModel.from_pretrained(controlnet_pose_id)
        # We will create a second ControlNet for the cloth, initialized with the same weights
        # It will learn to extract garment shape information
        self.controlnet_cloth = ControlNetModel.from_pretrained(controlnet_pose_id)

        # --- 2. Freeze Components that we don't want to train initially ---
        self.vae.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.unet.requires_grad_(False) # We will only train the ControlNets at first
        
        # Only the ControlNets will be trained
        self.controlnet_pose.train()
        self.controlnet_cloth.train()

        # --- 3. Set up other necessary components ---
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def forward(self, batch):
        """
        The forward pass for training.
        Takes a batch from the DataLoader and returns the loss.
        """
        # --- 1. Unpack and Prepare Inputs ---
        target_images = batch['target_image']  # The ground truth person image
        cloth_images = batch['cloth_image']
        pose_maps = batch['pose_map']

        # The 'cloth canny' will be the same as the pose map for now
        # In a more advanced version, we'd generate a canny edge map of the cloth
        cloth_canny_maps = pose_maps # Placeholder, but a valid starting point

        # --- 2. Encode Inputs ---
        
        # Encode target images into latents
        latents = self._encode_images(target_images)
        
        # Encode garment images using the CLIP encoder
        # The output shape will be (batch_size, 1, 768) for SDv1.5
        garment_embeds = self.image_encoder(cloth_images).image_embeds.unsqueeze(1)
        
        # --- 3. Diffusion Process ---
        
        # Sample noise to add to the latents
        noise = torch.randn_like(latents)
        
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # --- 4. Get ControlNet Guidance ---
        
        # Get guidance from the pose ControlNet
        down_block_res_samples_pose, mid_block_res_sample_pose = self.controlnet_pose(
            noisy_latents,
            timesteps,
            encoder_hidden_states=garment_embeds,
            controlnet_cond=pose_maps,
            return_dict=False,
        )

        # Get guidance from the cloth ControlNet
        down_block_res_samples_cloth, mid_block_res_sample_cloth = self.controlnet_cloth(
            noisy_latents,
            timesteps,
            encoder_hidden_states=garment_embeds,
            controlnet_cond=cloth_canny_maps, # Using pose map as a stand-in for cloth shape
            return_dict=False,
        )

        # Combine the guidance from both ControlNets (simple addition)
        # More advanced methods could use weighted sums
        down_block_res_samples = [
            p + c for p, c in zip(down_block_res_samples_pose, down_block_res_samples_cloth)
        ]
        mid_block_res_sample = mid_block_res_sample_pose + mid_block_res_sample_cloth

        # --- 5. Predict Noise with Main UNet ---

        # Predict the noise residual
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=garment_embeds,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        ).sample

        # --- 6. Calculate Loss ---
        # The loss is the mean squared error between the predicted noise and the actual noise
        loss = nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        
        return loss

    def _encode_images(self, images):
        """Encodes images from pixel space to VAE latent space."""
        # The posterior is a distribution, we sample from it
        posterior = self.vae.encode(images).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents