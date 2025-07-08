import argparse
import os
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
import wandb
from datetime import datetime
from tqdm.auto import tqdm
import bitsandbytes as bnb # Import bitsandbytes

# Import our custom classes
from src.data_preprocessing.viton_hd_dataset import VitonHDDataset
from src.model.tryon_model import VTOModel

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a Virtual Try-On model.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the root data directory.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size PER STEP. Effective batch size will be larger with accumulation.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--wandb_project", type=str, default="vto-system", help="Weights & Biases project name.")
    parser.add_argument("--gdrive_checkpoints_dir", type=str, required=True, help="Path to Google Drive folder for saving checkpoints.")
    # --- NEW ARGUMENT ---
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of steps to accumulate gradients before updating.")
    return parser.parse_args()

def main():
    """Main training script."""
    args = parse_args()
    
    # --- 1. Setup Environment and Paths ---
    # Configure the accelerator for gradient accumulation
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    device = accelerator.device

    run_id = f"vto_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # ... (rest of path setup is the same)
    root_data_dir = args.data_dir
    checkpoints_path = os.path.join(args.gdrive_checkpoints_dir, run_id)
    os.makedirs(checkpoints_path, exist_ok=True)
    
    print(f"--- VTO Training Initialized on {device} ---")
    
    # --- 2. W&B Init ---
    if accelerator.is_main_process:
        wandb.init(project=args.wandb_project, name=run_id, config=args)
        print("Weights & Biases initialized successfully.")

    # --- 3. Dataloading ---
    test_data_img_dir = os.path.join(root_data_dir, 'test')
    pairs_file_full_path = os.path.join(root_data_dir, 'test_pairs.txt')
    dataset = VitonHDDataset(data_dir=test_data_img_dir, pairs_file_path=pairs_file_full_path, image_size=(512, 384))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    print(f"Successfully loaded {len(dataset)} samples.")

    # --- 4. Model and Optimizer Setup ---
    print("Loading VTOModel...")
    model = VTOModel()
    
    # --- USE 8-BIT ADAM OPTIMIZER ---
    trainable_params = list(model.controlnet_pose.parameters()) + list(model.controlnet_cloth.parameters())
    optimizer = bnb.optim.AdamW8bit(trainable_params, lr=args.learning_rate)
    
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloaloader)
    print("Model, optimizer, and dataloader prepared with Accelerator.")

    # --- 5. Training Loop (UPDATED FOR GRADIENT ACCUMULATION) ---
    print("\n--- Starting Training ---")
    progress_bar = tqdm(range(args.epochs * len(dataloader)), disable=not accelerator.is_main_process)
    
    for epoch in range(args.epochs):
        model.train()
        for step, batch in enumerate(dataloader):
            # The with block handles the accumulation logic automatically
            with accelerator.accumulate(model):
                loss = model(batch)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.is_main_process:
                progress_bar.update(1)
                progress_bar.set_description(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")
                if wandb.run:
                    wandb.log({"loss": loss.item()})
        
        # --- 6. Save Checkpoint ---
        if accelerator.is_main_process and (epoch + 1) % 2 == 0:
            save_path = os.path.join(checkpoints_path, f"epoch_{epoch+1}.pth")
            unwrapped_model = accelerator.unwrap_model(model)
            state_to_save = {
                'controlnet_pose': unwrapped_model.controlnet_pose.state_dict(),
                'controlnet_cloth': unwrapped_model.controlnet_cloth.state_dict(),
            }
            torch.save(state_to_save, save_path)
            print(f"\n✅ Checkpoint saved to {save_path}")

    print("\n--- Training Complete ---")
    if accelerator.is_main_process and wandb.run:
        wandb.finish()

if __name__ == "__main__":
    main()