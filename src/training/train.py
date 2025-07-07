import argparse
import os
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
import wandb
from datetime import datetime
from tqdm.auto import tqdm

# Import our custom classes
from src.data_preprocessing.viton_hd_dataset import VitonHDDataset
from src.model.tryon_model import VTOModel

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a Virtual Try-On model.")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        required=True, 
        help="Path to the root data directory."
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=10, 
        help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1, 
        help="Batch size for training. Use 1 for limited VRAM."
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=1e-5, 
        help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--wandb_project", 
        type=str, 
        default="vto-system", 
        help="Weights & Biases project name."
    )
    parser.add_argument(
        "--gdrive_checkpoints_dir", 
        type=str, 
        required=True, 
        help="Path to Google Drive folder for saving checkpoints."
    )
    return parser.parse_args()

def main():
    """Main training script."""
    args = parse_args()

    # --- 1. Setup Environment and Paths ---
    # Use Accelerator for easier mixed-precision and device handling
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device

    run_id = f"vto_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    test_data_dir = os.path.join(args.data_dir, 'test')
    
    if not os.path.isdir(test_data_dir):
        raise FileNotFoundError(f"Test data directory not found at {test_data_dir}.")
        
    checkpoints_path = os.path.join(args.gdrive_checkpoints_dir, run_id)
    os.makedirs(checkpoints_path, exist_ok=True)
    
    print(f"--- VTO Training Initialized on {device} ---")
    print(f"Run ID: {run_id}")
    print(f"Checkpoints will be saved to: {checkpoints_path}")

    # --- 2. Initialize Weights & Biases (W&B) ---
    if accelerator.is_main_process:
        try:
            wandb.init(project=args.wandb_project, name=run_id, config=args)
            print("Weights & Biases initialized successfully.")
        except Exception as e:
            print(f"Could not initialize W&B: {e}.")

    # --- 3. Dataloading ---
    dataset = VitonHDDataset(data_dir=test_data_dir, pairs_file='test_pairs.txt', image_size=(512, 384))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    print(f"Successfully loaded {len(dataset)} samples.")

    # --- 4. Model, Optimizer, and Scheduler Setup ---
    print("Loading VTOModel...")
    model = VTOModel()
    
    # We are only training the ControlNet parameters
    trainable_params = list(model.controlnet_pose.parameters()) + list(model.controlnet_cloth.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    # Prepare everything with accelerator
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    print("Model, optimizer, and dataloader prepared with Accelerator.")


    # --- 5. Training Loop ---
    print("\n--- Starting Training ---")
    progress_bar = tqdm(range(args.epochs * len(dataloader)), disable=not accelerator.is_main_process)
    
    for epoch in range(args.epochs):
        model.train()
        for step, batch in enumerate(dataloader):
            # The model's forward pass calculates the loss internally
            loss = model(batch)
            
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            # Logging
            if accelerator.is_main_process:
                progress_bar.update(1)
                progress_bar.set_description(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")
                if wandb.run:
                    wandb.log({"loss": loss.item()})
        
        # --- 6. Save Checkpoint ---
        if accelerator.is_main_process:
            if (epoch + 1) % 2 == 0: # Save every 2 epochs
                save_path = os.path.join(checkpoints_path, f"epoch_{epoch+1}.pth")
                
                # Unwrap the model to save the raw state dict
                unwrapped_model = accelerator.unwrap_model(model)
                
                # We only need to save the trainable parts (the ControlNets)
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