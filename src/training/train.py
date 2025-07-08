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
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the root data directory (e.g., /content/dataset_unzipped/).")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training. Use 1 for limited VRAM.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--wandb_project", type=str, default="vto-system", help="Weights & Biases project name.")
    parser.add_argument("--gdrive_checkpoints_dir", type=str, required=True, help="Path to Google Drive folder for saving checkpoints.")
    return parser.parse_args()

def main():
    """Main training script."""
    args = parse_args()
    accelerator = Accelerator()
    device = accelerator.device
    run_id = f"vto_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # This path is now the parent of 'test', e.g., /content/dataset_unzipped/
    root_data_dir = args.data_dir 
    
    checkpoints_path = os.path.join(args.gdrive_checkpoints_dir, run_id)
    os.makedirs(checkpoints_path, exist_ok=True)
    
    print(f"--- VTO Training Initialized on {device} ---")
    print(f"Run ID: {run_id}")
    print(f"Checkpoints will be saved to: {checkpoints_path}")

    if accelerator.is_main_process:
        try:
            wandb.init(project=args.wandb_project, name=run_id, config=args)
            print("Weights & Biases initialized successfully.")
        except Exception as e:
            print(f"Could not initialize W&B: {e}.")

    # --- 3. Dataloading (UPDATED) ---
    print("\n--- Phase: Dataloading ---")
    
    # Explicitly define the correct paths
    test_data_img_dir = os.path.join(root_data_dir, 'test')
    pairs_file_full_path = os.path.join(root_data_dir, 'test_pairs.txt')
    
    if not os.path.exists(pairs_file_full_path):
        raise FileNotFoundError(f"CRITICAL: The pairs file was not found at {pairs_file_full_path}.")
    if not os.path.isdir(test_data_img_dir):
        raise FileNotFoundError(f"CRITICAL: The image directory was not found at {test_data_img_dir}.")

    dataset = VitonHDDataset(data_dir=test_data_img_dir, pairs_file_path=pairs_file_full_path, image_size=(512, 384))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    print(f"Successfully loaded {len(dataset)} samples.")

    # --- 4. Model, Optimizer, and Scheduler Setup ---
    print("Loading VTOModel...")
    model = VTOModel()
    trainable_params = list(model.controlnet_pose.parameters()) + list(model.controlnet_cloth.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    print("Model, optimizer, and dataloader prepared with Accelerator.")

    # --- 5. Training Loop ---
    print("\n--- Starting Training ---")
    progress_bar = tqdm(range(args.epochs * len(dataloader)), disable=not accelerator.is_main_process)
    
    for epoch in range(args.epochs):
        model.train()
        for step, batch in enumerate(dataloader):
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
        if accelerator.is_main_process:
            if (epoch + 1) % 2 == 0:
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