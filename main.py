import sys, os, json, random
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import torch
import torch.distributed as dist
from transformers import TimesformerForVideoClassification, TimesformerConfig

from utils.save_model import save_pretrained_model
from utils.load_model import load_model
from utils.save_finetuned_model import save_finetuned_model
from video_dataset import train_dataset, val_dataset, test_dataset
from train_model import train_model
from evaluate_model import evaluate_model

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

def configure_parameters(model):
    for param in model.parameters():
        param.requires_grad = False
    
    for i in [6, 7, 8, 9, 10, 11]: 
        for param in model.timesformer.encoder.layer[i].parameters():
            param.requires_grad = True
    model.timesformer.layernorm.weight.requires_grad = True
    model.timesformer.layernorm.bias.requires_grad = True
    model.classifier.weight.requires_grad = True
    model.classifier.bias.requires_grad = True

    return model

if __name__ == "__main__":
    print("Options:")
    print("  1 - Download the pretrained model from transformers library")
    print("  2 - Fine-tune the model on custom dataset")
    print("  3 - Evaluate the model on the test set")

    choice = input("Enter your choice (1, 2, 3): ")
    if choice not in ["1", "2", "3"]:
        print(f"Invalid command: {choice}. Please enter 1, 2, 3.")
        sys.exit(1)

    if choice == "1":
        save_pretrained_model(
            model_name="facebook/timesformer-base-finetuned-k600",
            save_directory="./model"
            )

    elif choice == "2":
        seed_everything()

        base_model = None
        resume_path = None
        num_epochs = 5  
        warmup_epochs = 1
        
        resume_choice = input("Do you want to resume training from a checkpoint? (y/n, default: n): ").strip().lower()
        if resume_choice in ["y", "yes"]:
            target_epoch = int(input("Enter the epoch number to resume from (e.g., 3): ").strip())
            checkpoint_folders = sorted([f for f in os.listdir("./ckpt") if f.startswith("checkpoint-")], 
                                        key=lambda x: int(x.split("-")[1]))
            for folder in checkpoint_folders:
                trainer_state_path = os.path.join("./ckpt", folder, "trainer_state.json")
                if os.path.exists(trainer_state_path):
                    with open(trainer_state_path, 'r') as f:
                        trainer_state = json.load(f)
                    current_epoch = int(trainer_state.get("epoch", 0))
                    if current_epoch == target_epoch:
                        resume_path = os.path.join("./ckpt", folder)
                        print(f"Resuming from epoch {target_epoch} at {resume_path}")
                        break

            if resume_path:
                base_model = TimesformerForVideoClassification.from_pretrained(resume_path)
                # config = TimesformerConfig.from_pretrained(resume_path)
                # base_model = TimesformerForVideoClassification(config)
                base_model = configure_parameters(base_model)

                while True:
                    try:
                        total_epochs = int(input(f"Enter total epochs (including current {target_epoch}): "))
                        if total_epochs <= target_epoch:
                                print(f"Total epochs must be greater than {target_epoch}.")
                        else:
                            remaining_epochs = total_epochs - target_epoch
                            num_epochs = remaining_epochs
                            break
                    except ValueError:
                            print("Invalid input. Enter a number.")
            else:
                print(f"No checkpoint for epoch {target_epoch}")
                resume_choice = "n"

        if not resume_path:
            base_model = load_model("./model")
            if not base_model:
                print("Failed to load model")
                sys.exit(1)
            base_model = configure_parameters(base_model)

            try:
                num_epochs = int(input("Training epochs (default 5): ") or 5)
            except ValueError:
                print("Invalid input, using default 5")
                num_epochs = 5

        while True:
            try:
                warmup_input = input(f"Enter number of warmup epochs (default: 1, max: {int(num_epochs)}): ").strip()
                warmup_epochs = int(warmup_input) if warmup_input else 1
                if warmup_epochs < 0:
                    print("Warmup epochs cannot be negative. Using default 1.")
                    warmup_epochs = 1
                    break
                elif warmup_epochs > num_epochs:
                    print(f"Warmup epochs cannot exceed total epochs. Setting to {int(num_epochs)}.")
                    warmup_epochs = int(num_epochs)
                    break
                else:
                    break
            except ValueError:
                print("Invalid input. Please enter an integer.")

        trained_model = train_model(
            base_model, 
            train_dataset,
            val_dataset,
            num_epochs=num_epochs,
            warmup_epochs=warmup_epochs,
            resume_from_checkpoint=resume_path
            )
        
        save_finetuned_model(trained_model, "./weights")
        print("Training complete. Model saved.")

    elif choice == "3":
        model = TimesformerForVideoClassification.from_pretrained("./weights/best_model")
        evaluate_model(model, test_dataset, batch_size=8)
    else:
        if dist.get_rank() == 0:
            print(f"Invalid command: {choice}. Please enter 1, 2, 3.")
        sys.exit(1)