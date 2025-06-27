import sys
import random
import json
import torch
import deepspeed
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.nn as nn

from utils.save_model import save_pretrained_model
from utils.load_model import load_model
from utils.save_finetuned_model import save_finetuned_model
from utils.load_finetuned_model import load_finetuned_model
from video_dataset import train_loader, val_loader, test_loader
from train_model import train_model
from evaluate_model import evaluate_model

class SwinForVideoClassification(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W) 
        outputs = self.base_model(x)
        logits = outputs.logits 
        logits = logits.view(B, T, -1)  
        logits = logits.mean(dim=1)
        return logits

with open('ds_config.json', 'r') as config_file:
    ds_config = json.load(config_file)

if __name__ == "__main__":
    deepspeed.init_distributed()

    choice_tensor = torch.zeros(1, dtype=torch.int)
    if dist.get_rank() == 0:
        print("Options:")
        print("  1 - Download the pretrained model from transformers library")
        print("  2 - Fine-tune the model on custom dataset")
        print("  3 - Evaluate the model on the test set")

        user_input = input("Enter your choice (1, 2, 3): ")
        if user_input not in ["1", "2", "3"]:
            print(f"Invalid command: {user_input}. Please enter 1, 2, 3.")
            sys.exit(1)
        choice_tensor[0] = int(user_input)
    dist.broadcast(choice_tensor, src=0)
    choice = str(choice_tensor.item())

    if choice == "1":
        if dist.get_rank() == 0:
            save_pretrained_model(
                model_name="microsoft/swin-base-patch4-window7-224-in22k",
                save_directory="./model/swin_b",
                swin_b=True
            )

    elif choice == "2":
        seed = 42
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        base_model = load_model("./model/swin_b")
        if not base_model:
            if dist.get_rank() == 0:
                print("Unable to load the model.")
            sys.exit()

        video_model = SwinForVideoClassification(base_model)

        for name, param in video_model.named_parameters():
            if "swin.encoder.layers.0" in name or "swin.encoder.layers.1" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        num_epochs = 10
        per_gpu_batch_size = ds_config["train_micro_batch_size_per_gpu"]

        train_sampler = DistributedSampler(train_loader.dataset, drop_last=True)
        train_loader_ds = DataLoader(
            train_loader.dataset,
            batch_size=per_gpu_batch_size,
            shuffle=False,
            sampler=train_sampler,
            num_workers=train_loader.num_workers,
            pin_memory=train_loader.pin_memory,
            persistent_workers=train_loader.persistent_workers,
            prefetch_factor=train_loader.prefetch_factor
        )

        val_sampler = DistributedSampler(val_loader.dataset, shuffle=False)
        val_loader_ds = DataLoader(
            val_loader.dataset,
            batch_size=per_gpu_batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=val_loader.num_workers,
            pin_memory=val_loader.pin_memory,
            persistent_workers=val_loader.persistent_workers,
            prefetch_factor=val_loader.prefetch_factor
        )

        trained_model = train_model(video_model, train_loader_ds, val_loader_ds, ds_config, num_epochs=num_epochs, swin_b=True)

        if dist.get_rank() == 0:
            save_finetuned_model(trained_model, "./weights/swin_b", swin_b=True)
            print("Training complete. Model saved.")

    elif choice == "3":
        weights_path = r"weights/swin_b/swin_finetuned_model.pth"
        config_path = r"model/swin_b/"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        finetuned_model = load_finetuned_model(weights_path, config_path, device)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for evaluation.")
            model = torch.nn.DataParallel(finetuned_model)

        evaluate_model(model, test_loader, device)

    else:
        if dist.get_rank() == 0:
            print(f"Invalid command: {choice}. Please enter 1, 2, 3.")
        sys.exit(1)