import os
import torch

def save_finetuned_model(model, save_directory, best=False, swin_b=False):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory, exist_ok=True)

    if swin_b:
         model_save_path = os.path.join(save_directory, "swin_b_finetuned_model.pth")
    else:
        model_save_path = os.path.join(save_directory, "timesformer_finetuned_best_model.pth" if best else "timesformer_finetuned_model.pth")
    
    torch.save(model.state_dict(), model_save_path)
    print(f"\nModel saved to {model_save_path}")