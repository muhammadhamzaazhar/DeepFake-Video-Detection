import os
import numpy as np
import pandas as pd
import wandb
import torch
import torch.nn.functional as F
import evaluate
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    default_data_collator,
)

from utils.loss_graph import plot_loss_curve

os.environ["WANDB_PROJECT"] = "deepfake-detection"

accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")
roc_auc_metric = evaluate.load("roc_auc")

class EpochProgressCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        current = int(state.epoch) + 1
        total = int(args.num_train_epochs)
        print(f"\n\n>>> Starting epoch {current}/{total}")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(references=labels, predictions=predictions)
    precision = precision_metric.compute(predictions=predictions, references=labels, average="binary")
    recall = recall_metric.compute(references=labels, predictions=predictions, average="binary", zero_division=0)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="binary")
    
    logits_t = torch.from_numpy(logits)
    probs = F.softmax(logits_t.float(), dim=1).cpu().numpy()
    auc = roc_auc_metric.compute(prediction_scores=probs[:, 1], references=labels)

    try:
        wandb.log({"roc": wandb.plot.roc_curve(labels, probs, labels=["real", "fake"])})
        wandb.log({"pr": wandb.plot.pr_curve(labels, probs, labels=["real", "fake"])})
    except Exception as e:
        print(f"Warning: Failed to log to wandb: {e}")

    metrics = {
        "accuracy": accuracy["accuracy"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "f1": f1["f1"],
        "auc": auc["roc_auc"]
    }

    return metrics

def train_model(
        model, 
        train_dataset, 
        val_dataset, 
        num_epochs, 
        warmup_epochs,
        resume_from_checkpoint=None
):
    per_device_batch_size = 8
    total_steps = num_epochs * (len(train_dataset) // per_device_batch_size)
    warmup_steps = warmup_epochs * (len(train_dataset) // per_device_batch_size)

    training_args = TrainingArguments(
        output_dir="./ckpt",
        overwrite_output_dir=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        
        optim="adamw_torch",
        learning_rate=1.5e-5,
        weight_decay=0.01,
        label_smoothing_factor=0.1,
        max_grad_norm=1.0,
        gradient_accumulation_steps=1,
        lr_scheduler_type="cosine",

        num_train_epochs=num_epochs,
        warmup_steps=warmup_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",

        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=4,
        
        fp16=True,
        # deepspeed="ds_config.json",
        disable_tqdm=False,
        report_to='wandb',
        run_name="TALL-TimeSformer-Tesla V100-Dropout(0.2)"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EpochProgressCallback()]
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    try:
        os.makedirs("logs", exist_ok=True)

        logs = trainer.state.log_history
        df = pd.DataFrame(logs)
        df = df.drop(columns=['step'], errors='ignore')
        df.to_json("logs/metrics_log.json", orient="records", lines=True)

        plot_loss_curve()
    except Exception as e:
        print(f"Something went wrong in post-training hooks: {e}")

    model.save_pretrained('./weights/best_model')
    return model