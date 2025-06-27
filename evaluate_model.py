import numpy as np
import torch
import torch.nn.functional as F
import evaluate
import wandb
from transformers import Trainer, TrainingArguments

accuracy_metric = evaluate.load("accuracy")
clf_metrics = evaluate.combine(["f1", "precision", "recall"])
roc_auc_metric = evaluate.load("roc_auc")

def compute_metrics(test_pred):
    logits, labels = test_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    clf_results = clf_metrics.compute(predictions=predictions, references=labels, average="binary")
    
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
    probs = F.softmax(logits, dim=1).detach().cpu().numpy()
    auc = roc_auc_metric.compute(prediction_scores=probs[:, 1], references=labels)

    try:
        wandb.log({"roc": wandb.plot.roc_curve(labels, probs, labels=["real", "fake"])})
        wandb.log({"pr": wandb.plot.pr_curve(labels, probs, labels=["real", "fake"])})
        cm = wandb.plot.confusion_matrix(
            y_true=labels, preds=predictions, class_names=["real", "fake"]
        )
        wandb.log({"conf_mat": cm})
    except Exception as e:
        print(f"Warning: Failed to log to wandb: {e}")

    metrics = {
        "accuracy": accuracy["accuracy"],
        "f1": clf_results["f1"],
        "precision": clf_results["precision"],
        "recall": clf_results["recall"],
        "auc": auc["roc_auc"]
    }

    return metrics

def evaluate_model(model, test_dataset, batch_size):
    wandb.init(project="deepfake-detection", name="TALL-TimeSformer-Evaluation")
    print("\nEvaluating on Test Dataset...")

    test_args = TrainingArguments(
        output_dir = "./results/test_results",
        do_train = False,
        do_predict = True,
        per_device_eval_batch_size = batch_size,   
        dataloader_drop_last = False,
        report_to="wandb"
    )

    trainer = Trainer(
        model = model, 
        args = test_args, 
        compute_metrics = compute_metrics
    )

    output = trainer.predict(test_dataset)

    metrics_table = wandb.Table(columns=["Metric", "Value"])
    for k, v in output.metrics.items():
        metrics_table.add_data(k, v)

    wandb.log({"test_metrics": metrics_table})
    
    print("\nTest Set Evaluation Metrics")
    print("=" * 40)
    for k, v in output.metrics.items():
        print(f"{k:<20}: {v:.4f}")
    print("=" * 40)