import json
import pandas as pd
import matplotlib.pyplot as plt

def plot_loss_curve(log_path: str = "logs/metrics_log.json"):
    with open(log_path) as f:
        data = [json.loads(line) for line in f]

    df = pd.DataFrame(data)

    df['epoch'] = df['epoch'].ffill()
    df = df[['epoch', 'loss', 'eval_loss']]

    train_df = df[df['loss'].notna()]
    eval_df = df[df['eval_loss'].notna()]

 
    plt.figure(figsize=(10, 6))

    plt.plot(train_df['epoch'], train_df['loss'], 
            marker='o', linestyle='-', color='blue', label='Training Loss')
    plt.plot(eval_df['epoch'], eval_df['eval_loss'], 
            marker='s', linestyle='--', color='red', label='Evaluation Loss')

    plt.title('Training and Evaluation Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(df['epoch'].unique())

    for x, y in zip(train_df['epoch'], train_df['loss']):
        plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0,5), ha='center')

    for x, y in zip(eval_df['epoch'], eval_df['eval_loss']):
        plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0,-15), ha='center')

    plt.tight_layout()
    plt.savefig('logs/loss_curve.png', dpi=300, bbox_inches='tight')
    print("Loss curve saved to logs.")
        