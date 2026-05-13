import matplotlib.pyplot as plt
import pandas as pd

trial = 74
arch = 7

# Logfile for attention only training
file1 = f"Architecture{arch}/output_trial_{trial}/train_log_attention.csv" # change to train_log.csv if file not found

df = pd.read_csv(file1)
# plot with points and lines, with grid
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['train_loss'], marker='o', label="Attention pretraining")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve - Pretraining Attention Layer')
# x ticks integers
plt.xticks(df['epoch'].astype(int))
plt.grid(True)
plt.savefig(f"Architecture{arch}/output_trial_{trial}/training_curve_attn.png")

# Log file for convolution training
file2 = f"Architecture{arch}/output_trial_{trial}/train_log_conv.csv"

df = pd.read_csv(file2)
# plot with points and lines, with grid
plt.plot(df['epoch'], df['train_loss'], marker='o', label="Convolutional training")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curves')
# x ticks integers
plt.xticks(df['epoch'].astype(int))
plt.grid(True)
plt.legend()
plt.savefig(f"Architecture{arch}/output_trial_{trial}/training_curve-conv.png")
