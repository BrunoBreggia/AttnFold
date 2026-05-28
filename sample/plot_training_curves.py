import matplotlib.pyplot as plt
import pandas as pd
import argparse

# Forma de usar el script
# $ python sample/plot_training_curves 111

# Manage parser
parser = argparse.ArgumentParser(description="Creacion de imagen png con las curvas de entrenamiento")
parser.add_argument("exp", help="Numero de experimento")
args = parser.parse_args()

exp = args.exp

# Logfile for attention only training
file1 = f"Experimentos/{exp}/train_log_attention.csv" # change to train_log.csv if file not found

df = pd.read_csv(file1)

# plot with points and lines, with grid
fig, ax1 = plt.subplots()

# Graficar la primera curva en ax1
color1 = 'tab:red'
ax1.set_xlabel('Epocas')
ax1.set_ylabel('Loss', color=color1)
ax1.plot(df['epoch'], df['train_loss'], marker='o', label="train loss", color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.legend()

# Crear un segundo eje (ax2) que comparte el mismo eje x
ax2 = ax1.twinx()
color2 = 'tab:blue'
ax2.set_ylabel('Gradient norm', color=color2)
ax2.plot(df['epoch'], df['grad_norm'], marker='o', label="gradient norm", color=color2)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.legend()

#plt.plot(df['epoch'], df['grad_norm'], marker='o', label="gradient norm")
plt.title('Training Curves')
# x ticks integers
#plt.xticks(df['epoch'].astype(int))
plt.grid(True)
fig.tight_layout() 
#plt.legend()
plt.savefig(f"Experimentos/{exp}/training_curve.png")

# Log file for convolution training
# file2 = f"Experimentos/exp_{exp}/train_log_conv.csv"

# df = pd.read_csv(file2)
# # plot with points and lines, with grid
# plt.plot(df['epoch'], df['train_loss'], marker='o', label="Convolutional training")
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Loss Curves')
# # x ticks integers
# plt.xticks(df['epoch'].astype(int))
# plt.grid(True)
# plt.legend()
# plt.savefig(f"Experimentos/exp_{exp}/training_curve-conv.png")
