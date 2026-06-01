import torch as tr
from torch.utils.data import DataLoader
from pathlib import Path
import os
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import json

from actions.config import load_config
from sincfold.model import SincFold
from sincfold.tokenizer import k3_tokenizer

def predict_sequence(sequence, out_dir, batch_size=1):

    # 1. Validate sequence
    sequence = sequence.upper().strip()
    valid_nt = set(['A', 'C', 'G', 'U', 'T'])
    if not set(sequence).issubset(valid_nt):
        raise ValueError(f"Invalid nucleotides in sequence. Valid: A, C, G, U, T")
    
    # Convert T -> U
    sequence = sequence.replace('T', 'U')
    
    if len(sequence) < 3:
        raise ValueError("Sequence too short. Minimum: 3 nucleotides")
    
    # 2. Load config
    config = load_config()
    device = config.pop("device")
    verbose = config.pop("verbose")
    
    # 3. Load model
    logfile = out_dir / "train_log_attention.csv"
    config_file = out_dir / "exp_config.json"

    if not os.path.exists(logfile):
        raise FileNotFoundError(f"Log file not found at {logfile}")
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Log file not found at {logfile}")
    
    with open(config_file, "r") as f:
        kwargs = json.load(f)
    print(kwargs)

    epochs = kwargs['epochs']
    weights_path = out_dir / f"checkpoint_epoch_{epochs}.pmt"
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Attention weights not found at {weights_path}")
    
    config = load_config()

    device = config.pop("device")
    verbose = config.pop("verbose")

    model = SincFold(device=device, verbose=verbose,
                    save_dir=out_dir, **kwargs)
    print(f"Loading model weights from {weights_path}...")
    model.load_checkpoint(weights_path)

    # 4. Tokenize: 3-mer tokenization
    tokens = k3_tokenizer(sequence)  # Tensor of length len//3 with IDs 0-63
    length_k = len(tokens)
    seq_len = len(sequence)
    
    # 5. Create batch [batch=1, length_k]
    embedding_pad = tr.zeros((1, length_k), dtype=tr.int16)
    embedding_pad[0, :length_k] = tokens
    
    batch = {
        "embedding": embedding_pad.to(device),
        "length_k": [length_k],
        "length": [seq_len],
        "canonical_mask": [None],
        "interaction_prior": [None],
        "sequence": [sequence],
        "id": ["pred_0"]
    }
    
    # 6. Predict
    model.eval()
    with tr.no_grad():
        pred_matrix = model(batch)  # [1, seq_len, seq_len]
    
    pred_matrix = pred_matrix.squeeze(0).cpu().numpy()
    
    # 7. Save heatmap
    os.makedirs("resultados", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plt.figure(figsize=(8, 8))
    sns.heatmap(pred_matrix, annot=False, cmap='plasma')
    plt.title(f"Predicted Contact Matrix\nSequence: {sequence[:30]}...")
    plt.xlabel("Position")
    plt.ylabel("Position")
    
    img_path = f"resultados/prediction_{timestamp}.png"
    plt.savefig(img_path, dpi=150)
    plt.close()
    print(f"Heatmap saved to {img_path}")
    
    return pred_matrix
