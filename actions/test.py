import torch as tr
from torch.utils.data import DataLoader
from pathlib import Path
import os
from tqdm import tqdm

from actions.config import load_config
from sincfold.model import SincFold
from sincfold.dataset import SeqDataset, pad_batch

def test_attention_model(
                        #config_file=None,
                        **kwargs
                        ):

    # unpacking
    test_file = kwargs.get('test_file')
    out_dir = Path(kwargs.get('out_dir'))
    nworkers = kwargs.get('nworkers', 2)
    batch_size = kwargs.get('batch_size', 1)

    logfile = out_dir / "train_log_attention.csv"
    config_file = out_dir / "exp_config.txt"

    if not os.path.exists(logfile):
        raise FileNotFoundError(f"Log file not found at {logfile}")
    
    attention_weights_path = out_dir / "weights_attention.pmt"
    if not os.path.exists(attention_weights_path):
        raise FileNotFoundError(f"Attention weights not found at {attention_weights_path}")
    
    config = load_config()

    device = config.pop("device")
    verbose = config.pop("verbose")

    test_loader = DataLoader(
        SeqDataset(test_file, **config),
        batch_size=batch_size,
        shuffle=False,
        num_workers=nworkers,
        collate_fn=pad_batch,
    )

    # Get model configuration from config file
    with open(config_file, "r") as f:
        kwargs = {}
        for line in f:
            k,v = line.strip().split(":")
            kwargs[k.strip()] = v.strip()

    model = SincFold(attention_only=True, device=device, verbose=verbose,
                    save_dir=out_dir, **kwargs)
    print(f"Loading model weights from {attention_weights_path}...")
    model.load_attention(attention_weights_path)
    
    print("Testing model...")
    metrics = model.test(test_loader)
    print(f"Test results:")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")
    
    return metrics
