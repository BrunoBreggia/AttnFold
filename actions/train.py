import torch as tr
from torch.utils.data import DataLoader
from pathlib import Path
import os
from tqdm import tqdm

from actions.config import load_config
from sincfold.model import SincFold
from sincfold.dataset import SeqDataset, pad_batch
from sincfold.utils import set_seed

def train_attention(
                    #config_file=None,
                    **kwargs
                    ):
    
    # unpack kwargs
    train_file = kwargs.get('train_file')
    out_dir = Path(kwargs.get('out_dir'))
    valid_file = kwargs.get('valid_file', None)
    epochs = kwargs.get('epochs', 100)
    nworkers = kwargs.get('nworkers', 2)
    batch_size = kwargs.get('batch_size', 4)
    continue_training = kwargs.get('continue', False)
    prev_epochs = 0
    
    # Set seed
    seed = kwargs.get('seed')
    set_seed(seed)

    # File names
    logfile = out_dir / "train_log_attention.csv"
    config_file = out_dir / "exp_config.txt"
    attention_weights_path = out_dir / "weights_attention.pmt"

    # if directory does not exist or do not want to continue last training cycle
    if not os.path.isdir(out_dir) or not continue_training:
        os.makedirs(out_dir, exist_ok=True)
        print("Training from scratch!")
    # if directory exists and continue flag is enabled...
    elif continue_training:
        total_epochs = epochs
        # the config, log and weights files should exist in this directory
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found at {config_file}")
        if not os.path.exists(logfile):
            raise FileNotFoundError(f"Log file not found at {logfile}")
        if not os.path.exists(attention_weights_path):
            raise FileNotFoundError(f"Attention weights not found at {attention_weights_path}")
    
        # load the configurations of last training cycles
        with open(config_file, "r") as f:
            for line in f:
                # override kwargs with file configuration
                k, v = [i.strip() for i in line.strip().split(":")]
                v = int(v) if v.isnumeric() else v
                if k == "epochs":
                    prev_epochs = v
                    continue
                kwargs[k] = v
            if total_epochs <= prev_epochs:
                print(f"Model is already trained with {kwargs['epochs']} epochs")
                print("Will be left as is")
                return ;
            else:
                aditional_epochs = total_epochs - prev_epochs
        print(f"Will continue training for an aditional {aditional_epochs} epochs")

    config = load_config() # load from config file
    device = config.pop("device")
    verbose = config.pop("verbose")

    train_loader = DataLoader(
        SeqDataset(train_file, training=True, **config),
        batch_size=batch_size,
        shuffle=True,
        num_workers=nworkers,
        collate_fn=pad_batch
    )
    
    valid_loader = None
    if valid_file:
        valid_loader = DataLoader(
            SeqDataset(valid_file, **config),
            batch_size=batch_size,
            shuffle=False,
            num_workers=nworkers,
            collate_fn=pad_batch,
        )

    model = SincFold(attention_only=True, device=device, verbose=verbose, **kwargs)
    if continue_training:
        model.load_attention(attention_weights_path)


    # if it exists it will overwrite the config file
    with open(config_file, "w") as f:
        for k,v in kwargs.items():
            print(f"{k:<10}:{v}", file=f)
    
    print(f"Training model for a total of {epochs} epochs...")
    for epoch in range(prev_epochs, epochs):
        model.train()
        train_loss = 0
        if verbose:
            iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        else:
            iterator = train_loader
        
        for batch in iterator:
            y = batch["contact"].to(device)
            batch.pop("contact")
            model.optimizer_attention.zero_grad()
            y_pred = model(batch)
            loss = model.loss_func(y_pred, y)
            train_loss += loss.item()
            loss.backward()
            tr.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # gradient clipping to avoid exploding gradients
            model.optimizer_attention.step()
        
        model.scheduler.step()
        
        train_loss /= len(train_loader)
        
        val_loss = None
        val_f1 = None
        if valid_loader:
            metrics = model.test(valid_loader)
            val_loss = metrics["loss"]
            val_f1 = metrics["f1"]
        
        if epoch == 0:
            with open(logfile, "w") as f:
                header = "epoch,train_loss"
                if val_loss is not None:
                    header += ",val_loss,val_f1"
                header += ",grad_norm"
                f.write(header + "\n")
                f.flush()
        
        with open(logfile, "a") as f:
            # log loss and validation metrics if available
            line = f"{epoch},{train_loss:.6f}"
            if val_loss is not None:
                line += f",{val_loss:.6f},{val_f1:.6f}"
            f.write(line)
            f.flush()

            # log gradients
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            f.write(f",{total_norm:.6f}\n")
        
        if verbose:
            if val_loss is not None:
                print(f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, val_f1: {val_f1:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f}")
    
    model.save_attention(attention_weights_path)
    print(f"Model weights saved to {attention_weights_path}")
    
    return model
