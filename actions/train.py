import torch as tr
from torch.utils.data import DataLoader
from pathlib import Path
import os
import json
from tqdm import tqdm

from actions.config import load_config
from sincfold.model import SincFold
from sincfold.dataset import SeqDataset, pad_batch
from sincfold.utils import set_seed

def train_attention(**kwargs):
    
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
    config_file = out_dir / "exp_config.json"

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
        try:
            checkpoints = [f for f in out_dir.iterdir() if f.is_file() and "checkpoint_epoch" in f.name]
            last_model = max(checkpoints, key=lambda x: int(x.stem.split("_")[-1])) if checkpoints else None
            attention_weights_path = last_model
            if attention_weights_path is None:
                raise FileNotFoundError(f"No checkpoint files found in {out_dir}")
        except Exception as e:
            raise FileNotFoundError(f"Attention weights not found in {out_dir}")
    
        # load the configurations of last training cycles
        with open(config_file, "r") as f:
            kwargs = json.load(f)
            if total_epochs <= kwargs['epochs']:
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

    model = SincFold(device=device, verbose=verbose, **kwargs)
    if continue_training:
        continue_epoch, last_loss = model.load_checkpoint(attention_weights_path)
    
    with open(config_file, "w") as f:
        json.dump(kwargs, f, indent=4, sort_keys=True)

    # Epochs
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
            model.optimizer.zero_grad()
            y_pred = model(batch)
            loss = model.loss_func(y_pred, y)
            train_loss += loss#.item()
            loss.backward()
            #tr.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # gradient clipping to avoid exploding gradients
            model.optimizer.step()
        
        train_loss /= len(train_loader)
        
        val_loss = None
        val_f1 = None
        if valid_loader:
            metrics = model.test(valid_loader)
            val_loss = metrics["loss"]
            val_f1 = metrics["f1"]
        
        # gradient clipping
        #tr.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if valid_loader:
            model.scheduler.step(val_loss)   # valid loss
        else:
            model.scheduler.step(train_loss)   # mean epoch loss
        
        if epoch == 0:
            with open(logfile, "w") as f:
                header = "epoch,train_loss"
                if val_loss is not None:
                    header += ",val_loss,val_f1"
                header += ",grad_norm,lr"
                f.write(header + "\n")
                f.flush()
        
        with open(logfile, "a") as f:
            # log loss and validation metrics if available
            line = f"{epoch},{train_loss.item():.6f}"
            if val_loss is not None:
                line += f",{val_loss.item():.6f},{val_f1.item():.6f}"
            f.write(line)
            f.flush()

            # log gradients
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            f.write(f",{total_norm:.6f}")

            # log learning rate
            current_lr = model.optimizer.param_groups[0]['lr']
            f.write(f",{current_lr}\n")
        
        if verbose:
            if val_loss is not None:
                print(f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, val_f1: {val_f1:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f}")
    
    # save model state
    model.save_checkpoint(epochs, train_loss, out_dir / f"checkpoint_epoch_{epochs}.pmt")
    
    
    return model
