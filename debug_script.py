import os
import shutil
#import src.sincfold as sincfold
from main import train_attention, test_attention_model
from src.sincfold.utils import set_seed

set_seed(79) # seed problematica en la epoca 66

command = "train"
target_file = "sample/train_tanda_15a.csv"
out_path = "Architecture7/output_trial_90/"

#if os.path.isdir(out_path):
#    shutil.rmtree(out_path)

config_train = {
    "device": "cuda",
    "batch_size": 4,
    "max_epochs": 20,
    "attention_only": True,
    "max_len": 512,
    "verbose": True,
    "cache_path": "cache/",
    "valid_split": 0.2,
    "patience": 10
}

config_test = {
    "device": "cuda",
    "batch_size": 1,
    "max_len": 512,
    "verbose": True
}

if "train" in command:
    print("=" * 50)
    print("Training attention-only model")
    print("=" * 50)
    train_attention(
        train_file=target_file,
        valid_file=None,
        out_dir=out_path,
        epochs=100
    )
if "test" in command:
    print("\n" + "=" * 50)
    print("Testing model")
    print("=" * 50)
    test_attention_model(
        test_file=target_file,
        out_dir=out_path
    )
