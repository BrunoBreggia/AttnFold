import os
import json
import shutil
import numpy as np
import torch as tr
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from actions.train import train_attention
from actions.test import test_attention_model
from actions.predict import predict_sequence

parser = argparse.ArgumentParser(description="SincFold training and testing")
subparsers = parser.add_subparsers(dest="command", help="Commands")

# parser for training model
parser_attention = subparsers.add_parser("train-attention", help="Train attention layer only")
parser_attention.add_argument("train_file", help="Training data CSV file")
parser_attention.add_argument("out_dir", help="Output directory (will be created)")
parser_attention.add_argument("--continue", action="store_true", help="Continues training the existent model up to the amout of epochs specified")
#parser_attention.add_argument("--valid-file", help="Validation data CSV file")
#parser_attention.add_argument("--config-file", help="JSON config file")
parser_attention.add_argument("--epochs", type=int, default=100, help="Number of epochs")
parser_attention.add_argument("--emb_dim", type=int, default=32, help="Embedding dimension")
parser_attention.add_argument("--pos_enc", type=str, default="absolute", help="Positional encoding")
parser_attention.add_argument("--force_symmetry", action="store_true", help="Forces symmetry in internal matrix prediction")
parser_attention.add_argument("--mask", action="store_true", help="Sets upper triangular mask")
parser_attention.add_argument("--k_bias", action="store_true", help="Sets k bias in attention matrix")
parser_attention.add_argument("--nworkers", type=int, default=2, help="Number of workers")
parser_attention.add_argument("--batch-size", type=int, default=4, help="Batch size")
parser_attention.add_argument("--seed", type=int, default=78, help="Seed for random number generators")

# parser for test mode
parser_test_attention = subparsers.add_parser("test-attention", help="Test attention-only model")
parser_test_attention.add_argument("test_file", help="Test data CSV file")
parser_test_attention.add_argument("out_dir", help="Output directory (must contain weights_attention.pmt)")
#parser_test_attention.add_argument("--config-file", help="JSON config file")
parser_test_attention.add_argument("--nworkers", type=int, default=2, help="Number of workers")
parser_test_attention.add_argument("--batch-size", type=int, default=1, help="Batch size")
#parser_test_attention.add_argument("--pos_enc", type=str, default="abs", help="Positional encoding")
#parser_test_attention.add_argument("--force_symmetry", action="store_true", help="Forces symmetry in internal matrix prediction")

# parser for predicting in-line RNA sequence
parser_predict = subparsers.add_parser("predict-sequence", help="Predict RNA structure from raw sequence")
parser_predict.add_argument("sequence", help="RNA sequence (A, C, G, U/T)")
parser_predict.add_argument("out_dir", help="Output directory (must contain weights file)")
parser_predict.add_argument("--weights-type", choices=['attention', 'full'], default='full',
                        help="Type of weights: 'attention' or 'full'")
parser_predict.add_argument("--config", help="JSON config file")
parser_predict.add_argument("--batch-size", type=int, default=1, help="Batch size")

args = parser.parse_args()

if __name__ == "__main__":
    
    if args.command == "train-attention":
        train_attention(
            **vars(args)
        )
    elif args.command == "test-attention":
        test_attention_model(
            **vars(args)
        )
    elif args.command == "predict-sequence":
        predict_sequence(
            args.sequence,
            args.out_dir,
            weights_type=args.weights_type,
            config_file=args.config,
            batch_size=args.batch_size
        )
    else:
        parser.print_help()
