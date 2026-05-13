import json
import torch as tr

def load_config(config_file=None):
    config = {
        "device": "cuda" if tr.cuda.is_available() else "cpu",
        "batch_size": 4,
        "max_len": 512,
        "verbose": True,
    }
    if config_file and os.path.exists(config_file):
        config.update(json.load(open(config_file)))
    return config
