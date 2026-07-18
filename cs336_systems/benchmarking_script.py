import argparse
import

from argparse import Namespace
from cs366-basics.model import

import torch

configs = {
    "small": {
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12,
        "vocab_size": 10000,
        "context_length": 256,
        "rope_theta": 10000,
    },
}

def parse_args() -> argparse.Namespace:
    parser= argparse.ArgumentParser("Benchmark Model Script")
    parser.add_argument("--mode", type=str, choices=configs.keys(), default="small")
    parser.add_argument(
        "--type",
        type=str,
        choices=["forward", "forward-backward", "forward-backward-optimizer"],
        default="forward-backward-optimizer")
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--benchmark_steps", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--context_window", type=int, default=256)

    return parser.parse_args()

def benchmark(cfg: Namespace) -> None:
    model_configs = configs[cfg.model]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BasicsTransformerLM(

    )