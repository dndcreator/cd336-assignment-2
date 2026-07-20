import argparse
import math
import timeit
from contextlib import nullcontext

import numpy as np
from einops import einsum
from jaxtyping import Float, Bool
from torch import Tensor
from torch.cuda import nvtx

import cs336_basics

from argparse import Namespace
from cs336_basics.model import BasicsTransformerLM

import torch

from cs336_basics.nn_utils import softmax, cross_entropy
from cs336_basics.optimizer import AdamW

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

@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
        Q: Float[Tensor, "... queries d_k"],
        K: Float[Tensor, "... keys d_k"],
        V: Float[Tensor, "... keys d_v"],
        mask: Bool[Tensor, "... queries keys"] | None = None,
) -> Float[Tensor, "... queries d_v"]:
    d_k = K.shape[-1]
    with nvtx.range("computing attention scores"):
        attention_scores = einsum(
            Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)
        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("computing softmax"):
        attention_weights = softmax(
            attention_scores, dim=-1
        )

    with nvtx.range("computing matmul"):
        output = einsum(
            attention_weights, V, "... queries keys, ... keys d_v -> ... queries d_v"
        )

        return output

cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

def get_random_batch(cfg: Namespace) -> torch.Tensor:

    return torch.randint(0, configs[cfg.mode]["vocab_size"], (cfg.batch_size, cfg.context_length))

@nvtx.range("Forward Model")
def forward(cfg: Namespace, model: BasicsTransformerLM, device: torch.device) -> tuple[float, float]:
    if cfg.mixed_precision:
        ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
    else:
        ctx = nullcontext()

    for _ in range(cfg.warmup_steps):
        data = get_random_batch(cfg=cfg).to(device=device)
        with ctx:
            _ = model(data)
        torch.cuda.synchronize()

    times = []
    for _ in range(cfg.benchmark_steps):
        data = get_random_batch(cfg=cfg).to(device=device)
        start_time = timeit.default_timer()
        with ctx:
            with nvtx.range("Forward-Pass"):
                _ = model(data)
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        times.append(end_time - start_time)

    return np.average(times).item(), np.std(times).item()

@nvtx.range("Forward-Backward Model")
def forward_backward(cfg: Namespace, model: BasicsTransformerLM, device: torch.device) -> tuple[float, float]:
    criterion = cross_entropy
    for _ in range(cfg.warmup_steps):
        x = get_random_batch(cfg=cfg).to(device=device)
        y = get_random_batch(cfg=cfg).to(device=device)
        logits = model(x)
        loss: torch.Tensor = criterion(logits, y)
        loss.backward()
        torch.cuda.synchronize()

    times = []
    for _ in range(cfg.benchmark_steps):
        model.zero_grad()
        x = get_random_batch(cfg=cfg).to(device=device)
        y = get_random_batch(cfg=cfg).to(device=device)
        start_time = timeit.default_timer()
        with nvtx.range("Forward-Pass"):
            logits = model(x)
            loss: torch.Tensor = criterion(logits, y)
        with nvtx.range("Backward-Pass"):
            loss.backward()
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        times.append(end_time - start_time)

    return np.average(times).item(), np.std(times).item()

@nvtx.range("Forward-Backward-Optimizer Model")
def forward_backward_optimizer(cfg: Namespace, model: BasicsTransformerLM, device: torch.device) -> tuple[float, float]:
    criterion = cross_entropy
    optimizer = AdamW(model.parameters())
    for _ in range(cfg.warmup_steps):
        optimizer.zero_grad()
        x = get_random_batch(cfg=cfg).to(device=device)
        y = get_random_batch(cfg=cfg).to(device=device)
        logits = model(x)
        loss: torch.Tensor = criterion(logits, y)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

    times = []
    for _ in range(cfg.benchmark_steps):
        x = get_random_batch(cfg=cfg).to(device=device)
        y = get_random_batch(cfg=cfg).to(device=device)
        start_time = timeit.default_timer()
        optimizer.zero_grad()
        with nvtx.range("Forward-Pass"):
            logits = model(x)
            loss: torch.Tensor = criterion(logits, y)
        with nvtx.range("Backward-Pass"):
            loss.backward()
        with nvtx.range("Optimizer-Pass"):
            optimizer.step()
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        times.append(end_time - start_time)

    return np.average(times).item(), np.std(times).item()




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
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("mixed_precision", action="store_true")

    return parser.parse_args()

def benchmark(cfg: Namespace) -> None:
    model_configs = configs[cfg.mode]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BasicsTransformerLM(
        vocab_size=model_configs["vocab_size"],
        context_length=model_configs["context_length"],
        d_model=model_configs["d_model"],
        num_layers=model_configs["num_layers"],
        num_heads=model_configs["num_heads"],
        d_ff=model_configs["d_ff"],
        rope_theta=model_configs["rope_theta"],
    )

    model = model.to(device)
    print(model.device if hasattr(model, "device") else device)
    print(next(model.parameters()).device)
    match cfg.type:
        case "forward":
            avg_time, std = forward(cfg=cfg, model=model, device=device)
            print(f"Average forward computation time: {avg_time}")
            print(f"Std of forward computation time: {std}")
        case "forward-backward":
            avg_time, std = forward_backward(cfg=cfg, model=model, device=device)
            print(f"Average forward and backward computation time: {avg_time}")
            print(f"Std of forward and backward computation time: {std}")
        case "forward-backward-optimizer":
            avg_time, std = forward_backward_optimizer(cfg=cfg, model=model, device=device)
            print(f"Average forward and backward optimization time: {avg_time}")
            print(f"Std of forward and backward optimization time: {std}")
        case _:
            raise ValueError(f"Unrecognized mode: {cfg.type}")

if __name__ == "__main__":
    cfg = parse_args()
    benchmark(cfg=cfg)