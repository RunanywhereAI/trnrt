"""Prices per chip-hour for tokens per dollar, shared by the scorers.

- "sa": SemiAnalysis's rate for trn2, $0.58 per chip-hour, the rate InferenceX's comparisons use for trn2.
- "aws_capacity_block": AWS's EC2 Capacity Block list price for trn2.48xlarge, $35.76 an hour for 16 chips = $2.235
  per chip-hour (trn2.48xlarge has no on-demand price in US regions).

Tokens per dollar (millions) = total tokens / s / chip x 3,600 / price per chip-hour / 10^6.
"""

from __future__ import annotations

import argparse
import math

DEFAULT_PRICES = {"sa": 0.58, "aws_capacity_block": 2.235}


def parse_price(s: str) -> tuple[str, float]:
    """NAME=USD_PER_CHIP_HOUR, for argparse."""
    name, _, v = s.partition("=")
    try:
        p = float(v)
    except ValueError:
        p = float("nan")
    if not name or not math.isfinite(p) or p <= 0:
        raise argparse.ArgumentTypeError(f"--price {s!r}: want NAME=USD_PER_CHIP_HOUR, a positive number")
    return name, p


def tokens_per_usd_millions(tok_s_per_chip: float, prices: dict[str, float], digits: int = 3) -> dict[str, float]:
    return {k: round(tok_s_per_chip * 3600 / p / 1e6, digits) for k, p in prices.items()}
