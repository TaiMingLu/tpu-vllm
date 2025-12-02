#!/usr/bin/env python3
"""
Analyze token lengths in parquet dataset.

Randomly samples rows from parquet files, tokenizes them, and computes statistics.
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict

import pandas as pd
import numpy as np
from tqdm import tqdm
import transformers


def get_parquet_files(input_dir: str) -> List[Path]:
    """Get list of all parquet files in directory."""
    input_path = Path(input_dir)
    parquet_files = sorted(input_path.glob("*.parquet"))
    return parquet_files


def sample_rows_from_parquets(
    parquet_files: List[Path],
    num_samples: int,
    text_column: str
) -> List[str]:
    """Randomly sample rows from parquet files."""

    print(f"\nSampling {num_samples} rows from {len(parquet_files)} parquet files...")

    # Calculate samples per file
    samples_per_file = max(1, num_samples // len(parquet_files))
    extra_samples = num_samples % len(parquet_files)

    all_texts = []

    for i, parquet_file in enumerate(tqdm(parquet_files, desc="Sampling files")):
        # How many samples from this file
        n_from_file = samples_per_file + (1 if i < extra_samples else 0)

        try:
            # Read parquet
            df = pd.read_parquet(parquet_file)

            if text_column not in df.columns:
                print(f"Warning: Column '{text_column}' not found in {parquet_file.name}")
                continue

            # Sample randomly
            if len(df) < n_from_file:
                sampled = df[text_column].tolist()
            else:
                sampled = df[text_column].sample(n=n_from_file, random_state=42).tolist()

            # Filter out non-string or empty
            sampled = [t for t in sampled if isinstance(t, str) and t.strip()]
            all_texts.extend(sampled)

        except Exception as e:
            print(f"Error reading {parquet_file.name}: {e}")
            continue

    print(f"Collected {len(all_texts)} text samples")
    return all_texts


def tokenize_and_analyze(
    texts: List[str],
    tokenizer_path: str,
    hf_token: str = None
) -> Dict:
    """Tokenize texts and compute length statistics."""

    print(f"\nLoading tokenizer from {tokenizer_path}...")

    # Load tokenizer
    if Path(tokenizer_path).exists():
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_path,
            token=hf_token
        )

    print(f"Tokenizing {len(texts)} samples...")

    # Tokenize all texts
    token_lengths = []
    for text in tqdm(texts, desc="Tokenizing"):
        try:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            token_lengths.append(len(tokens))
        except Exception as e:
            # Skip problematic texts
            continue

    token_lengths = np.array(token_lengths)

    # Compute statistics
    stats = {
        "num_samples": len(token_lengths),
        "mean": float(np.mean(token_lengths)),
        "median": float(np.median(token_lengths)),
        "std": float(np.std(token_lengths)),
        "min": int(np.min(token_lengths)),
        "max": int(np.max(token_lengths)),
        "percentiles": {
            "p25": float(np.percentile(token_lengths, 25)),
            "p50": float(np.percentile(token_lengths, 50)),
            "p75": float(np.percentile(token_lengths, 75)),
            "p90": float(np.percentile(token_lengths, 90)),
            "p95": float(np.percentile(token_lengths, 95)),
            "p99": float(np.percentile(token_lengths, 99)),
        },
        "distribution": {
            "0-512": int(np.sum(token_lengths <= 512)),
            "513-1024": int(np.sum((token_lengths > 512) & (token_lengths <= 1024))),
            "1025-2048": int(np.sum((token_lengths > 1024) & (token_lengths <= 2048))),
            "2049-4096": int(np.sum((token_lengths > 2048) & (token_lengths <= 4096))),
            "4097+": int(np.sum(token_lengths > 4096)),
        }
    }

    return stats, token_lengths


def print_statistics(stats: Dict):
    """Pretty print statistics."""
    print("\n" + "=" * 70)
    print("TOKEN LENGTH STATISTICS")
    print("=" * 70)

    print(f"\nSample size: {stats['num_samples']:,} documents")

    print(f"\nBasic Statistics:")
    print(f"  Mean:       {stats['mean']:.1f} tokens")
    print(f"  Median:     {stats['median']:.1f} tokens")
    print(f"  Std Dev:    {stats['std']:.1f} tokens")
    print(f"  Min:        {stats['min']:,} tokens")
    print(f"  Max:        {stats['max']:,} tokens")

    print(f"\nPercentiles:")
    for pct, val in stats['percentiles'].items():
        print(f"  {pct.upper()}: {val:.1f} tokens")

    print(f"\nLength Distribution:")
    total = stats['num_samples']
    for range_name, count in stats['distribution'].items():
        pct = 100 * count / total if total > 0 else 0
        print(f"  {range_name:15s}: {count:6,} ({pct:5.1f}%)")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze token lengths in parquet dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing parquet files"
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        required=True,
        help="Path to tokenizer"
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Column name containing text"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of documents to sample"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional: Save results to JSON file"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token if needed"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    # Get parquet files
    parquet_files = get_parquet_files(args.input_dir)
    if not parquet_files:
        print(f"Error: No parquet files found in {args.input_dir}")
        return

    print(f"Found {len(parquet_files)} parquet files")

    # Sample texts
    texts = sample_rows_from_parquets(
        parquet_files,
        args.num_samples,
        args.text_column
    )

    if not texts:
        print("Error: No texts collected")
        return

    # Tokenize and analyze
    stats, token_lengths = tokenize_and_analyze(
        texts,
        args.tokenizer_path,
        args.hf_token
    )

    # Print results
    print_statistics(stats)

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\nResults saved to: {output_path}")

        # Also save raw lengths for plotting
        lengths_file = output_path.with_suffix('.lengths.npy')
        np.save(lengths_file, token_lengths)
        print(f"Raw lengths saved to: {lengths_file}")


if __name__ == "__main__":
    main()
