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
import pyarrow.dataset as ds


def sample_rows_from_parquet_dataset(
    input_dir: str,
    num_samples: int,
    text_column: str,
    random_seed: int = 42
) -> List[str]:
    """Efficiently sample rows from parquet dataset without opening files multiple times."""

    print(f"\nLoading parquet dataset from {input_dir}...")

    # Create dataset from directory (treats all parquet files as one dataset)
    dataset = ds.dataset(input_dir, format='parquet')

    # Count total rows efficiently (without loading data)
    total_rows = dataset.count_rows()
    print(f"Found {total_rows:,} total rows across all parquet files")

    # Generate random row indices to sample
    print(f"Sampling {num_samples} random rows...")
    np.random.seed(random_seed)

    if num_samples >= total_rows:
        print(f"Warning: Requested {num_samples} samples but only {total_rows} available")
        sample_indices = np.arange(total_rows)
    else:
        sample_indices = np.random.choice(total_rows, size=num_samples, replace=False)

    sample_indices = sorted(sample_indices)  # Sort for efficient sequential reading

    # Read only the text column for sampled rows
    print(f"Reading sampled rows from dataset...")
    all_texts = []

    # Read in batches for efficiency
    batch_size = 10000
    scanner = dataset.scanner(columns=[text_column])

    current_idx = 0
    sample_set = set(sample_indices)

    for batch in tqdm(scanner.to_batches(batch_size=batch_size), desc="Processing batches"):
        # Convert batch to pandas for easier handling
        df = batch.to_pandas()

        # Find which rows in this batch we want to sample
        batch_end = current_idx + len(df)
        rows_to_sample = [idx - current_idx for idx in sample_set
                          if current_idx <= idx < batch_end]

        if rows_to_sample:
            # Extract the sampled rows
            sampled = df[text_column].iloc[rows_to_sample].tolist()

            # Filter out non-string or empty
            sampled = [t for t in sampled if isinstance(t, str) and len(t.strip()) > 0]
            all_texts.extend(sampled)

            # Remove sampled indices from set
            for idx in rows_to_sample:
                sample_set.discard(current_idx + idx)

            # Early exit if we've collected all samples
            if not sample_set:
                break

        current_idx = batch_end

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

    # Sample texts directly from dataset (efficient - no multiple file opens)
    texts = sample_rows_from_parquet_dataset(
        args.input_dir,
        args.num_samples,
        args.text_column,
        args.random_seed
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
