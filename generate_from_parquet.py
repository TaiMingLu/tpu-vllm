#!/usr/bin/env python3
"""
Generate text completions from parquet files using vLLM on TPU.

This script reads parquet files containing prompts, generates completions using
a vLLM model, and saves the results back to parquet format.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from vllm import LLM, SamplingParams


def load_parquet_files(input_path: str, prompt_column: str = "prompt") -> pd.DataFrame:
    """Load parquet file(s) into a DataFrame.

    Args:
        input_path: Path to parquet file or directory containing parquet files
        prompt_column: Name of the column containing prompts

    Returns:
        DataFrame with prompts
    """
    input_path = Path(input_path)

    if input_path.is_file():
        print(f"Loading parquet file: {input_path}")
        df = pd.read_parquet(input_path)
    elif input_path.is_dir():
        print(f"Loading parquet files from directory: {input_path}")
        parquet_files = list(input_path.glob("*.parquet"))
        if not parquet_files:
            raise ValueError(f"No parquet files found in {input_path}")
        print(f"Found {len(parquet_files)} parquet files")
        df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    else:
        raise ValueError(f"Input path does not exist: {input_path}")

    if prompt_column not in df.columns:
        raise ValueError(f"Prompt column '{prompt_column}' not found in parquet. "
                        f"Available columns: {list(df.columns)}")

    print(f"Loaded {len(df)} rows")
    return df


def generate_completions(
    llm: LLM,
    prompts: List[str],
    sampling_params: SamplingParams,
    batch_size: Optional[int] = None,
    show_progress: bool = True
) -> List[str]:
    """Generate completions for a list of prompts.

    Args:
        llm: vLLM LLM instance
        prompts: List of prompt strings
        sampling_params: Sampling parameters for generation
        batch_size: If specified, process in batches (vLLM handles batching internally)
        show_progress: Show progress bar

    Returns:
        List of generated completions
    """
    print(f"\nGenerating completions for {len(prompts)} prompts...")

    # vLLM processes all prompts efficiently in one call
    outputs = llm.generate(prompts, sampling_params)

    completions = []
    for output in outputs:
        # Get the first (and typically only) completion
        generated_text = output.outputs[0].text
        completions.append(generated_text)

    return completions


def save_results(
    df: pd.DataFrame,
    output_path: str,
    completion_column: str = "completion",
    compression: str = "snappy"
):
    """Save DataFrame with completions to parquet.

    Args:
        df: DataFrame with results
        output_path: Output parquet file path
        completion_column: Name for the completion column
        compression: Parquet compression codec (snappy, gzip, brotli, none)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving results to: {output_path}")
    df.to_parquet(
        output_path,
        engine='pyarrow',
        compression=compression,
        index=False
    )
    print(f"Saved {len(df)} rows to {output_path}")

    # Print file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Output file size: {file_size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Generate text completions from parquet files using vLLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/Output
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input parquet file or directory containing parquet files"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output parquet file path"
    )
    parser.add_argument(
        "--prompt-column",
        type=str,
        default="prompt",
        help="Name of column containing prompts"
    )
    parser.add_argument(
        "--completion-column",
        type=str,
        default="completion",
        help="Name for output completion column"
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model (HuggingFace format)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to tokenizer (if different from model)"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of tensor parallel replicas"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="Maximum model sequence length"
    )

    # Generation parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling threshold"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=-1,
        help="Top-k sampling (-1 to disable)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty (1.0 = no penalty)"
    )
    parser.add_argument(
        "--skip-special-tokens",
        action="store_true",
        default=True,
        help="Skip special tokens in output"
    )

    # Processing options
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Maximum number of rows to process (for testing)"
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="snappy",
        choices=["snappy", "gzip", "brotli", "none"],
        help="Parquet compression codec"
    )

    args = parser.parse_args()

    # Load input data
    print("=" * 80)
    print("Loading input data...")
    print("=" * 80)
    df = load_parquet_files(args.input, args.prompt_column)

    # Limit rows if specified
    if args.max_rows is not None:
        print(f"Limiting to first {args.max_rows} rows")
        df = df.head(args.max_rows)

    # Extract prompts
    prompts = df[args.prompt_column].tolist()
    print(f"Processing {len(prompts)} prompts")

    # Initialize model
    print("\n" + "=" * 80)
    print("Initializing vLLM model...")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Tokenizer: {args.tokenizer or args.model}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print(f"Max model length: {args.max_model_len}")

    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer or args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )

    # Setup sampling parameters
    print("\n" + "=" * 80)
    print("Generation parameters:")
    print("=" * 80)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
        skip_special_tokens=args.skip_special_tokens,
        ignore_eos=True,
    )
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-p: {args.top_p}")
    print(f"  Top-k: {args.top_k}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Repetition penalty: {args.repetition_penalty}")
    print(f"  Skip special tokens: {args.skip_special_tokens}")

    # Generate completions
    print("\n" + "=" * 80)
    print("Generating completions...")
    print("=" * 80)
    completions = generate_completions(
        llm=llm,
        prompts=prompts,
        sampling_params=sampling_params,
        show_progress=True
    )

    # Add completions to dataframe
    df[args.completion_column] = completions

    # Print sample results
    print("\n" + "=" * 80)
    print("Sample results (first 3):")
    print("=" * 80)
    for i in range(min(3, len(df))):
        print(f"\n--- Sample {i+1} ---")
        print(f"Prompt: {df.iloc[i][args.prompt_column][:100]}...")
        print(f"Completion: {df.iloc[i][args.completion_column][:200]}...")

    # Save results
    print("\n" + "=" * 80)
    print("Saving results...")
    print("=" * 80)
    save_results(
        df=df,
        output_path=args.output,
        completion_column=args.completion_column,
        compression=args.compression
    )

    # Print summary statistics
    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    prompt_lengths = df[args.prompt_column].str.len()
    completion_lengths = df[args.completion_column].str.len()

    print(f"Total prompts processed: {len(df)}")
    print(f"\nPrompt length stats (characters):")
    print(f"  Mean: {prompt_lengths.mean():.1f}")
    print(f"  Median: {prompt_lengths.median():.1f}")
    print(f"  Min: {prompt_lengths.min()}")
    print(f"  Max: {prompt_lengths.max()}")

    print(f"\nCompletion length stats (characters):")
    print(f"  Mean: {completion_lengths.mean():.1f}")
    print(f"  Median: {completion_lengths.median():.1f}")
    print(f"  Min: {completion_lengths.min()}")
    print(f"  Max: {completion_lengths.max()}")

    print("\n" + "=" * 80)
    print("✓ Generation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
