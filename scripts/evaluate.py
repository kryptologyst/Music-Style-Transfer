#!/usr/bin/env python3
"""Evaluation script for music style transfer."""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch
import json
from datetime import datetime

from src.models import StyleTransferModel
from src.data import create_dataloaders
from src.metrics import evaluate_model, create_evaluation_report
from src.utils.device import get_device, set_seed


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate music style transfer model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = StyleTransferModel.load_model(args.checkpoint, device)
    
    print(f"Model info: {model.get_model_info()}")
    
    # Create data loaders
    print("Loading data...")
    _, _, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4,
        max_duration=30.0,
        augment=False
    )
    
    print(f"Test batches: {len(test_loader)}")
    
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(model, test_loader, device)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    # Create evaluation report
    report = create_evaluation_report(metrics, model.get_model_info())
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save metrics as JSON
    metrics_file = os.path.join(args.output_dir, f"metrics_{timestamp}.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save report as text
    report_file = os.path.join(args.output_dir, f"evaluation_report_{timestamp}.txt")
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\nResults saved to:")
    print(f"- Metrics: {metrics_file}")
    print(f"- Report: {report_file}")
    
    # Print report
    print("\n" + "="*50)
    print("EVALUATION REPORT")
    print("="*50)
    print(report)


if __name__ == "__main__":
    main()
