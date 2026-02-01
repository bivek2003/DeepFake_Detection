#!/usr/bin/env python3
"""
Evaluation Script for Deepfake Detection Model

Computes comprehensive metrics and generates evaluation reports:
- Accuracy, Precision, Recall, F1 Score
- AUC-ROC and AUC-PR curves
- Confusion matrix
- Per-class analysis
- Temperature scaling calibration
- Cross-dataset generalization

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --calibrate
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import argparse
import json
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from app.ml.architecture import DeepfakeDetector, create_model
from app.ml.data.datasets import create_dataloaders, DeepfakeDataset
from app.ml.data.transforms import get_val_transforms


class ModelEvaluator:
    """
    Comprehensive model evaluation with metrics and visualization.
    """
    
    def __init__(
        self,
        model: DeepfakeDetector,
        device: str = "cuda",
        output_dir: str = "./evaluation",
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model
            device: Device for inference
            output_dir: Directory for output files
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calibration temperature
        self.temperature = 1.0
    
    @torch.no_grad()
    def predict(
        self,
        dataloader: DataLoader,
        return_features: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Get predictions for entire dataset.
        
        Args:
            dataloader: Data loader
            return_features: Whether to return feature vectors
        
        Returns:
            Tuple of (probabilities, labels, features)
        """
        all_probs = []
        all_labels = []
        all_features = [] if return_features else None
        
        for images, labels in tqdm(dataloader, desc="Predicting"):
            images = images.to(self.device)
            
            # Get logits
            logits = self.model(images)
            
            # Apply temperature scaling
            logits = logits / self.temperature
            
            # Get probabilities
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            
            all_probs.extend(probs.tolist() if probs.ndim > 0 else [probs.item()])
            all_labels.extend(labels.numpy().tolist())
            
            if return_features:
                features = self.model.get_features(images).cpu().numpy()
                all_features.append(features)
        
        probs = np.array(all_probs)
        labels = np.array(all_labels)
        features = np.concatenate(all_features, axis=0) if return_features else None
        
        return probs, labels, features
    
    def compute_metrics(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Compute comprehensive metrics.
        
        Args:
            probs: Predicted probabilities
            labels: True labels
            threshold: Classification threshold
        
        Returns:
            Dictionary of metrics
        """
        preds = (probs >= threshold).astype(int)
        
        metrics = {
            "threshold": threshold,
            "accuracy": float(accuracy_score(labels, preds)),
            "precision": float(precision_score(labels, preds, zero_division=0)),
            "recall": float(recall_score(labels, preds, zero_division=0)),
            "f1_score": float(f1_score(labels, preds, zero_division=0)),
            "specificity": float(recall_score(1 - labels, 1 - preds, zero_division=0)),
        }
        
        # AUC metrics
        if len(np.unique(labels)) > 1:
            metrics["auc_roc"] = float(roc_auc_score(labels, probs))
            metrics["auc_pr"] = float(average_precision_score(labels, probs))
        else:
            metrics["auc_roc"] = 0.5
            metrics["auc_pr"] = 0.5
        
        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        metrics["confusion_matrix"] = cm.tolist()
        
        # Class distribution
        metrics["n_samples"] = len(labels)
        metrics["n_real"] = int((labels == 0).sum())
        metrics["n_fake"] = int((labels == 1).sum())
        
        return metrics
    
    def find_optimal_threshold(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        metric: str = "f1",
    ) -> Tuple[float, float]:
        """
        Find optimal classification threshold.
        
        Args:
            probs: Predicted probabilities
            labels: True labels
            metric: Metric to optimize ('f1', 'accuracy', 'youden')
        
        Returns:
            Tuple of (optimal_threshold, metric_value)
        """
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_threshold = 0.5
        best_value = 0.0
        
        for thresh in thresholds:
            preds = (probs >= thresh).astype(int)
            
            if metric == "f1":
                value = f1_score(labels, preds, zero_division=0)
            elif metric == "accuracy":
                value = accuracy_score(labels, preds)
            elif metric == "youden":
                # Youden's J statistic = sensitivity + specificity - 1
                sens = recall_score(labels, preds, zero_division=0)
                spec = recall_score(1 - labels, 1 - preds, zero_division=0)
                value = sens + spec - 1
            else:
                value = f1_score(labels, preds, zero_division=0)
            
            if value > best_value:
                best_value = value
                best_threshold = thresh
        
        return best_threshold, best_value
    
    def calibrate_temperature(
        self,
        val_loader: DataLoader,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> float:
        """
        Find optimal temperature for calibration using validation set.
        
        Args:
            val_loader: Validation data loader
            lr: Learning rate for optimization
            max_iter: Maximum iterations
        
        Returns:
            Optimal temperature
        """
        print("Calibrating temperature...")
        
        # Collect logits and labels
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                logits = self.model(images)
                all_logits.append(logits.cpu())
                all_labels.append(labels)
        
        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0).float()
        
        # Optimize temperature
        temperature = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)
        
        def eval_temp():
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(
                logits.squeeze() / temperature,
                labels,
            )
            loss.backward()
            return loss
        
        optimizer.step(eval_temp)
        
        optimal_temp = temperature.item()
        self.temperature = optimal_temp
        
        print(f"Optimal temperature: {optimal_temp:.4f}")
        
        return optimal_temp
    
    def compute_ece(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 15,
    ) -> float:
        """
        Compute Expected Calibration Error.
        
        Args:
            probs: Predicted probabilities
            labels: True labels
            n_bins: Number of bins
        
        Returns:
            ECE value
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probs > bin_lower) & (probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = probs[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return float(ece)
    
    def plot_roc_curve(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        save_path: Optional[str] = None,
    ):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14)
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved ROC curve: {save_path}")
        
        plt.close()
    
    def plot_precision_recall_curve(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        save_path: Optional[str] = None,
    ):
        """Plot Precision-Recall curve."""
        precision, recall, _ = precision_recall_curve(labels, probs)
        auc = average_precision_score(labels, probs)
        
        plt.figure(figsize=(8, 8))
        plt.plot(recall, precision, 'b-', linewidth=2, label=f'PR (AP = {auc:.4f})')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14)
        plt.legend(loc='lower left', fontsize=11)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved PR curve: {save_path}")
        
        plt.close()
    
    def plot_confusion_matrix(
        self,
        labels: np.ndarray,
        preds: np.ndarray,
        save_path: Optional[str] = None,
    ):
        """Plot confusion matrix."""
        cm = confusion_matrix(labels, preds)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix', fontsize=14)
        plt.colorbar()
        
        classes = ['Real', 'Fake']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, fontsize=11)
        plt.yticks(tick_marks, classes, fontsize=11)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=14)
        
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved confusion matrix: {save_path}")
        
        plt.close()
    
    def plot_calibration_curve(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10,
        save_path: Optional[str] = None,
    ):
        """Plot reliability/calibration diagram."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probs > bin_lower) & (probs <= bin_upper)
            if in_bin.sum() > 0:
                bin_accuracies.append(labels[in_bin].mean())
                bin_confidences.append(probs[in_bin].mean())
                bin_counts.append(in_bin.sum())
            else:
                bin_accuracies.append(0)
                bin_confidences.append((bin_lower + bin_upper) / 2)
                bin_counts.append(0)
        
        ece = self.compute_ece(probs, labels, n_bins)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Calibration curve
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfectly calibrated')
        ax1.bar(bin_confidences, bin_accuracies, width=0.1, alpha=0.5, label='Model')
        ax1.scatter(bin_confidences, bin_accuracies, s=50, zorder=5)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax1.set_ylabel('Fraction of Positives', fontsize=12)
        ax1.set_title(f'Calibration Curve (ECE = {ece:.4f})', fontsize=14)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Histogram
        ax2.hist(probs, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Predicted Probability', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Distribution of Predictions', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved calibration curve: {save_path}")
        
        plt.close()
    
    def evaluate(
        self,
        test_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        calibrate: bool = True,
        generate_plots: bool = True,
    ) -> Dict[str, Any]:
        """
        Run full evaluation.
        
        Args:
            test_loader: Test data loader
            val_loader: Validation loader for calibration
            calibrate: Whether to perform temperature calibration
            generate_plots: Whether to generate visualization plots
        
        Returns:
            Dictionary of all metrics and results
        """
        print("\n" + "="*80)
        print("Model Evaluation")
        print("="*80)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "temperature": self.temperature,
        }
        
        # Calibrate if requested
        if calibrate and val_loader is not None:
            results["temperature"] = self.calibrate_temperature(val_loader)
        
        # Get predictions
        probs, labels, _ = self.predict(test_loader)
        
        # Compute metrics at default threshold
        print("\nComputing metrics...")
        metrics = self.compute_metrics(probs, labels, threshold=0.5)
        results["metrics_default"] = metrics
        
        # Find optimal threshold
        optimal_thresh, optimal_f1 = self.find_optimal_threshold(probs, labels, "f1")
        results["optimal_threshold"] = optimal_thresh
        results["optimal_f1"] = optimal_f1
        
        # Metrics at optimal threshold
        metrics_optimal = self.compute_metrics(probs, labels, threshold=optimal_thresh)
        results["metrics_optimal"] = metrics_optimal
        
        # Calibration error
        ece_before = self.compute_ece(probs, labels)
        results["ece"] = ece_before
        
        # Print summary
        print("\n" + "-"*40)
        print("Results Summary")
        print("-"*40)
        print(f"Samples: {metrics['n_samples']} (Real: {metrics['n_real']}, Fake: {metrics['n_fake']})")
        print(f"\nDefault Threshold (0.5):")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1_score']:.4f}")
        print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
        print(f"  AUC-PR:    {metrics['auc_pr']:.4f}")
        print(f"\nOptimal Threshold ({optimal_thresh:.2f}):")
        print(f"  Accuracy:  {metrics_optimal['accuracy']:.4f}")
        print(f"  Precision: {metrics_optimal['precision']:.4f}")
        print(f"  Recall:    {metrics_optimal['recall']:.4f}")
        print(f"  F1 Score:  {metrics_optimal['f1_score']:.4f}")
        print(f"\nCalibration:")
        print(f"  Temperature: {results['temperature']:.4f}")
        print(f"  ECE:         {ece_before:.4f}")
        
        # Generate plots
        if generate_plots:
            print("\nGenerating plots...")
            
            self.plot_roc_curve(
                probs, labels,
                save_path=str(self.output_dir / "roc_curve.png")
            )
            
            self.plot_precision_recall_curve(
                probs, labels,
                save_path=str(self.output_dir / "pr_curve.png")
            )
            
            preds = (probs >= 0.5).astype(int)
            self.plot_confusion_matrix(
                labels, preds,
                save_path=str(self.output_dir / "confusion_matrix.png")
            )
            
            self.plot_calibration_curve(
                probs, labels,
                save_path=str(self.output_dir / "calibration_curve.png")
            )
        
        # Save results
        results_path = self.output_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate deepfake detection model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--datasets-dir", type=str, default="./datasets", help="Datasets directory")
    parser.add_argument("--output-dir", type=str, default="./evaluation", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--calibrate", action="store_true", help="Perform temperature calibration")
    parser.add_argument("--backbone", type=str, default="efficientnet_b4", help="Model backbone")
    
    args = parser.parse_args()
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Fix paths
    if not Path(args.datasets_dir).exists() and Path("/app/datasets").exists():
        args.datasets_dir = "/app/datasets"
    if not Path(args.output_dir).parent.exists():
        args.output_dir = "./evaluation"
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = create_model(
        backbone=args.backbone,
        pretrained=False,
        checkpoint_path=args.checkpoint,
        device=device,
    )
    
    # Create data loaders
    print(f"Loading data from: {args.datasets_dir}")
    _, val_loader, test_loader = create_dataloaders(
        root_dir=args.datasets_dir,
        batch_size=args.batch_size,
        num_workers=4,
        val_transform=get_val_transforms(),
    )
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model=model,
        device=device,
        output_dir=args.output_dir,
    )
    
    # Run evaluation
    results = evaluator.evaluate(
        test_loader=test_loader,
        val_loader=val_loader if args.calibrate else None,
        calibrate=args.calibrate,
        generate_plots=True,
    )
    
    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)


if __name__ == "__main__":
    main()
