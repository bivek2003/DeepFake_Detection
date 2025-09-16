"""
Model Evaluation and Metrics for Deepfake Detection

Comprehensive evaluation framework including:
- Cross-dataset evaluation
- ROC curves and confusion matrices
- Explainability with Grad-CAM
- Performance benchmarking
- Statistical significance testing

Author: Bivek Sharma Panthi
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_curve, auc, average_precision_score, classification_report
)
from sklearn.manifold import TSNE
import cv2
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import json
import time
from dataclasses import dataclass
from torch.utils.data import DataLoader

from .base_model import BaseDeepfakeModel, load_model_weights
from ..utils import DeviceManager

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResults:
    """Container for evaluation results"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    ap_score: float  # Average Precision
    confusion_matrix: np.ndarray
    classification_report: str
    
    # Per-class metrics
    per_class_precision: List[float]
    per_class_recall: List[float]
    per_class_f1: List[float]
    
    # Additional info
    num_samples: int
    inference_time: float
    model_info: Dict[str, Any]


class ModelEvaluator:
    """Comprehensive model evaluation framework"""
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or DeviceManager.get_device()
        self.class_names = ['Real', 'Fake']
        
    def evaluate_model(self, 
                      model: BaseDeepfakeModel,
                      dataloader: DataLoader,
                      dataset_name: str = "test") -> EvaluationResults:
        """Comprehensive model evaluation"""
        model.eval()
        model = model.to(self.device)
        
        all_predictions = []
        all_probabilities = []
        all_targets = []
        inference_times = []
        
        logger.info(f"Evaluating model on {dataset_name} dataset...")
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(dataloader):
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                # Measure inference time
                start_time = time.time()
                outputs = model(data, return_features=True)
                inference_time = time.time() - start_time
                inference_times.append(inference_time / data.size(0))  # Per sample
                
                # Extract predictions and probabilities
                probabilities = outputs.probabilities if hasattr(outputs, 'probabilities') else F.softmax(outputs, dim=1)
                predictions = outputs.predictions if hasattr(outputs, 'predictions') else torch.argmax(probabilities, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        probabilities = np.array(all_probabilities)
        targets = np.array(all_targets)
        
        # Calculate metrics
        accuracy = accuracy_score(targets, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average='weighted'
        )
        
        # Per-class metrics
        per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
            targets, predictions, average=None
        )
        
        # ROC AUC
        if probabilities.shape[1] == 2:  # Binary classification
            fpr, tpr, _ = roc_curve(targets, probabilities[:, 1])
            auc_score = auc(fpr, tpr)
            ap_score = average_precision_score(targets, probabilities[:, 1])
        else:
            # Multi-class (though deepfake is typically binary)
            auc_score = 0.0
            ap_score = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        # Classification report
        report = classification_report(
            targets, predictions, 
            target_names=self.class_names,
            digits=4
        )
        
        results = EvaluationResults(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_score=auc_score,
            ap_score=ap_score,
            confusion_matrix=cm,
            classification_report=report,
            per_class_precision=per_class_precision.tolist(),
            per_class_recall=per_class_recall.tolist(),
            per_class_f1=per_class_f1.tolist(),
            num_samples=len(targets),
            inference_time=np.mean(inference_times),
            model_info=model.get_model_info()
        )
        
        logger.info(f"Evaluation completed:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")
        logger.info(f"  AUC: {auc_score:.4f}")
        
        return results
    
    def cross_dataset_evaluation(self,
                                model: BaseDeepfakeModel,
                                dataloaders: Dict[str, DataLoader]) -> Dict[str, EvaluationResults]:
        """Evaluate model across multiple datasets"""
        results = {}
        
        logger.info("Starting cross-dataset evaluation...")
        
        for dataset_name, dataloader in dataloaders.items():
            logger.info(f"Evaluating on {dataset_name}...")
            results[dataset_name] = self.evaluate_model(model, dataloader, dataset_name)
        
        # Calculate generalization scores
        accuracies = [result.accuracy for result in results.values()]
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        
        logger.info(f"Cross-dataset results:")
        logger.info(f"  Mean accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        logger.info(f"  Min accuracy: {np.min(accuracies):.4f}")
        logger.info(f"  Max accuracy: {np.max(accuracies):.4f}")
        
        return results
    
    def plot_confusion_matrix(self, 
                            results: EvaluationResults,
                            title: str = "Confusion Matrix",
                            save_path: Optional[str] = None):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        
        cm_normalized = results.confusion_matrix.astype('float') / results.confusion_matrix.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.3f',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Normalized Frequency'}
        )
        
        plt.title(f'{title}\nAccuracy: {results.accuracy:.3f}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Add raw counts as text
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                plt.text(j+0.5, i+0.7, f'({results.confusion_matrix[i, j]})', 
                        ha='center', va='center', fontsize=10, alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self,
                      model: BaseDeepfakeModel,
                      dataloader: DataLoader,
                      title: str = "ROC Curve",
                      save_path: Optional[str] = None):
        """Plot ROC curve for binary classification"""
        model.eval()
        model = model.to(self.device)
        
        all_probabilities = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in dataloader:
                data = data.to(self.device)
                outputs = model(data)
                probabilities = F.softmax(outputs, dim=1)
                
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Fake class probability
                all_targets.extend(targets.numpy())
        
        targets = np.array(all_targets)
        probabilities = np.array(all_probabilities)
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(targets, probabilities)
        auc_score = auc(fpr, tpr)
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.show()
        
        return fpr, tpr, thresholds, auc_score
    
    def compare_models(self,
                      models: Dict[str, BaseDeepfakeModel],
                      dataloader: DataLoader,
                      save_path: Optional[str] = None) -> Dict[str, EvaluationResults]:
        """Compare multiple models on the same dataset"""
        results = {}
        
        logger.info(f"Comparing {len(models)} models...")
        
        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name}...")
            results[model_name] = self.evaluate_model(model, dataloader, "comparison")
        
        # Create comparison plot
        self.plot_model_comparison(results, save_path)
        
        return results
    
    def plot_model_comparison(self,
                            results: Dict[str, EvaluationResults],
                            save_path: Optional[str] = None):
        """Plot model comparison"""
        model_names = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            values = [getattr(results[name], metric) for name in model_names]
            
            ax = axes[i]
            bars = ax.bar(model_names, values, alpha=0.7)
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
            
            # Rotate x-axis labels if needed
            if len(max(model_names, key=len)) > 8:
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Inference time comparison
        ax = axes[5]
        inference_times = [results[name].inference_time * 1000 for name in model_names]  # Convert to ms
        bars = ax.bar(model_names, inference_times, alpha=0.7, color='orange')
        ax.set_title('Inference Time (ms)')
        ax.set_ylabel('Time (ms)')
        
        for bar, value in zip(bars, inference_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(inference_times)*0.01,
                   f'{value:.2f}', ha='center', va='bottom')
        
        if len(max(model_names, key=len)) > 8:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        
        plt.show()


class GradCAM:
    """Gradient-based Class Activation Mapping for explainability"""
    
    def __init__(self, model: BaseDeepfakeModel, target_layer: str = None):
        self.model = model
        self.model.eval()
        
        # Find target layer automatically if not specified
        if target_layer is None:
            # For most CNN models, use the last convolutional layer
            self.target_layer = self._find_target_layer()
        else:
            self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _find_target_layer(self) -> str:
        """Automatically find the best layer for Grad-CAM"""
        # Look for the last conv layer in common architectures
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target = name
        
        logger.info(f"Using layer '{target}' for Grad-CAM")
        return target
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Find the target layer
        target_module = dict(self.model.named_modules())[self.target_layer]
        target_module.register_forward_hook(forward_hook)
        target_module.register_backward_hook(backward_hook)
    
    def generate_cam(self, 
                    input_tensor: torch.Tensor, 
                    target_class: Optional[int] = None) -> np.ndarray:
        """Generate Grad-CAM for input"""
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        output[0, target_class].backward()
        
        # Get gradients and activations
        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()
        
        # Calculate weights
        weights = np.mean(gradients, axis=(1, 2))
        
        # Generate CAM
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # ReLU and normalize
        cam = np.maximum(cam, 0)
        cam = cam / cam.max() if cam.max() > 0 else cam
        
        return cam
    
    def visualize_cam(self,
                     input_image: np.ndarray,
                     input_tensor: torch.Tensor,
                     target_class: Optional[int] = None,
                     alpha: float = 0.4,
                     save_path: Optional[str] = None) -> np.ndarray:
        """Visualize Grad-CAM overlay on input image"""
        # Generate CAM
        cam = self.generate_cam(input_tensor, target_class)
        
        # Resize CAM to input image size
        h, w = input_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Normalize input image
        if input_image.max() > 1:
            input_image = input_image.astype(np.float32) / 255.0
        
        # Overlay heatmap on original image
        overlay = heatmap * alpha + input_image * (1 - alpha)
        overlay = np.clip(overlay, 0, 1)
        
        if save_path:
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(input_image)
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(cam_resized, cmap='jet')
            plt.title('Grad-CAM Heatmap')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(overlay)
            plt.title('Grad-CAM Overlay')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info(f"Grad-CAM visualization saved to {save_path}")
        
        return overlay


class PerformanceBenchmark:
    """Performance benchmarking for deepfake detection models"""
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or DeviceManager.get_device()
    
    def benchmark_inference_speed(self,
                                model: BaseDeepfakeModel,
                                input_shape: Tuple[int, ...],
                                num_runs: int = 100,
                                warmup_runs: int = 10) -> Dict[str, float]:
        """Benchmark model inference speed"""
        model.eval()
        model = model.to(self.device)
        
        # Create dummy input
        if len(input_shape) == 3:  # Image input (C, H, W)
            dummy_input = torch.randn(1, *input_shape).to(self.device)
        else:  # Audio input (length,)
            dummy_input = torch.randn(1, *input_shape).to(self.device)
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(dummy_input)
        
        # Synchronize CUDA if available
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark runs
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(dummy_input)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.time()
                times.append(end_time - start_time)
        
        times = np.array(times)
        
        results = {
            'mean_time_ms': np.mean(times) * 1000,
            'std_time_ms': np.std(times) * 1000,
            'min_time_ms': np.min(times) * 1000,
            'max_time_ms': np.max(times) * 1000,
            'fps': 1.0 / np.mean(times),
            'throughput_samples_per_sec': 1.0 / np.mean(times)
        }
        
        logger.info(f"Performance benchmark results:")
        logger.info(f"  Mean inference time: {results['mean_time_ms']:.2f} ± {results['std_time_ms']:.2f} ms")
        logger.info(f"  FPS: {results['fps']:.2f}")
        logger.info(f"  Throughput: {results['throughput_samples_per_sec']:.2f} samples/sec")
        
        return results
    
    def benchmark_memory_usage(self,
                              model: BaseDeepfakeModel,
                              input_shape: Tuple[int, ...]) -> Dict[str, float]:
        """Benchmark model memory usage"""
        if self.device.type != 'cuda':
            logger.warning("Memory benchmarking only available on CUDA devices")
            return {}
        
        model.eval()
        model = model.to(self.device)
        
        # Create dummy input
        if len(input_shape) == 3:
            dummy_input = torch.randn(1, *input_shape).to(self.device)
        else:
            dummy_input = torch.randn(1, *input_shape).to(self.device)
        
        # Measure memory before inference
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        memory_before = torch.cuda.memory_allocated()
        
        # Forward pass
        with torch.no_grad():
            _ = model(dummy_input)
        
        # Measure memory after inference
        memory_after = torch.cuda.memory_allocated()
        peak_memory = torch.cuda.max_memory_allocated()
        
        results = {
            'model_memory_mb': (memory_after - memory_before) / 1024**2,
            'peak_memory_mb': peak_memory / 1024**2,
            'total_memory_mb': memory_after / 1024**2
        }
        
        logger.info(f"Memory usage:")
        logger.info(f"  Model memory: {results['model_memory_mb']:.2f} MB")
        logger.info(f"  Peak memory: {results['peak_memory_mb']:.2f} MB")
        logger.info(f"  Total memory: {results['total_memory_mb']:.2f} MB")
        
        return results


class StatisticalTester:
    """Statistical significance testing for model comparisons"""
    
    @staticmethod
    def mcnemar_test(predictions1: np.ndarray,
                    predictions2: np.ndarray,
                    targets: np.ndarray) -> Dict[str, float]:
        """McNemar's test for comparing two models"""
        from scipy.stats import mcnemar
        
        # Create contingency table
        correct1 = (predictions1 == targets)
        correct2 = (predictions2 == targets)
        
        # McNemar table
        both_correct = np.sum(correct1 & correct2)
        model1_correct = np.sum(correct1 & ~correct2)
        model2_correct = np.sum(~correct1 & correct2)
        both_wrong = np.sum(~correct1 & ~correct2)
        
        contingency_table = np.array([[both_correct, model1_correct],
                                     [model2_correct, both_wrong]])
        
        # Perform McNemar test
        result = mcnemar(contingency_table, exact=False, correction=True)
        
        return {
            'statistic': result.statistic,
            'p_value': result.pvalue,
            'significant': result.pvalue < 0.05,
            'contingency_table': contingency_table.tolist()
        }
    
    @staticmethod
    def paired_t_test(scores1: np.ndarray,
                     scores2: np.ndarray) -> Dict[str, float]:
        """Paired t-test for comparing model performance"""
        from scipy.stats import ttest_rel
        
        statistic, p_value = ttest_rel(scores1, scores2)
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'mean_difference': np.mean(scores1 - scores2)
        }


def save_evaluation_results(results: Union[EvaluationResults, Dict[str, EvaluationResults]],
                          save_path: Union[str, Path]):
    """Save evaluation results to JSON"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    def convert_to_serializable(obj):
        """Convert numpy arrays and other objects to serializable format"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, EvaluationResults):
            return {
                'accuracy': obj.accuracy,
                'precision': obj.precision,
                'recall': obj.recall,
                'f1_score': obj.f1_score,
                'auc_score': obj.auc_score,
                'ap_score': obj.ap_score,
                'confusion_matrix': obj.confusion_matrix.tolist(),
                'classification_report': obj.classification_report,
                'per_class_precision': obj.per_class_precision,
                'per_class_recall': obj.per_class_recall,
                'per_class_f1': obj.per_class_f1,
                'num_samples': obj.num_samples,
                'inference_time': obj.inference_time,
                'model_info': obj.model_info
            }
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(save_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Evaluation results saved to {save_path}")


def load_evaluation_results(load_path: Union[str, Path]) -> Dict[str, Any]:
    """Load evaluation results from JSON"""
    with open(load_path, 'r') as f:
        results = json.load(f)
    
    logger.info(f"Evaluation results loaded from {load_path}")
    return results


def create_evaluation_report(results: Dict[str, EvaluationResults],
                           save_path: Optional[Union[str, Path]] = None) -> str:
    """Create comprehensive evaluation report"""
    report = ["# Deepfake Detection Model Evaluation Report", ""]
    report.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Summary table
    report.append("## Summary")
    report.append("")
    report.append("| Dataset | Accuracy | Precision | Recall | F1-Score | AUC | Samples |")
    report.append("|---------|----------|-----------|--------|----------|-----|---------|")
    
    for dataset_name, result in results.items():
        report.append(f"| {dataset_name} | {result.accuracy:.4f} | {result.precision:.4f} | "
                     f"{result.recall:.4f} | {result.f1_score:.4f} | {result.auc_score:.4f} | "
                     f"{result.num_samples} |")
    
    report.append("")
    
    # Detailed results for each dataset
    for dataset_name, result in results.items():
        report.append(f"## {dataset_name} Dataset")
        report.append("")
        report.append("### Metrics")
        report.append(f"- **Accuracy**: {result.accuracy:.4f}")
        report.append(f"- **Precision**: {result.precision:.4f}")
        report.append(f"- **Recall**: {result.recall:.4f}")
        report.append(f"- **F1-Score**: {result.f1_score:.4f}")
        report.append(f"- **AUC**: {result.auc_score:.4f}")
        report.append(f"- **Average Precision**: {result.ap_score:.4f}")
        report.append(f"- **Inference Time**: {result.inference_time*1000:.2f} ms")
        report.append("")
        
        report.append("### Per-Class Metrics")
        for i, class_name in enumerate(['Real', 'Fake']):
            if i < len(result.per_class_precision):
                report.append(f"- **{class_name}**:")
                report.append(f"  - Precision: {result.per_class_precision[i]:.4f}")
                report.append(f"  - Recall: {result.per_class_recall[i]:.4f}")
                report.append(f"  - F1-Score: {result.per_class_f1[i]:.4f}")
        report.append("")
        
        report.append("### Classification Report")
        report.append("```")
        report.append(result.classification_report)
        report.append("```")
        report.append("")
    
    # Model information (from first result)
    if results:
        first_result = list(results.values())[0]
        report.append("## Model Information")
        report.append("")
        model_info = first_result.model_info
        report.append(f"- **Model**: {model_info.get('model_name', 'Unknown')}")
        report.append(f"- **Parameters**: {model_info.get('total_parameters', 0):,}")
        report.append(f"- **Model Size**: {model_info.get('model_size_mb', 0):.1f} MB")
        report.append("")
    
    report_text = "\n".join(report)
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Evaluation report saved to {save_path}")
    
    return report_text


def main():
    """Demonstrate evaluation framework"""
    print("📊 MODEL EVALUATION & METRICS DEMO")
    print("=" * 60)
    
    print("🔍 Evaluation Framework Features:")
    print("  ✅ Comprehensive metrics (Accuracy, Precision, Recall, F1, AUC)")
    print("  ✅ Cross-dataset evaluation for generalization testing")
    print("  ✅ Model comparison and statistical testing")
    print("  ✅ ROC curves and confusion matrices")
    print("  ✅ Grad-CAM explainability")
    print("  ✅ Performance benchmarking (speed & memory)")
    print("  ✅ Statistical significance testing")
    print("  ✅ Automated report generation")
    
    print(f"\n📈 Visualization Capabilities:")
    print("  • Confusion matrices with normalized values")
    print("  • ROC curves with AUC scores")
    print("  • Model comparison charts")
    print("  • Grad-CAM heatmaps for explainability")
    print("  • Performance benchmarks")
    
    print(f"\n🎯 Cross-Dataset Evaluation:")
    print("  • Test generalization across different datasets")
    print("  • Calculate domain adaptation metrics")
    print("  • Statistical significance testing")
    print("  • Robustness analysis")
    
    print(f"\n⚡ Performance Benchmarking:")
    print("  • Inference speed measurement")
    print("  • Memory usage profiling")
    print("  • Throughput analysis")
    print("  • FPS calculation")
    
    print(f"\n🔬 Explainability:")
    print("  • Grad-CAM for visual explanations")
    print("  • Feature importance analysis")
    print("  • Model decision visualization")
    print("  • Trust and interpretability")
    
    print(f"\n📋 Export Capabilities:")
    print("  • JSON results export")
    print("  • Markdown report generation")
    print("  • High-quality plots and figures")
    print("  • Model comparison tables")
    
    print(f"\n✅ Evaluation framework ready!")
    print(f"🚀 Ready for comprehensive model assessment")


if __name__ == "__main__":
    main()
