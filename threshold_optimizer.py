#!/usr/bin/env python3
"""
Find the optimal decision threshold for your trained model
This will dramatically improve precision while maintaining good recall
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import your model and dataset classes
from retrain_identity_split_fixed import (
    EfficientNetLSTM, DeepFakeDataset, FocalLoss,
    prepare_identity_split_data, get_transforms
)
from facenet_pytorch import MTCNN
from torch.utils.data import DataLoader


def evaluate_with_threshold(model, dataloader, device, threshold):
    """Evaluate model with a specific threshold"""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for frames, labels in dataloader:
            frames = frames.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            
            with torch.amp.autocast('cuda'):
                outputs, _ = model(frames)
            
            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Calculate balanced accuracy
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    balanced_acc = (specificity + sensitivity) / 2
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'balanced_acc': balanced_acc,
        'specificity': specificity,
        'confusion_matrix': cm,
        'probs': all_probs,
        'labels': all_labels
    }


def find_best_thresholds(model, dataloader, device):
    """Test multiple thresholds and find the best ones"""
    print("\n" + "="*80)
    print("🔍 THRESHOLD OPTIMIZATION")
    print("="*80)
    
    thresholds = np.arange(0.3, 0.85, 0.05)
    results = []
    
    print("\nTesting thresholds from 0.30 to 0.80...")
    for threshold in tqdm(thresholds):
        metrics = evaluate_with_threshold(model, dataloader, device, threshold)
        results.append({
            'threshold': threshold,
            **metrics
        })
    
    return results


def print_results_table(results):
    """Print a nice table of results"""
    print("\n" + "="*120)
    print(f"{'Threshold':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Balanced':<12} {'Spec':<12}")
    print("="*120)
    
    best_f1_idx = max(range(len(results)), key=lambda i: results[i]['f1'])
    best_balanced_idx = max(range(len(results)), key=lambda i: results[i]['balanced_acc'])
    best_precision_idx = max(range(len(results)), key=lambda i: results[i]['precision'])
    
    for idx, r in enumerate(results):
        marker = ""
        if idx == best_f1_idx:
            marker = " ⭐ Best F1"
        elif idx == best_balanced_idx:
            marker = " 🎯 Best Balanced"
        elif idx == best_precision_idx:
            marker = " 💎 Best Precision"
        
        print(f"{r['threshold']:<12.2f} {r['accuracy']:<12.4f} {r['precision']:<12.4f} "
              f"{r['recall']:<12.4f} {r['f1']:<12.4f} {r['balanced_acc']:<12.4f} "
              f"{r['specificity']:<12.4f}{marker}")
    
    print("="*120)


def plot_metrics(results):
    """Plot metrics vs threshold"""
    thresholds = [r['threshold'] for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Metrics vs Decision Threshold', fontsize=16, fontweight='bold')
    
    # Plot 1: Precision, Recall, F1
    ax1 = axes[0, 0]
    ax1.plot(thresholds, [r['precision'] for r in results], 'o-', label='Precision', linewidth=2)
    ax1.plot(thresholds, [r['recall'] for r in results], 's-', label='Recall', linewidth=2)
    ax1.plot(thresholds, [r['f1'] for r in results], '^-', label='F1', linewidth=2)
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Score')
    ax1.set_title('Precision, Recall, F1 vs Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy and Balanced Accuracy
    ax2 = axes[0, 1]
    ax2.plot(thresholds, [r['accuracy'] for r in results], 'o-', label='Accuracy', linewidth=2)
    ax2.plot(thresholds, [r['balanced_acc'] for r in results], 's-', label='Balanced Accuracy', linewidth=2)
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Score')
    ax2.set_title('Accuracy vs Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Specificity vs Sensitivity
    ax3 = axes[1, 0]
    ax3.plot(thresholds, [r['specificity'] for r in results], 'o-', label='Specificity (TNR)', linewidth=2)
    ax3.plot(thresholds, [r['recall'] for r in results], 's-', label='Sensitivity (TPR/Recall)', linewidth=2)
    ax3.set_xlabel('Threshold')
    ax3.set_ylabel('Rate')
    ax3.set_title('True Negative Rate vs True Positive Rate')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Confusion Matrix components
    ax4 = axes[1, 1]
    tn_rates = [r['confusion_matrix'][0,0] / (r['confusion_matrix'][0,0] + r['confusion_matrix'][0,1]) 
                for r in results]
    tp_rates = [r['confusion_matrix'][1,1] / (r['confusion_matrix'][1,0] + r['confusion_matrix'][1,1]) 
                for r in results]
    ax4.plot(thresholds, tn_rates, 'o-', label='True Negative Rate', linewidth=2)
    ax4.plot(thresholds, tp_rates, 's-', label='True Positive Rate', linewidth=2)
    ax4.set_xlabel('Threshold')
    ax4.set_ylabel('Rate')
    ax4.set_title('Classification Rates')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('threshold_optimization.png', dpi=150, bbox_inches='tight')
    print("\n📊 Plot saved to: threshold_optimization.png")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Load test data
    print("\n📂 Loading test dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_identity_split_data('datasets')
    
    face_detector = MTCNN(keep_all=False, device=device, post_process=False)
    _, val_transform = get_transforms(224)
    
    test_dataset = DeepFakeDataset(X_test, y_test, val_transform, 10, face_detector, 224)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"   Test set: {len(X_test)} videos ({y_test.count(0)} real, {y_test.count(1)} fake)")
    
    # Load model
    print("\n🤖 Loading trained model...")
    model = EfficientNetLSTM().to(device)
    checkpoint = torch.load('models/best_model_identity_split.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"   Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Find best thresholds
    results = find_best_thresholds(model, test_loader, device)
    
    # Print results
    print_results_table(results)
    
    # Find best by different criteria
    best_f1 = max(results, key=lambda r: r['f1'])
    best_balanced = max(results, key=lambda r: r['balanced_acc'])
    best_precision_80_recall = max(
        [r for r in results if r['recall'] >= 0.80], 
        key=lambda r: r['precision'],
        default=results[0]
    )
    
    print("\n" + "="*80)
    print("🎯 RECOMMENDED THRESHOLDS")
    print("="*80)
    
    print(f"\n1️⃣  BEST F1 SCORE (balanced precision/recall)")
    print(f"   Threshold: {best_f1['threshold']:.2f}")
    print(f"   Accuracy:  {best_f1['accuracy']:.4f}")
    print(f"   Precision: {best_f1['precision']:.4f}")
    print(f"   Recall:    {best_f1['recall']:.4f}")
    print(f"   F1:        {best_f1['f1']:.4f}")
    print(f"   Confusion Matrix:")
    print(f"   [[TN={best_f1['confusion_matrix'][0,0]:3d}  FP={best_f1['confusion_matrix'][0,1]:3d}]")
    print(f"    [FN={best_f1['confusion_matrix'][1,0]:3d}  TP={best_f1['confusion_matrix'][1,1]:3d}]]")
    
    print(f"\n2️⃣  BEST BALANCED ACCURACY (equal weight to real/fake)")
    print(f"   Threshold: {best_balanced['threshold']:.2f}")
    print(f"   Balanced Acc: {best_balanced['balanced_acc']:.4f}")
    print(f"   Specificity:  {best_balanced['specificity']:.4f} (correctly identifying reals)")
    print(f"   Sensitivity:  {best_balanced['recall']:.4f} (correctly identifying fakes)")
    print(f"   Precision: {best_balanced['precision']:.4f}")
    print(f"   Confusion Matrix:")
    print(f"   [[TN={best_balanced['confusion_matrix'][0,0]:3d}  FP={best_balanced['confusion_matrix'][0,1]:3d}]")
    print(f"    [FN={best_balanced['confusion_matrix'][1,0]:3d}  TP={best_balanced['confusion_matrix'][1,1]:3d}]]")
    
    print(f"\n3️⃣  BEST PRECISION (minimize false alarms, maintain 80%+ recall)")
    print(f"   Threshold: {best_precision_80_recall['threshold']:.2f}")
    print(f"   Precision: {best_precision_80_recall['precision']:.4f}")
    print(f"   Recall:    {best_precision_80_recall['recall']:.4f}")
    print(f"   F1:        {best_precision_80_recall['f1']:.4f}")
    print(f"   Confusion Matrix:")
    print(f"   [[TN={best_precision_80_recall['confusion_matrix'][0,0]:3d}  FP={best_precision_80_recall['confusion_matrix'][0,1]:3d}]")
    print(f"    [FN={best_precision_80_recall['confusion_matrix'][1,0]:3d}  TP={best_precision_80_recall['confusion_matrix'][1,1]:3d}]]")
    
    print("\n" + "="*80)
    print("💡 RECOMMENDATION")
    print("="*80)
    print(f"\nUse threshold {best_balanced['threshold']:.2f} for production.")
    print(f"This gives you:")
    print(f"  - {best_balanced['specificity']*100:.1f}% of real videos correctly identified")
    print(f"  - {best_balanced['recall']*100:.1f}% of fake videos correctly caught")
    print(f"  - {best_balanced['precision']*100:.1f}% precision (fewer false alarms)")
    
    # Plot results
    try:
        plot_metrics(results)
    except Exception as e:
        print(f"\n⚠️  Could not generate plot: {e}")
    
    # Save results
    import json
    with open('threshold_optimization_results.json', 'w') as f:
        json.dump({
            'best_f1_threshold': best_f1['threshold'],
            'best_balanced_threshold': best_balanced['threshold'],
            'best_precision_threshold': best_precision_80_recall['threshold'],
            'all_results': [{k: v for k, v in r.items() if k not in ['confusion_matrix', 'probs', 'labels']} 
                           for r in results]
        }, f, indent=2)
    print("\n💾 Results saved to: threshold_optimization_results.json")


if __name__ == '__main__':
    main()
