"""
Generate accuracy graphs from training log
"""

import re
import matplotlib.pyplot as plt
import numpy as np

def parse_training_log(log_file):
    """Parse training log to extract metrics."""
    epochs = []
    train_losses = []
    val_losses = []
    conv_accuracies = []
    nat_f1_scores = []
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    current_epoch = None
    
    for i, line in enumerate(lines):
        # Find epoch start
        epoch_match = re.search(r'Epoch (\d+)/(\d+)', line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
        
        # Find train/val loss (format: "Train Loss: 1.2581 (Conv: 0.2058, Nat: 0.7633)")
        if 'Train Loss:' in line:
            train_match = re.search(r'Train Loss: ([\d.]+)', line)
            if train_match:
                train_losses.append(float(train_match.group(1)))
                if current_epoch:
                    epochs.append(current_epoch)
        
        # Find validation loss
        if 'Val Loss:' in line:
            val_match = re.search(r'Val Loss: ([\d.]+)', line)
            if val_match:
                val_losses.append(float(val_match.group(1)))
        
        # Find final test accuracy
        if 'Conventional Treatment Metrics:' in line:
            # Look ahead for accuracy
            for j in range(i, min(i+15, len(lines))):
                if 'accuracy:' in lines[j] and 'accuracy:' not in lines[j-1] if j > 0 else True:
                    acc_match = re.search(r'accuracy:\s+([\d.]+)', lines[j])
                    if acc_match:
                        conv_accuracies.append(float(acc_match.group(1)))
                        break
        
        if 'Natural Remedies Metrics:' in line:
            # Look ahead for f1_macro
            for j in range(i, min(i+15, len(lines))):
                if 'f1_macro:' in lines[j]:
                    f1_match = re.search(r'f1_macro:\s+([\d.]+)', lines[j])
                    if f1_match:
                        nat_f1_scores.append(float(f1_match.group(1)))
                        break
    
    # Ensure epochs match losses
    if len(epochs) < len(train_losses):
        epochs = list(range(1, len(train_losses) + 1))
    elif len(epochs) > len(train_losses):
        epochs = epochs[:len(train_losses)]
    
    return {
        'epochs': epochs,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'conv_accuracies': conv_accuracies,
        'nat_f1_scores': nat_f1_scores
    }

def create_graphs(data, output_dir='results'):
    """Create and save accuracy graphs."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    epochs = data['epochs']
    train_losses = data['train_losses']
    val_losses = data['val_losses']
    conv_accuracies = data['conv_accuracies']
    nat_f1_scores = data['nat_f1_scores']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Training Metrics', fontsize=16, fontweight='bold')
    
    # 1. Training and Validation Loss
    ax1 = axes[0, 0]
    if len(epochs) > 0 and len(epochs) == len(train_losses):
        ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2, marker='o', markersize=3)
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=3)
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax1.set_title('Training and Validation Loss Over Epochs', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11, loc='upper right')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xlim(left=1)
        if epochs:
            ax1.set_xlim(right=max(epochs))
    else:
        ax1.text(0.5, 0.5, 'No training data available', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=12)
    
    # 2. Conventional Treatment Accuracy
    ax2 = axes[0, 1]
    if conv_accuracies:
        # Show final accuracy as bar
        final_acc = conv_accuracies[-1] if conv_accuracies else 0
        ax2.bar(['Conventional\nTreatment'], [final_acc * 100], 
                color='green', alpha=0.7, width=0.5)
        ax2.axhline(y=80, color='red', linestyle='--', linewidth=2, label='Target (80%)')
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title(f'Conventional Treatment Accuracy\nFinal: {final_acc*100:.2f}%', 
                     fontsize=14, fontweight='bold')
        ax2.set_ylim([0, 100])
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Natural Remedies F1-Score
    ax3 = axes[1, 0]
    if nat_f1_scores:
        # Show final F1 as bar
        final_f1 = nat_f1_scores[-1] if nat_f1_scores else 0
        ax3.bar(['Natural\nRemedies'], [final_f1 * 100], 
                color='blue', alpha=0.7, width=0.5)
        ax3.axhline(y=80, color='red', linestyle='--', linewidth=2, label='Target (80%)')
        ax3.set_ylabel('F1-Score (%)', fontsize=12)
        ax3.set_title(f'Natural Remedies F1-Score\nFinal: {final_f1*100:.2f}%', 
                     fontsize=14, fontweight='bold')
        ax3.set_ylim([0, 100])
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Combined Accuracy Comparison
    ax4 = axes[1, 1]
    if conv_accuracies and nat_f1_scores:
        models = ['Conventional\nTreatment', 'Natural\nRemedies']
        accuracies = [conv_accuracies[-1] * 100, nat_f1_scores[-1] * 100]
        colors = ['green', 'blue']
        
        bars = ax4.bar(models, accuracies, color=colors, alpha=0.7, width=0.6)
        ax4.axhline(y=80, color='red', linestyle='--', linewidth=2, label='Target (80%)')
        ax4.set_ylabel('Accuracy / F1-Score (%)', fontsize=12)
        ax4.set_title('Final Model Performance', fontsize=14, fontweight='bold')
        ax4.set_ylim([0, 100])
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.2f}%',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_path = f'{output_dir}/accuracy_graphs.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Graphs saved to: {output_path}")
    
    # Also create a simpler single graph
    fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
    if conv_accuracies and nat_f1_scores:
        models = ['Conventional Treatment', 'Natural Remedies']
        metrics = [conv_accuracies[-1] * 100, nat_f1_scores[-1] * 100]
        colors = ['#2ecc71', '#3498db']
        
        bars = ax.bar(models, metrics, color=colors, alpha=0.8, width=0.5, edgecolor='black', linewidth=2)
        ax.axhline(y=80, color='red', linestyle='--', linewidth=3, label='Target (80%)', zorder=0)
        ax.set_ylabel('Accuracy / F1-Score (%)', fontsize=14, fontweight='bold')
        ax.set_title('Model Performance - Final Accuracy', fontsize=16, fontweight='bold', pad=20)
        ax.set_ylim([0, 100])
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Add value labels
        for bar, metric in zip(bars, metrics):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{metric:.2f}%',
                    ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        # Add target line annotation
        ax.text(len(models) - 0.5, 82, 'Target: 80%', 
                fontsize=11, color='red', fontweight='bold')
    
    plt.tight_layout()
    output_path2 = f'{output_dir}/final_accuracy_comparison.png'
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"✅ Comparison graph saved to: {output_path2}")
    
    plt.close('all')

if __name__ == '__main__':
    print("="*80)
    print("GENERATING ACCURACY GRAPHS")
    print("="*80)
    
    log_file = 'training_natural_80percent.log'
    
    print(f"\n📊 Parsing training log: {log_file}")
    data = parse_training_log(log_file)
    
    print(f"\n✅ Extracted data:")
    print(f"   Epochs: {len(data['epochs'])}")
    print(f"   Train losses: {len(data['train_losses'])}")
    print(f"   Val losses: {len(data['val_losses'])}")
    if data['conv_accuracies']:
        print(f"   Conventional accuracy: {data['conv_accuracies'][-1]*100:.2f}%")
    if data['nat_f1_scores']:
        print(f"   Natural remedies F1: {data['nat_f1_scores'][-1]*100:.2f}%")
    
    print(f"\n📈 Creating graphs...")
    create_graphs(data)
    
    print(f"\n✅ All graphs generated successfully!")

