"""
Train all 3 models sequentially: without KD, response-based KD, and feature-based KD
This script automates the process of training all models for the lab.
"""

import subprocess
import sys

def modify_train_script(method):
    """Modify train.py to use specified distillation method"""
    with open('train.py', 'r') as f:
        content = f.read()
    
    # Replace the distillation_method line
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'distillation_method = ' in line and not line.strip().startswith('#'):
            lines[i] = f"    distillation_method = '{method}'  # Auto-set by train_all.py"
            break
    
    with open('train.py', 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"✓ Modified train.py to use distillation_method = '{method}'")

def train_model(method, description):
    """Train a model with specified method"""
    print("\n" + "="*80)
    print(f"TRAINING: {description}")
    print("="*80)
    
    modify_train_script(method)
    
    # Run training
    try:
        subprocess.run([sys.executable, 'train.py'], check=True)
        print(f"\n✓ Successfully completed training with method: {method}")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error during training with method: {method}")
        print(f"Error: {e}")
        return False
    
    return True

def main():
    print("="*80)
    print("AUTOMATED TRAINING FOR ALL 3 MODELS")
    print("="*80)
    print("This will train 3 models sequentially:")
    print("  1. Without distillation (baseline)")
    print("  2. Response-based knowledge distillation")
    print("  3. Feature-based knowledge distillation")
    print("\nThis may take a long time depending on your hardware.")
    print("You can also train models individually by editing train.py")
    print("="*80)
    
    response = input("\nContinue with automated training? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled. You can train models individually using train.py")
        return
    
    # Training sequence
    methods = [
        ('none', 'Model WITHOUT Distillation (Baseline)'),
        ('response', 'Model WITH Response-Based Knowledge Distillation'),
        ('feature', 'Model WITH Feature-Based Knowledge Distillation')
    ]
    
    successful = []
    failed = []
    
    for method, description in methods:
        if train_model(method, description):
            successful.append(description)
        else:
            failed.append(description)
            print(f"\nWarning: Training failed for {description}")
            response = input("Continue with remaining models? (y/n): ")
            if response.lower() != 'y':
                break
    
    # Summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"Successfully trained: {len(successful)}/{len(methods)} models")
    
    if successful:
        print("\n✓ Successful:")
        for model in successful:
            print(f"  - {model}")
    
    if failed:
        print("\n✗ Failed:")
        for model in failed:
            print(f"  - {model}")
    
    if len(successful) == len(methods):
        print("\n" + "="*80)
        print("🎉 ALL MODELS TRAINED SUCCESSFULLY!")
        print("="*80)
        print("\nNext step: Run test.py to evaluate all models")
        print("  python test.py")
    else:
        print("\n" + "="*80)
        print("⚠️  SOME MODELS FAILED TO TRAIN")
        print("="*80)
        print("Please check the errors above and train failed models individually.")
    
    print("\nGenerated model files:")
    print("  - best_model_none.pth")
    print("  - best_model_response.pth")
    print("  - best_model_feature.pth")

if __name__ == '__main__':
    main()