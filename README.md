# FFNN From Scratch - Deep Learning Project

**Authors:** David Lindahl (S234817), Benjamin Banks (S234802), Oscar Svendsen (S224177), Mikkel Broch-Lips (S234860)

Implementation of a fully-connected feedforward neural network from scratch using only NumPy, with comprehensive hyperparameter sweeps and experiment tracking using Weights & Biases.

## 🎯 Project Overview

This project implements a neural network from scratch for the DTU course 02456 Deep Learning. We've built:

- Custom neural network framework using only NumPy
- Forward and backward propagation
- Multiple optimizers (SGD, SGDMomentum, ADAM)
- Various activation functions (ReLU, Sigmoid)
- Different initialization strategies (Normal, Uniform)
- Comprehensive experiment tracking with WandB


## 🚀 Quick Start

### 1. Setup



```bash
# Install dependencies
pip install -r requirements.txt

# Login to Weights & Biases
wandb login
```

### 2.a Run the main.py file

```bash
python main.py
```

### 2.b For the CNN, run the main_CNN.py file

```bash
python main_CNN.py
```


### 3. If you want to run experiments, you can do the following:

**All experiments are now unified in `experiments.py`**

```bash
# Compare activation functions (ReLU vs Sigmoid)
python experiments.py --mode compare --compare-type activation --runs 3

# Compare initializers (Normal vs Uniform)
python experiments.py --mode compare --compare-type initializer --runs 3

# Run hyperparameter sweep (Bayesian optimization)
python experiments.py --mode sweep --sweep-count 20 --sweep-method bayes

# Run all comparisons
python experiments.py --mode compare-all --runs 3
```

### 4. View Results

Go to https://wandb.ai and open your project to see:
- Real-time learning curves
- Parameter histograms
- Gradient norms
- Hyperparameter sweep results
- Parallel coordinates plots


## 🎓 Project Requirements

This implementation fulfills all project requirements:

✅ **Forward pass:** Matrix multiplications + activation functions  
✅ **Loss computation:** Cross-entropy with L2 regularization  
✅ **Backward pass:** Manual derivative calculation and weight updates  
✅ **Training loop:** Mini-batch gradient descent  
✅ **Evaluation:** Accuracy, loss curves, confusion matrices  
✅ **WandB logging:** Learning curves, parameter histograms, gradient norms  
✅ **Hyperparameter sweeps:** Random and Bayesian search ⭐  
✅ **Summary reports:** Comparing activations and initializations ⭐  


## 📊 Experiment Features

The unified `experiments.py` script provides:

**Training with WandB Logging:**
- Learning curves (train_loss, val_loss, accuracy, val_acc)
- Parameter histograms (built-in + custom charts) per layer, per epoch 📊
- Gradient histograms and norms (detect vanishing/exploding gradients) 📊
- Weight statistics (mean, std, min, max)
- Real-time monitoring of training progress

**Hyperparameter Sweeps:**
- Random, Bayesian, or Grid search
- Automatic tracking of all configurations
- Parallel coordinates plots showing parameter relationships
- Parameter importance rankings

**Comparison Experiments:**
- Activation functions (ReLU vs Sigmoid)
- Initializers (Normal vs Uniform)
- Optimizers (SGD, SGDMomentum, ADAM)
- Multiple runs with statistical analysis

## 🛠️ Troubleshooting

**Issue: Import errors**
```bash
# Make sure you're in the project directory
cd /path/to/02456_project_group60
python experiments.py --mode single --epochs 2
```

**Issue: WandB not configured**
```bash
wandb login
```

**Issue: Out of memory**
- Reduce batch size: `--batch-size 128` or `--batch-size 64`
- Run with fewer epochs: `--epochs 5`
- Reduce number of runs: `--runs 1`

**Issue: Experiments too slow**
```bash
# Quick test with minimal configuration
python experiments.py --mode single --epochs 2 --batch-size 64

# Fewer comparison runs
python experiments.py --mode compare --compare-type activation --runs 1

# Smaller sweep
python experiments.py --mode sweep --sweep-count 10
```

## 👥 Team

Group 60 - DTU Deep Learning Course 02456

## 📝 License

This project is for educational purposes as part of DTU course 02456.

## 🙏 Acknowledgments

- DTU 02456 Deep Learning course staff
- Weights & Biases for experiment tracking platform
- Fashion-MNIST dataset creators

---