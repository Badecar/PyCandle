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

# Verify everything works
python quick_test.py
```


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

#
We have a FFNN class that has two parameters: num_classes and in_channels.

We have a trainer where you can set:
Num epochs

Learning rate and l2_coeff is an option for the optimizer

batch_size is an option in the dataloader

Wa have implemented a loss function class loss, to make it able to generalize


## 🛠️ Troubleshooting

**Issue: Import errors**
```bash
# Make sure you're in the project directory
cd /path/to/02456_project_group60
python quick_test.py
```

**Issue: WandB not configured**
```bash
wandb login
```

**Issue: Out of memory**
- Reduce batch size in configurations
- Run with fewer epochs
- Use `--quick` mode for testing

**Issue: Experiments too slow**
```bash
# Use quick mode
python run_all_experiments.py --quick

# Or reduce counts
python generate_summary_report.py --all --runs 1
```

See [USAGE_GUIDE.md](USAGE_GUIDE.md) for more troubleshooting tips.

## 📦 Dependencies

Install all:
```bash
pip install -r requirements.txt
```

## 🎯 Recommended Workflow

## 👥 Team

Group 60 - DTU Deep Learning Course 02456

## 📝 License

This project is for educational purposes as part of DTU course 02456.

## 🙏 Acknowledgments

- DTU 02456 Deep Learning course staff
- Weights & Biases for experiment tracking platform
- Fashion-MNIST dataset creators

---