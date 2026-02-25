# PyCandle

PyCandle is a lightweight deep learning framework built with NumPy, with custom autograd, layers, optimizers, and training loops for Fashion-MNIST classification.

It includes both fully connected and convolutional model paths, plus experiment scripts for configuration sweeps and result tracking.

## Quick Start

```bash
pip install -r requirements.txt
```

Optional (for experiment tracking):

```bash
wandb login
```

## Run and Test with Main Files

Use the main scripts to quickly verify the setup and run end-to-end training/evaluation:

```bash
# Feedforward network baseline
python main.py

# Convolutional network baseline
python main_CNN.py
```

## Repository Layout

### Entry Points

- `main.py` - baseline FFNN training and evaluation run.
- `main_CNN.py` - baseline CNN training and evaluation run.
- `experiments.py` - unified CLI for single runs, model/component comparisons, and hyperparameter sweeps.
- `CNN_pytorch_comparison.py` - PyTorch reference implementation for comparison with the custom framework.

### Core Framework (`utils/`)

- `utils/tensor.py` - core `Tensor` implementation with autograd and numerical ops.
- `utils/cn.py` - neural network building blocks (`Module`, `Parameter`, `Linear`, `Conv2D`, `MaxPool2D`, activations, `Sequential`).
- `utils/candle.py` - gradient mode controls (`no_grad` context and global grad state).
- `utils/optimizer.py` - optimizers (`SGD`, `SGDMomentum`, `ADAM`).
- `utils/loss_function.py` - cross-entropy loss.
- `utils/initializer.py` - weight/bias initialization strategies.
- `utils/dataloader.py` - Fashion-MNIST download/loading and custom `Dataset`/`DataLoader`.

### Models, Training, and Plotting

- `models/ffnn.py` - configurable FFNN definition.
- `models/ffnn_standard.py` - standard FFNN variant.
- `models/ffnn_wide.py` - wider FFNN variant.
- `models/cnn.py` - CNN architecture built on the custom layers.
- `training/traning.py` - training loop, evaluation, metrics, and confusion matrix support.
- `plotting/plotting.py` - confusion matrix visualization helpers.

## Experiments

`experiments.py` supports:

- single configurable runs
- activation/initializer/optimizer comparisons
- hyperparameter sweeps with Weights & Biases

Example:

```bash
python experiments.py --mode single --epochs 2 --batch-size 64
```