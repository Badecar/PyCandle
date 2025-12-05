"""
Comprehensive Experiment Runner with WandB Integration
This script provides all experiment functionality in one place:
- Training with WandB logging (learning curves, parameter histograms, gradient norms)
- Hyperparameter sweeps (random or Bayesian)
- Activation function and initialization comparison
- Summary reports with visualizations
"""

import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.dataloader import DataLoader, train_dataset, test_dataset
from utils.loss_function import cross_entropy_loss
from utils.cn import *  # Import this first to avoid circular import issues
from utils.initializer import NormalInitializer, UniformInitializer
from utils.optimizer import SGD, SGDMomentum, ADAM
from training.traning import eval_model
import json
import os
from datetime import datetime
import argparse

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class ExperimentRunner:
    """Unified experiment runner with WandB integration"""
    
    def __init__(self, project_name="ffnn-experiments", use_wandb=True):
        self.project_name = project_name
        self.use_wandb = use_wandb
        self.results = []
    
    def build_model(self, config):
        """Build a model based on configuration"""
        # Select activation function
        if config['activation'] == 'ReLU':
            act_fn = ReLU
        elif config['activation'] == 'Sigmoid':
            act_fn = Sigmoid
        else:
            raise ValueError(f"Unknown activation: {config['activation']}")
        
        # Select initializer
        if config['initializer'] == 'Normal':
            initializer = NormalInitializer()
        elif config['initializer'] == 'Uniform':
            initializer = UniformInitializer()
        else:
            raise ValueError(f"Unknown initializer: {config['initializer']}")
        
        # Get architecture (support both list and individual hidden layers)
        if 'architecture' in config:
            arch = config['architecture']
        else:
            arch = [config.get('hidden_layer_1', 600),
                   config.get('hidden_layer_2', 600),
                   config.get('hidden_layer_3', 120)]
        
        # Build model
        class Model(Module):
            def __init__(self):
                super().__init__()
                self.layers = Sequential(
                    Flatten(),
                    Linear(n_in=28*28, n_out=arch[0], bias=True, initializer=initializer),
                    act_fn(),
                    Linear(n_in=arch[0], n_out=arch[1], bias=True, initializer=initializer),
                    act_fn(),
                    Linear(n_in=arch[1], n_out=arch[2], bias=True, initializer=initializer),
                    act_fn(),
                    Linear(n_in=arch[2], n_out=10, bias=True, initializer=initializer)
                )
            
            def forward(self, x):
                return self.layers(x)
        
        return Model()
    
    def get_optimizer(self, optimizer_name, params, lr):
        """Get optimizer based on name"""
        if optimizer_name == 'SGD':
            return SGD(params, lr=lr)
        elif optimizer_name == 'SGDMomentum':
            return SGDMomentum(params, lr=lr)
        elif optimizer_name == 'ADAM':
            return ADAM(params, lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def train_model_with_logging(self, model, train_loader, val_loader, optimizer, 
                                 criterion, num_epochs, config_name, log_histograms=True):
        """
        Train model with comprehensive WandB logging:
        - Learning curves (train_loss, train_acc, val_loss, val_acc)
        - Parameter histograms
        - Gradient norms
        """
        loss_list = []
        
        for epoch in range(num_epochs):
            model.train()
            epoch_grad_norms = []
            
            for i, batch in enumerate(train_loader):
                x, y = batch
                
                output = model(x)
                loss = criterion(y, output)
                loss.backward()
                
                # Compute gradient norms and histograms before optimizer step
                if log_histograms and self.use_wandb:
                    for param_idx, param in enumerate(model.parameters()):
                        if param.grad is not None:
                            grad_norm = np.linalg.norm(param.grad.flatten())
                            epoch_grad_norms.append(grad_norm)
                            
                            # Log gradient norm
                            wandb.log({f"gradients/layer_{param_idx}_norm": grad_norm})
                            
                            # Log gradient histogram (every 10 batches to avoid too much data)
                            if i % 10 == 0:
                                grad_values = param.grad.flatten()
                                wandb.log({
                                    f"gradients/layer_{param_idx}_histogram": wandb.Histogram(grad_values),
                                    f"gradients/layer_{param_idx}_mean": np.mean(grad_values),
                                    f"gradients/layer_{param_idx}_std": np.std(grad_values)
                                })
                elif log_histograms:  # log_histograms but not wandb
                    for param_idx, param in enumerate(model.parameters()):
                        if param.grad is not None:
                            grad_norm = np.linalg.norm(param.grad.flatten())
                            epoch_grad_norms.append(grad_norm)
                
                optimizer.step()
                optimizer.zero_grad()
                
                # Calculate training accuracy
                y_true = np.argmax(y.v, axis=1)
                y_pred = np.argmax(output.v, axis=1)
                acc = (y_true == y_pred).sum() / len(y_true)
                
                print(f"[{config_name}] Epoch {epoch}, Batch {i}, Loss: {loss.v:.4f}, Acc: {acc:.4f}")
                loss_list.append(loss.v)
                
                if self.use_wandb:
                    wandb.log({
                        "train/loss": loss.v,
                        "train/acc": acc,
                        "train/step": epoch * len(train_loader) + i
                    })
            
            # Log parameter histograms at end of epoch
            if log_histograms and self.use_wandb:
                for param_idx, param in enumerate(model.parameters()):
                    # Log basic statistics
                    wandb.log({
                        f"params/layer_{param_idx}_mean": np.mean(param.v),
                        f"params/layer_{param_idx}_std": np.std(param.v),
                        f"params/layer_{param_idx}_max": np.max(param.v),
                        f"params/layer_{param_idx}_min": np.min(param.v),
                    })
                    
                    # Log built-in histogram (auto-binning)
                    wandb.log({
                        f"params/layer_{param_idx}_histogram": wandb.Histogram(param.v.flatten())
                    })
                    
                    # Log custom histogram chart for better visualization
                    param_values = param.v.flatten()
                    table_data = [[i, val] for i, val in enumerate(param_values)]
                    table = wandb.Table(
                        data=table_data,
                        columns=["param_index", "value"]
                    )
                    histogram_chart = wandb.plot.histogram(
                        table,
                        value="value",
                        title=f"Layer {param_idx} Weight Distribution (Epoch {epoch})"
                    )
                    wandb.log({
                        f"params/layer_{param_idx}_custom_histogram": histogram_chart
                    })
            
            # Validation step
            if val_loader is not None:
                val_metrics = eval_model(model, val_loader, plot_confusion_matrix=False)
                print(f"[{config_name}] Epoch {epoch} Validation: Loss: {val_metrics['cross_entropy']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
                
                if self.use_wandb:
                    wandb.log({
                        "val/loss": val_metrics['cross_entropy'],
                        "val/acc": val_metrics['accuracy'],
                        "val/f1": val_metrics['f1_mean'],
                        "epoch": epoch
                    })
        
        return model, loss_list
    
    def run_single_experiment(self, config, experiment_name):
        """Run a single experiment with given configuration"""
        print(f"\n{'='*60}")
        print(f"Running: {experiment_name}")
        print(f"Config: {config}")
        print(f"{'='*60}\n")
        
        # Initialize wandb if enabled
        if self.use_wandb:
            run = wandb.init(
                project=self.project_name,
                name=experiment_name,
                config=config,
                reinit=True
            )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                                 shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                                shuffle=True, drop_last=True)
        
        # Build model
        model = self.build_model(config)
        
        # Get optimizer
        optimizer = self.get_optimizer(config['optimizer'], model.parameters(), 
                                      config['learning_rate'])
        
        # Train model
        model, loss_list = self.train_model_with_logging(
            model, train_loader, test_loader, optimizer, cross_entropy_loss,
            config['num_epochs'], experiment_name, log_histograms=config.get('log_histograms', True)
        )
        
        # Final evaluation
        final_metrics = eval_model(model, test_loader, plot_confusion_matrix=False)
        
        # Log final metrics
        if self.use_wandb:
            wandb.log({
                "final/test_accuracy": final_metrics['accuracy'],
                "final/test_f1": final_metrics['f1_mean'],
                "final/test_loss": final_metrics['cross_entropy']
            })
            
            wandb.summary.update({
                'test_accuracy': final_metrics['accuracy'],
                'test_f1': final_metrics['f1_mean'],
                'test_loss': final_metrics['cross_entropy']
            })
            
            run.finish()
        
        print(f"\n[{experiment_name}] Final Results:")
        print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
        print(f"  F1 Score: {final_metrics['f1_mean']:.4f}")
        print(f"  Loss: {final_metrics['cross_entropy']:.4f}\n")
        
        # Store results
        result = {
            'experiment_name': experiment_name,
            'config': config,
            'metrics': {
                'accuracy': float(final_metrics['accuracy']),
                'f1_mean': float(final_metrics['f1_mean']),
                'cross_entropy': float(final_metrics['cross_entropy'])
            },
            'loss_history': [float(x) for x in loss_list]
        }
        self.results.append(result)
        
        return result
    
    def run_comparison_experiments(self, comparison_type='activation', num_runs=3, 
                                   base_config=None):
        """
        Run comparison experiments for activation functions, initializers, or optimizers
        
        Args:
            comparison_type: 'activation', 'initializer', or 'optimizer'
            num_runs: Number of runs per configuration
            base_config: Base configuration dictionary
        """
        if base_config is None:
            base_config = {
                'batch_size': 256,
                'num_epochs': 10,
                'learning_rate': 0.001,
                'optimizer': 'ADAM',
                'activation': 'ReLU',
                'initializer': 'Normal',
                'architecture': [600, 600, 120],
                'log_histograms': True
            }
        
        print(f"\n{'#'*60}")
        print(f"COMPARING {comparison_type.upper()}S")
        print(f"{'#'*60}")
        
        if comparison_type == 'activation':
            options = ['ReLU', 'Sigmoid']
            config_key = 'activation'
        elif comparison_type == 'initializer':
            options = ['Normal', 'Uniform']
            config_key = 'initializer'
        elif comparison_type == 'optimizer':
            options = ['SGD', 'SGDMomentum', 'ADAM']
            config_key = 'optimizer'
        else:
            raise ValueError(f"Unknown comparison type: {comparison_type}")
        
        for option in options:
            for run_num in range(num_runs):
                config = base_config.copy()
                config[config_key] = option
                experiment_name = f"{comparison_type}_{option}_run{run_num+1}"
                
                self.run_single_experiment(config, experiment_name)
        
        print(f"\n{'='*60}")
        print(f"{comparison_type.upper()} COMPARISON COMPLETE")
        print(f"{'='*60}\n")
    
    def save_results(self, save_dir='reports'):
        """Save all results to JSON file"""
        os.makedirs(save_dir, exist_ok=True)
        
        json_path = os.path.join(save_dir, f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Saved results: {json_path}")
        return json_path


class SweepRunner:
    """Hyperparameter sweep runner with WandB"""
    
    def __init__(self, project_name="ffnn-sweeps"):
        self.project_name = project_name
    
    def create_sweep_config(self, method='bayes'):
        """Create sweep configuration"""
        return {
            'method': method,
            'metric': {
                'name': 'val/acc',
                'goal': 'maximize'
            },
            'parameters': {
                'learning_rate': {
                    'distribution': 'log_uniform_values',
                    'min': 0.0001,
                    'max': 0.01
                },
                'batch_size': {
                    'values': [64, 128, 256, 512]
                },
                'optimizer': {
                    'values': ['SGD', 'SGDMomentum', 'ADAM']
                },
                'num_epochs': {
                    'values': [10, 15, 20]
                },
                'hidden_layer_1': {
                    'values': [256, 512, 600, 800]
                },
                'hidden_layer_2': {
                    'values': [256, 512, 600, 800]
                },
                'hidden_layer_3': {
                    'values': [64, 120, 256]
                },
                'activation': {
                    'values': ['ReLU', 'Sigmoid']
                },
                'initializer': {
                    'values': ['Normal', 'Uniform']
                }
            }
        }
    
    def train_sweep_run(self):
        """Training function for one sweep run"""
        run = wandb.init()
        config = wandb.config
        
        # Log configuration
        print(f"\n{'='*60}")
        print(f"Sweep run with configuration:")
        print(f"  Learning Rate: {config.learning_rate}")
        print(f"  Batch Size: {config.batch_size}")
        print(f"  Optimizer: {config.optimizer}")
        print(f"  Epochs: {config.num_epochs}")
        print(f"  Architecture: [{config.hidden_layer_1}, {config.hidden_layer_2}, {config.hidden_layer_3}]")
        print(f"  Activation: {config.activation}")
        print(f"  Initializer: {config.initializer}")
        print(f"{'='*60}\n")
        
        # Create experiment runner
        runner = ExperimentRunner(project_name=self.project_name, use_wandb=False)
        
        # Create config dict
        config_dict = {
            'batch_size': config.batch_size,
            'num_epochs': config.num_epochs,
            'learning_rate': config.learning_rate,
            'optimizer': config.optimizer,
            'activation': config.activation,
            'initializer': config.initializer,
            'hidden_layer_1': config.hidden_layer_1,
            'hidden_layer_2': config.hidden_layer_2,
            'hidden_layer_3': config.hidden_layer_3,
            'log_histograms': False  # Disable histograms for faster sweeps
        }
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                                 shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, 
                                shuffle=True, drop_last=True)
        
        # Build model
        model = runner.build_model(config_dict)
        
        # Get optimizer
        optimizer = runner.get_optimizer(config.optimizer, model.parameters(), 
                                        config.learning_rate)
        
        # Train model
        runner.use_wandb = True  # Enable wandb for logging
        model, loss_list = runner.train_model_with_logging(
            model, train_loader, test_loader, optimizer, cross_entropy_loss,
            config.num_epochs, "sweep_run", log_histograms=False
        )
        
        # Final evaluation
        final_metrics = eval_model(model, test_loader, plot_confusion_matrix=False)
        
        # Log final metrics
        wandb.log({
            "final/test_accuracy": final_metrics['accuracy'],
            "final/test_f1": final_metrics['f1_mean'],
            "final/test_loss": final_metrics['cross_entropy']
        })
        
        wandb.summary.update({
            'test_accuracy': final_metrics['accuracy'],
            'test_f1': final_metrics['f1_mean'],
            'test_loss': final_metrics['cross_entropy']
        })
        
        print(f"\nFinal Test Accuracy: {final_metrics['accuracy']:.4f}")
        print(f"Final Test F1 Score: {final_metrics['f1_mean']:.4f}")
        
        run.finish()
    
    def run_sweep(self, count=20, method='bayes'):
        """
        Run hyperparameter sweep
        
        Args:
            count: Number of runs to execute
            method: Sweep method ('random', 'grid', or 'bayes')
        """
        sweep_config = self.create_sweep_config(method=method)
        sweep_config['name'] = f'{method}_sweep_{count}runs'
        
        # Initialize sweep
        sweep_id = wandb.sweep(sweep_config, project=self.project_name)
        
        print(f"\n{'='*60}")
        print(f"Starting {method.upper()} hyperparameter sweep")
        print(f"Sweep ID: {sweep_id}")
        print(f"Number of runs: {count}")
        print(f"Project: {self.project_name}")
        print(f"{'='*60}\n")
        
        # Run sweep agent
        wandb.agent(sweep_id, self.train_sweep_run, count=count)
        
        print(f"\n{'='*60}")
        print(f"🎉 Sweep completed!")
        print(f"View results at: https://wandb.ai")
        print(f"{'='*60}\n")


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description='Comprehensive experiment runner with WandB integration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run activation function comparison
  python experiments.py --mode compare --compare-type activation --runs 3
  
  # Run initializer comparison
  python experiments.py --mode compare --compare-type initializer --runs 3
  
  # Run optimizer comparison
  python experiments.py --mode compare --compare-type optimizer --runs 3
  
  # Run hyperparameter sweep
  python experiments.py --mode sweep --sweep-count 20 --sweep-method bayes
  
  # Run single experiment with custom config
  python experiments.py --mode single --activation ReLU --initializer Normal --optimizer ADAM
  
  # Run all comparisons
  python experiments.py --mode compare-all --runs 3
        """
    )
    
    parser.add_argument('--mode', type=str, required=True,
                       choices=['single', 'compare', 'compare-all', 'sweep'],
                       help='Experiment mode')
    
    # Comparison arguments
    parser.add_argument('--compare-type', type=str, 
                       choices=['activation', 'initializer', 'optimizer'],
                       help='Type of comparison (required for compare mode)')
    parser.add_argument('--runs', type=int, default=3,
                       help='Number of runs per configuration (default: 3)')
    
    # Single experiment arguments
    parser.add_argument('--activation', type=str, default='ReLU',
                       choices=['ReLU', 'Sigmoid'],
                       help='Activation function (default: ReLU)')
    parser.add_argument('--initializer', type=str, default='Normal',
                       choices=['Normal', 'Uniform'],
                       help='Weight initializer (default: Normal)')
    parser.add_argument('--optimizer', type=str, default='ADAM',
                       choices=['SGD', 'SGDMomentum', 'ADAM'],
                       help='Optimizer (default: ADAM)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size (default: 256)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    
    # Sweep arguments
    parser.add_argument('--sweep-count', type=int, default=20,
                       help='Number of sweep runs (default: 20)')
    parser.add_argument('--sweep-method', type=str, default='bayes',
                       choices=['random', 'bayes', 'grid'],
                       help='Sweep method (default: bayes)')
    
    # General arguments
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    parser.add_argument('--project', type=str, default='ffnn-experiments',
                       help='WandB project name (default: ffnn-experiments)')
    
    args = parser.parse_args()
    
    # Mode: Single experiment
    if args.mode == 'single':
        print(f"\n{'='*60}")
        print("RUNNING SINGLE EXPERIMENT")
        print(f"{'='*60}\n")
        
        runner = ExperimentRunner(project_name=args.project, use_wandb=not args.no_wandb)
        
        config = {
            'batch_size': args.batch_size,
            'num_epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'optimizer': args.optimizer,
            'activation': args.activation,
            'initializer': args.initializer,
            'architecture': [600, 600, 120],
            'log_histograms': True
        }
        
        runner.run_single_experiment(config, 'single_experiment')
        runner.save_results()
    
    # Mode: Comparison
    elif args.mode == 'compare':
        if args.compare_type is None:
            parser.error("--compare-type is required for compare mode")
        
        runner = ExperimentRunner(project_name=args.project, use_wandb=not args.no_wandb)
        runner.run_comparison_experiments(
            comparison_type=args.compare_type,
            num_runs=args.runs
        )
        runner.save_results()
    
    # Mode: Compare all
    elif args.mode == 'compare-all':
        print(f"\n{'='*60}")
        print("RUNNING ALL COMPARISONS")
        print(f"{'='*60}\n")
        
        runner = ExperimentRunner(project_name=args.project, use_wandb=not args.no_wandb)
        
        for comp_type in ['activation', 'initializer', 'optimizer']:
            runner.run_comparison_experiments(
                comparison_type=comp_type,
                num_runs=args.runs
            )
        
        runner.save_results()
    
    # Mode: Hyperparameter sweep
    elif args.mode == 'sweep':
        if args.no_wandb:
            print("ERROR: Cannot run sweep without WandB enabled")
            return
        
        sweep_runner = SweepRunner(project_name=args.project)
        sweep_runner.run_sweep(count=args.sweep_count, method=args.sweep_method)
    
    print(f"\n{'='*60}")
    print("✅ EXPERIMENTS COMPLETE!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
