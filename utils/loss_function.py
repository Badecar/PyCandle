from var import Tensor
from math import exp, log
import numpy as np

def cross_entropy_loss(t: Tensor, h: Tensor):
    """Cross entropy loss function for multi-class classification.
    
    Computes the cross entropy loss between true labels (t) and predicted logits (h).
    The loss is calculated as: -sum(t_i * log(softmax(h_i))) for each sample.
    
    Args:
        t: True labels (one-hot encoded)
        h: Predicted logits
        
    Returns:
        Cross entropy loss value
    """
    # Numerically stable log-softmax
    # 1. Subtract the max for stability
    h_max = Tensor(h.v.max(axis=1, keepdims=True))
    log_softmax = h - h_max - (h - h_max).exp().sum(axis=1, keepdims=True).log()
    
    # 2. Compute the negative log-likelihood loss
    # The multiplication with `t` selects the correct log-probability.
    # The negative sign makes it a loss to be minimize d.
    # The final sum averages the loss over the batch.
    loss = -(t * log_softmax).sum()
    
    return loss


if __name__ == "__main__":
    # Targets (Indices: 3, 0)
    t = Tensor([[0,0,0,1],[1,0,0,0]])

    # FIX: Use Logits (Raw scores), not Probabilities

    # 1. Perfect prediction (High score for correct class, low for others)
    # Softmax of [..., 10] results in probability ~0.9999
    pred_1 = Tensor([[-10,-10,-10,10],[10,-10,-10,-10]]) 

    # 2. Random/Uniform prediction (All scores equal)
    # Softmax of [0,0,0,0] results in probability 0.25 each
    pred_2 = Tensor([[0,0,0,0],[0,0,0,0]])

    # 3. Wrong prediction (High score for wrong class)
    # We predict class 0 (index 0) for the first sample (target is index 3)
    pred_3 = Tensor([[10,-10,-10,-10],[-10,10,-10,-10]])

    # Doing the test
    CE_1 = cross_entropy_loss(t,pred_1)
    CE_2 = cross_entropy_loss(t,pred_2)
    CE_3 = cross_entropy_loss(t,pred_3)

    # Display
    print(f"Loss for pred 1 (perfect): {CE_1.v:.4f}") # Expected: ~0.0000
    print(f"Loss for pred 2 (random):  {CE_2.v:.4f}") # Expected: ~2.7726 (-ln(0.25) * 2)
    print(f"Loss for pred 3 (wrong):   {CE_3.v:.4f}") # Expected: ~20.00 or higher