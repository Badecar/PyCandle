from utils.tensor import Tensor

def cross_entropy_loss(t: Tensor, x: Tensor):

    # Compute log_softmax = log(exp(x) / sum(exp(x))) = x - log(sum(exp(x)))
    log_sum_exp = x.exp().sum(axis=1, keepdims=True).log()
    log_softmax = x - log_sum_exp

    # Compute the negative log-likelihood loss
    # The multiplication with `t` selects the correct log-probability.
    # Sum over all samples and classes, then divide by batch size
    batch_size = t.v.shape[0]
    loss = -(t * log_softmax).sum() * Tensor(1.0 / batch_size)
    
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