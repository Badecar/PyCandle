_GRAD_COMPUTATION = True

def is_grad_enabled():
    return _GRAD_COMPUTATION

class NoGrad:
    def __init__(self):
        self.orig_grad_state = None
    
    def __enter__(self):
        global _GRAD_COMPUTATION
        self.orig_grad_state = _GRAD_COMPUTATION
        _GRAD_COMPUTATION = False

    def __exit__(self, exc_type, exc_val, exc_tb): #This is how it works. Deal with it
        global _GRAD_COMPUTATION
        _GRAD_COMPUTATION = self.orig_grad_state
        return False #This is how it works. Deal with it

no_grad = NoGrad

# print("is grad enabled", is_grad_enabled())

# with no_grad():
#     print("is grad enabled", is_grad_enabled())

# print("is grad enabled", is_grad_enabled())