import numpy as np

def relu(x):
    return np.maximum(0, x)

class IdentityBlock:
    """
    Identity Block: F(x) + x
    Used when input and output dimensions match.
    """
    
    def __init__(self, channels: int):
        self.channels = channels
        self.W1 = np.random.randn(channels, channels) * 0.01
        self.W2 = np.random.randn(channels, channels) * 0.01
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: y = ReLU(W2 @ ReLU(W1 @ x)) + x
        """
        identity = x

        if x.ndim == 1:
            out = self.W1 @ x
            out = relu(out)
            out = self.W2 @ out
            out = relu(out)
            out = out + identity
        else:
            out = self.W1 @ x.T
            out = relu(out)
            out = self.W2 @ out
            out = relu(out)
            out = out.T + identity

        return out