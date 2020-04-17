import numpy as np
from pythano.tensor import Tensor


class Layer(object):
    def forward(self):
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError

    def zero_grad(self):
        for param in self.parameters():
            param.grad = 0

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Linear(Layer):
    def __init__(self, in_dim, out_dim, bias=True):
        # Kaiming initialization for relus
        self.linear = Tensor(np.random.randn(in_dim, out_dim) * np.sqrt(2/in_dim))
        self.bias = Tensor(np.zeros(1, out_dim))

    def forward(self, x):
        x = x if isinstance(x, Tensor) else Tensor(data=x, requires_grad=False)
        return self.x.dot(self.linear) + self.bias

    def parameters(self):
        return [self.linear, self.bias]

class ReLU(Layer):
