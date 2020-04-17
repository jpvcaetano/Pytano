import numpy as np


class Tensor(object):
    def __init__(self, data, requires_grad=True, _parents=()):
        self.data = data if isinstance(data, np.ndarray) else np.array(data)
        self.requires_grad = requires_grad
        self.grad = 0
        # for backpropagation
        self._parents = _parents
        self._backprop = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(data=other, requires_grad=False)
        out = Tensor(data=self.data + other.data,
                     requires_grad=(self.requires_grad or other.requires_grad),
                     _parents=(self, other))

        def _backprop():
            self.grad += self.out.grad
            other.grad += self.out.grad
        out._backprop = _backprop

        return out

    def dot(self, other):
        other = other if isinstance(other, Tensor) else Tensor(data=other, requires_grad=False)
        return Tensor(data=self.data.dot(other.data),
                     requires_grad=(self.requires_grad or other.requires_grad),
                     _parents=(self, other))

        def _backprop():
            self.grad += self.out.grad.dot(other.data)
            other.grad += self.out.grad.dot(self.data)
        out._backprop = _backprop

        return out

    def __str__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad}, grad={self.grad})"






