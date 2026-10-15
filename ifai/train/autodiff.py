"""L4 手写自动微分 (Reverse-mode autodiff)

零重写: 复用深度学习标准反向模式自动微分 (Griewank & Walther 2008
"Evaluating Derivatives")；用于 INV-22 手写反向传播 vs 有限差分 校验，
并驱动可训练微型 Transformer 的训练。

实现算子: matmul, add, mul, sub, neg, sigmoid, tanh, relu, exp, log,
mean, sum, transpose, pow, getitem, concat。每个算子实现精确解析梯度。
"""
from __future__ import annotations

import numpy as np


def _unbroadcast(grad, shape):
    """把梯度累加值收缩回 ``shape`` (处理 numpy 广播的反向, INV-22)。"""
    grad = np.asarray(grad)
    if grad.ndim > len(shape):
        diff = grad.ndim - len(shape)
        grad = grad.sum(axis=tuple(range(diff)))
    for i in range(len(shape)):
        if grad.ndim > i and shape[i] == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    def __init__(self, data, _children=(), _op="", requires_grad=True):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None
        self._prev = list(_children)

    @property
    def shape(self):
        return self.data.shape

    def _set(self, val):
        self.data = np.asarray(val, dtype=np.float64)

    # ---- 基础运算 ----
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _b():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad, other.data.shape)
        out._backward = _b
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _b():
            if self.requires_grad:
                self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(self.data * out.grad, other.data.shape)
        out._backward = _b
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        return self + (-other)

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data / other.data, (self, other), "/")

        def _b():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad / other.data, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(-self.data / (other.data ** 2) * out.grad, other.data.shape)
        out._backward = _b
        return out

    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(other.data / self.data, (self, other), "/")

        def _b():
            if self.requires_grad:
                self.grad += _unbroadcast(-other.data / (self.data ** 2) * out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad / self.data, other.data.shape)
        out._backward = _b
        return out

    def __neg__(self):
        out = Tensor(-self.data, (self,), "neg")

        def _b():
            if self.requires_grad:
                self.grad -= out.grad
        out._backward = _b
        return out

    def matmul(self, other):
        out = Tensor(self.data @ other.data, (self, other), "matmul")

        def _b():
            a, b = self.data, other.data
            if self.requires_grad:
                if b.ndim == 1:
                    self.grad += np.outer(out.grad, b)
                else:
                    da = np.matmul(out.grad, np.swapaxes(b, -1, -2))
                    self.grad += _unbroadcast(da, a.shape)
            if other.requires_grad:
                if b.ndim == 1:
                    other.grad += a.T @ out.grad
                else:
                    db = np.matmul(np.swapaxes(a, -1, -2), out.grad)
                    other.grad += _unbroadcast(db, b.shape)
        out._backward = _b
        return out

    def transpose(self, axes=None):
        out = Tensor(np.transpose(self.data, axes), (self,), "T")

        def _b():
            if self.requires_grad:
                if axes is None:
                    inv = tuple(reversed(range(self.data.ndim)))
                else:
                    inv = tuple(np.argsort(axes))
                self.grad += np.transpose(out.grad, inv)
        out._backward = _b
        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), (self,), "reshape")

        def _b():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.data.shape)
        out._backward = _b
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, (self,), "sigmoid")

        def _b():
            if self.requires_grad:
                self.grad += s * (1 - s) * out.grad
        out._backward = _b
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh")

        def _b():
            if self.requires_grad:
                self.grad += (1 - t * t) * out.grad
        out._backward = _b
        return out

    def relu(self):
        m = self.data > 0
        out = Tensor(np.where(m, self.data, 0.0), (self,), "relu")

        def _b():
            if self.requires_grad:
                self.grad += m * out.grad
        out._backward = _b
        return out

    def exp(self):
        e = np.exp(self.data)
        out = Tensor(e, (self,), "exp")

        def _b():
            if self.requires_grad:
                self.grad += e * out.grad
        out._backward = _b
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), "log")

        def _b():
            if self.requires_grad:
                self.grad += out.grad / self.data
        out._backward = _b
        return out

    def mean(self, axis=None, keepdims=False):
        out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), (self,), "mean")

        def _b():
            if self.requires_grad:
                denom = self.data.size if axis is None else self.data.shape[axis]
                if axis is None:
                    self.grad += np.ones_like(self.data) * (out.grad / denom)
                elif keepdims:
                    self.grad += np.broadcast_to(out.grad, self.data.shape) / denom
                else:
                    shape = [1] * self.data.ndim
                    shape[axis] = self.data.shape[axis]
                    self.grad += np.broadcast_to(np.expand_dims(out.grad, axis), self.data.shape) / denom
        out._backward = _b
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), (self,), "sum")

        def _b():
            if self.requires_grad:
                if axis is None:
                    self.grad += np.ones_like(self.data) * out.grad
                elif keepdims:
                    self.grad += np.broadcast_to(out.grad, self.data.shape)
                else:
                    self.grad += np.broadcast_to(np.expand_dims(out.grad, axis), self.data.shape)
        out._backward = _b
        return out

    def max(self, axis=None, keepdims=False):
        out = Tensor(np.max(self.data, axis=axis, keepdims=keepdims), (self,), "max")

        def _b():
            if self.requires_grad:
                if axis is None:
                    mx = self.data.max()
                    mask = (self.data == mx)
                    self.grad += mask * out.grad / max(int(mask.sum()), 1)
                else:
                    mx = np.max(self.data, axis=axis, keepdims=True)
                    mask = (self.data == mx)
                    if keepdims:
                        self.grad += mask * np.broadcast_to(out.grad, self.data.shape)
                    else:
                        self.grad += mask * np.broadcast_to(np.expand_dims(out.grad, axis), self.data.shape)
        out._backward = _b
        return out

    def pow(self, p):
        out = Tensor(self.data ** p, (self,), "pow")

        def _b():
            if self.requires_grad:
                self.grad += p * (self.data ** (p - 1)) * out.grad
        out._backward = _b
        return out

    def __getitem__(self, idx):
        out = Tensor(self.data[idx], (self,), "getitem")

        def _b():
            if self.requires_grad:
                g = np.zeros_like(self.data)
                g[idx] = out.grad
                self.grad += g
        out._backward = _b
        return out

    # ---- 反向传播 ----
    def backward(self):
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for ch in v._prev:
                    build(ch)
                topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    @staticmethod
    def zeros(*shape):
        return Tensor(np.zeros(shape), requires_grad=False)

    @staticmethod
    def parameter(data):
        return Tensor(data, requires_grad=True)
