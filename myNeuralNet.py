import random
import math

class Value:
    def __init__(self, value, _prev=()):
        self.value = value
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_prev)
        
    def backward(self):
        def topological_sort(node):
            visited = set()
            sorted_nodes = []

            def dfs(node):
                if node in visited:
                    return
                visited.add(node)
                for child in node._prev:
                    dfs(child)
                sorted_nodes.append(node)

            dfs(node)
            return sorted_nodes[::-1]

        sorted_nodes = topological_sort(self)
        self.grad = 1.0
        
        for node in sorted_nodes:
            node._backward()
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.value + other.value, (self, other))
        
        def backward():
            self.grad += output.grad
            other.grad += output.grad
        output._backward = backward
        return output

    def __radd__(self, other):
        return self + other

    def __repr__(self):
        return f'Value(value={self.value}, grad={self.grad})'

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.value * other.value, (self, other))
        
        def backward():
            self.grad += output.grad * other.value
            other.grad += output.grad * self.value
        output._backward = backward
        return output

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        output = Value(self.value ** other, (self,))
        
        def backward():
            self.grad += output.grad * other * (self.value ** (other - 1))
        output._backward = backward
        return output

    def tanh(self):
        output = Value(math.tanh(self.value), (self,))
        
        def backward():
            self.grad += (1 - output.value ** 2) * output.grad
        output._backward = backward
        return output

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other**-1

    def __neg__(self):
        return Value(-self.value, (self,))

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self + (-other)


class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        z = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return z.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        return [neuron(x) for neuron in self.neurons]

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(sz) - 1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# Example usage with synthetic data
xs = []
ys = []
for _ in range(100):
    x = [random.uniform(-1000, 1000), random.uniform(-1000, 1000)]
    xs.append(x)
    y = 0 if sum(x) % 2 else 1
    ys.append(y)

# Create the MLP
nn = MLP(2, [2, 2, 1])

for i in range(1000):
# Forward pass
    error = Value(0.0)
    for p in nn.parameters():
        p.grad = 0.0
    for x, y in zip(xs, ys):
        pred = nn(x)[0]
        loss = (pred - y) ** 2
        error += loss
    error /= len(xs)

    error.backward()
    print(error)
   # Backward pass
  


    # Update weights
    for p in nn.parameters():
        p.value -= 0.1 * p.grad
    
