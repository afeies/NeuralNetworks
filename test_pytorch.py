import torch
import numpy as np

data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
print(x_data)

np_array = np.array(data)
print()
print(np_array)
x_np = torch.from_numpy(np_array)
print(x_np)

x_ones = torch.ones_like(x_data)
print(x_ones)

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(x_rand)

print()

shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(rand_tensor)
print(ones_tensor)
print(zeros_tensor)

print()

tensor = torch.rand(3,4)
print(tensor)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

print()

print(torch.is_tensor(tensor))
print(torch.is_storage(tensor))
print(torch.is_floating_point(tensor))

print(torch.get_default_dtype())

print(torch.numel(tensor))

x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)

x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])
print(f"x + y: {x + y}")
print(f"x * y: {x * y}")

# Dot product: 1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32
print(f"Dot product: {torch.dot(x, y)}")
print(f"Mean of x: {x.mean()}")
print(f"Sum of y: {y.sum()}")

m = torch.arange(12).reshape(3, 4)
print(f"Matrix m:\n{m}")
print(f"m[1]: {m[1]}")
print(f"m[:, 2]: {m[:, 2]}")

# Broadcasting allows operations on tensors of different shapes
# by automatically expanding one or both tensors

# When two tensor have shapes that don't match,
# PyTorch stretches the smaller one along the mismatched dimensions
# as long as the sizes are compatible

# Broadcasting rules
# 1. If the tensors have different numbers of dimensions, prepend 1s to the smaller one
# 2. Dimensions are compatible if:
#   - ther are equal, or
#   - one of them is 1
# 3. The smaller tensor is virtually replicated along the 1-sized dimensions to match the larger shape
v = torch.tensor([1.0, 2.0, 3.0])
M = torch.ones(3, 3)
print(f"Matrix M:\n{M}")
print(f"M + v (broadcasting):\n{M + v}")

x = torch.tensor([[1], [2], [3]])  # shape: (3, 1)
y = torch.tensor([10, 20, 30])     # shape: (3,) → becomes (1, 3)
print(x + y)