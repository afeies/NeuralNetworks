## Summary of Algorithm
1. Initialize Parameters
- random W1, W2 and zeros for b1, b2
- requires_grad=True
2. Forward Pass
- hidden layer pre-activation: Z1 = X @ W1 + b1
    - broadcasting
- apply ReLU activation: A1 = torch.relu(Z1)
- output layer pre-activation: Z2 = A1 @ W2 + b2
- apply sigmoid activation: A2 = torch.sigmoid(Z2) to get predicted probabilities
3. Loss Calculation
- BCE implements with torch.log() and torch.clamp()
- compare A2 (predicted probabilities) with y (true labels)
4. Backpropagation
- call loss.backward() to let PyTorch autograd compute all gradients for W1, b1, W2, b2 based on the computation graph
5. Parameter Update
- within torch.no_grad(), update weights and biases using gradient descent:
    - parameter -= lr * parameter.grad
- reset gradients to zero with .grad.zero()
6. Repeat
- iterate over multiple epochs until loss converges

## Broadcasting
1. If the tensors have different numbers of dimensions, prepend 1s to the smaller one
2. Dimensions are compatible if:
- they are equal, or
- one of them is 1
3. The smaller tensor is virtually replicated along the 1-sized dimensions to match the larger shape

## New PyTorch APIs Used
### tensor creation and shaping
- torch.tensor(...) - create tensors
- torch.randn(shape) - random normal init for weights
- torch.zeros(shape) - zero init for biases
- torch.cat([t1, t2], dim=...) - concatenate x0/x1 and label vectors
- Tensor.view(new_shape) - reshape labels to (N, 1)

### randomness / reproducibility
- torch.manual_seed(seed) - reproducible data/weights
- torch.randint(low, high, size) - random sample indices for inspection

### activations and math
- torch.relu(x) - hidden activiation
- torch.sigmoid(x) - output activation (probabilities)
- torch.clamp(x, min, max) - stabilize BCE (avoid log(0))
- torch.log(x) - log term inside BCE

### autograd (automatic differentiation)
- requires_grad=True - track weights/biases
- loss.backward() - compute gradients via the graph
- Tensor.grad - access accumulated gradients
- with torch.no_grad() - disable tracking during manual updates
- Tensor.grad.zero_() - clear accumulated gradients after each step

### visualization helpers (for decision boundary)
- torch.linspace(start, end, steps) - grid coordinates
- torch.meshgrid(xs, ys, indexing='ij') - 2D grid for model eval on the plane