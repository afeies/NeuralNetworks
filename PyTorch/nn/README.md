## Summary of Algorithm
1. Initialize Parameters
- define a neural network class inheriting from `nn.Module`
- use `nn.Linear` to automatically create and register parameters (weights and biases)
- layers used: `Linear(2, 4)` for hidden and `Linear(4, 1)` for output
- activations: `nn.ReLU()` and `nn.Sigmoid()`
2. Forward Pass
- call `model(X)` which internally runs `forward()`
- hidden layer: input passed to `self.hidden` and activated with ReLU
- output layer: activated with Sigmoid to get predicted probabilities
- shapes flow: (batch_size, 2) -> (batch_size, 4) -> (batch_size, 1)
3. Loss Calculation
- use `nn.BCELoss()` to compute binary cross-entropy between predictions and labels
- compares output of the Sigmoid layer to labels shaped as (batch_size, 1)
4. Backpropagation
- call `loss.backward()` to compute gradients of all parameters
- PyTorch traces the computation graph automatically and stores .grad in each layer's parameters
5. Parameter Update
- call `optimizer.step()` to apply updates using the Adam optimizer
- clears existing gradients with `optimizer.zero_grad()` to prevent accumulation
6. Repeat
- iterate over multiple epochs until loss converges

## New PyTorch Methods and Submodules Used
- `nn.Module`
    - base class for all PyTorch models
    - enables tracking of parameters and structured model building
- `super().__init__`
    - initializes the parent `nn.Module` class so the model can register submodules properly
- `nn.ReLU`
- `nn.Sigmoid`
- `model(x)`
    - triggers the model's `forward()` method automatically
    - inherited from `nn.Module` via `__call__`
- `nn.BCELoss()`
- `torch.optim.Adam(...)`
    - optimizer that adapts learning rates for each parameter
    - uses momentum and RMSProp ideas
- `model.parameters()`
    - returns all trainable parameters in the model (registered by `nn.Linear`)
    - passed into the optimizer
- `optimizer.zero_grad()`
    - resets gradients to zero before each backward pass to prevent accumulation
- `optimizer.step()`
    - updates the model's parameters using stored gradients and the optimizer algorithm
- `torch.no_grad()`
    - disables gradient tracking
    - used during evaluation to speed things up and reduce memory usage