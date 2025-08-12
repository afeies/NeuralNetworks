# Summary of Algorithm
1. Initialize Parameters
- random W1, W2 and zeros for b1, b2
2. Forward Pass
- hidden layer pre-activation: Z1 = X @ W1 + b1
- apply ReLU activation: A1 = ReLu(Z1)
- output layer pre-activation: Z2 = A1 @ W2 + b2
- apple sigmoid activation: A2 = sigma(Z2) to get predicted probabilities
3. Loss Calculation
- use BCE to compare A2 (predicted probabilities) with y (true labels)
4. Backpropagation
- compute gradients of loss with respect to each parameter: dW1, db1, dW2, db2
5. Parameter Update
- update weights and biases using gradient descent:
    - parameter = parameter - lr * gradient
6. Repeat
- iterate over multiple epochs until loss converges

## Parameters
- weighted sum - XW: combine inputs using learned weights
- add bias - +b: shift outputs independently of inputs
- apply activation - f(Z): introduce non-linearity

## Activation functions
- hidden layer - ReLU: fast, avoids vanishing gradient, good feature transformation
- output layer - Sigmoid: outputs probability for binary classification

## Binary Cross-Entropy (BCE)
- loss function used for binary classification problems where there are only two possible classes (e.g, 0 or 1)
- measures the difference between:
    - the true label y (0 or 1)
    - the predicted probability $\hat{y}$ (between 0 and 1, probability output of the sigmoid)
- for one sample:
    
    $\text{BCE Loss} = - \left[ y \cdot \log(\hat{y}) + (1 - y) \cdot \log(1 - \hat{y}) \right]$
- for m samples:
    - take negative mean of sum of individual BCE losses
        - mean: we want a single loss value
        - negate: logarithms of probabilities are always negative
    
    $\text{BCE Loss} = -1/m(sum of BCE losses)$

## Backpropagation
- compute the gradients of loss with respect to eahc parameter
- this tells us how much each weight and bias contributed to the error
    - gives us dW1, db1, dW2, db2



