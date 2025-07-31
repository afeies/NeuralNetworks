torch.nn: https://docs.pytorch.org/docs/stable/nn.html

- containers - designed to hold and organize layers or other modules

    - `nn.Module` - base class for all neural network modules

        - holds parameters and other submodules

    - `nn.Sequential` - stacks layers in a sequential order

        - don't need to define custom forward() method

        ```
        model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        ```

        equivalent to

        ```
        def forward(x):
            x = linear1(x)
            x = relu(x)
            x = linear2(x)
            return x
        ```

- non-linear activations