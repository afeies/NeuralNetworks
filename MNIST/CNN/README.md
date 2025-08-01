convolutional neural network (cnn) - designed to precess data that has a grid-like structure, like images.
- fully connected networks connect every input pixel to every neuron
- cnns use convolutional layers that scan smaller filters over the input

core components
1. convolution layer (`nn.Conv2d`)
- learns small filters (e.g. 3x3) to detect local features in the image
2. activation function (`nn.Relu`)
- adds non-linearity so the network can learn complex patterns
3. pooling layer (`nn.MaxPool2d`)
- reduces the spatial size (downsampling) to make computation faster and improve robustness
4. full connected layers (`nn.Linear`)
- combine extracted features for the final classification

cnn advantages for image data
- preserve spatial structure: filters see pixels in context
- fewer parameters: more efficient and less prone to overfitting
- translation invariance: can recognize patterns anywhere in the image
- better generalization: especially on more complex datasets

architecture for this cnn
- input image: [1, 28, 28]
    - 1 channel, 28x28 pixels

- `nn.Conv2d(1, 16, kernel_size=3, padding=1)`
    - 16 filters of size 3x3x1
    - output: 16 feature map
    - padding keeps spatial size same

        - output: [16 x 28 x 28]

- `MaxPool2d(kernel_size=2)`
    - downsamples by 2x2 blocks

        - output: [16 x 14 x 14]

- `Conv2d(16, 32, kernel_size=3, padding=1)`
    - 32 filters of size 3x3x16
    - learns more complex features

        - output: [32 x 14 x 14]

- `MaxPool2d(kernel_size=2)`
    - downsamples by 2x2 blocks

        - output: [32 x 7 x 7]

- `Flatten`
    - converts 3D tensor to 1D tensor
    - 32 channels * 7 * 7 = 1568 features

        - output: [1568]

- `Linear(1568 -> 128)`
    - fully connected layer

        - output: [128]

- `Dropout(p=0.3)`
    - randomly sets 30% of elements to 0 during training

        - output: [128]

- `Linear(128 -> 10)`
    - final output: logits for 10 clothing categories

        - output: [10]