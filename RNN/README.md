- token - the smallest unit of text that a model processes as a single element
    - can be whatever segmentation you choose
    - in this RNN, tokens are individual characters from our training text

    1. text -> tokens
    - "R" -> ID 21, "O" -> ID 14, etc.
    2. tokens -> embeddings
    - each ID is mapped to an embedding vector (size = embedding_dim)
    3. embeddings -> RNN
    - the RNN processes them one by one, updating the hidden state
    4. RNN output -> probabilites over tokens
    - softmax gives probabilities for each token in the vocabulary as the next character

https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html
- `nn.GRU`
    - a GRU keeps a hidden state h_t and updates it with two gates:
    
    GRU operations

    1. reset gate: $r_t = \sigma(W_r * [h_{t - 1}, x_t])$
    - determines how much of the previous hidden state h_(t - 1) should be forgotten
        - increases when the current char should be interpreted through what just came (e.g. prefixes)
        - lowers when a new topic appears (stop consulting old context)

    2. update gate: $z_t = \sigma(W_z * [h_{t - 1}, x_t])$
    - determines how much of the new information x_t should be used to update the hidden state
        - increases when the broader state should persist across characters
        - lowers when the new input should overwrie the old memory (e.g. negation)


- `seq_len` = 128
    - input: 128 characters from the text
    - output: the next 128 characters (each shifted by 1 position)

- `batch_size` = 128
    - batch size: the number of training examples process together on one forward and backward pass
    - 128 sequences

        - Input tensor to the model for one batch has shape:
            - [B, T] = [128, 128]

- `nn.Embedding(vocab_size, embedding_dim)`
    - character embedding vector - numeric representation of a character
        - `vocab_size`: number of unique characters
            - depends on the dataset
        - `embedding_dim`: size of the vector for each character
            - the larger, the more detailed representation of each character
    
    V (vocab size) = 3: ids {0, 1, 2}
    E (embedding_dim) = 4


- hidden state - the model's memory
    - `hidden_dim`: how many features (neurons) the hdden state has at each time step
        - how much information it can store in its memory

- `grad_clip`
    - gradient clipping - a technique to limit the size of gradients during backpropagation
        - if they get too large, they cause exploding gradients

- `temperature` - controls how random or confident the model's predictions are when generating text
    - lower t < 1: sharper distribution (model more confident)
    - higher t > 1: flatter distribution (model less confident)

- `top_k`
    - softmax gives probabilities for every character in the vocabulary and many of these have very low probability

- `top_p`
    - selectd from the smallest set of tokens whose cumulative probability is at least p
    - adaptive compared to top k
        - model is confident: few tokens are considered
        - model is uncertain: moke tokens are considered