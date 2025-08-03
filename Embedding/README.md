# Understanding `nn.Embedding` in Pytorch
- demonstrate how word embedding work and how they are learned inside a neural network
- build a tiny text corpus, tokenize it, create a vocabulary, and train a simple context -> next work prediction
- start by implementing our own embedding layer from scratch, then compare it to `nn.Embedding`

## How `nn.Embedding` Works Under the Hood

### The Lookup Table
`emb = nn.Embedding(num_embedding=V, embedding_dim=D)`
- allocates a weight matrix of shape (V, D):
    - V = vocab size (# of unique tokens)
    - D = embedding dimension (length of each vector)
- initializes it with random values
- this weight matrix is the lookup table

### Forward Pass = Row Indexing
`vecs = emb(ids)`
- indexes into the weight matrix to retrieve the rows corresponding to the token IDs
- equivalent to `vecs = emb.weight[ids]`

### Backpropagation
- the embedding layer has no standalone training logic
- gradients flow into it from the loss via the rest of the model
- only the rows for the tokens used in the batch get non-zero gradients
- optimizer updates those rows so they move in directions that reduce the loss