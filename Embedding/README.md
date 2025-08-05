# Understanding `nn.Embedding` in Pytorch
- demonstrate how word embedding work and how they are learned inside a neural network
- build a tiny text corpus, tokenize it, create a vocabulary, and train a simple context -> next work prediction
- start by implementing our own embedding layer from scratch, then compare it to `nn.Embedding`

## Word Embeddings
- learned vector representation of a word (e.g., [0.13, -1.92, 0.04, ...])
    - captures semantic meaning based on the word's usage in text
- these vectors are stored in a lookup table
    - a matrix of shape vocab_size x embedding_dim
- trained to help the model minize its loss (predict the next word)

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

## Letters and Shapes
### Embedding Layer
- `V`: vocab size
    - 17: total unique words from the corpus
- `D`: embedding dimension
    - 6: chosen manually
- `W`: embedding weight table
    - (17, 6)
- `ids`: input word IDs
    - (3,)
- `E`: embedding lookup result
    - (3, 6): output of embedding(ids)
- `h`: averages embedding vector
    - (8, 6): each row is the mean of 2 context vectors in a batch
### Batching + Context
- `B`: batch size
    - 8: chosen maunally
- `C`: context size
    - 2: we used 2-word windows
- `ctx`: context words IDs (batch)
    - (8, 2): 8 training examples, 2 words each
- `tgt`: target word IDs (batch)
    - (8,): one next-word target per example
- `logits`: model output
    - (8, 17): raw scores for each vocab word, one per example


## Terms
- batch - a group of training examples that the model processes together in a single forward/backward pass
- stochastic gradient descent (SGD)
    - update the model parameters after every single example (batch size = 1)
- batch gradient descent
    - update the model parameters once per epoch
    (batch size = size of entire dataset)
- mini-batch gradient descent 
    - update the model parameters after a batch of n examples
    - standard for training deep learning models