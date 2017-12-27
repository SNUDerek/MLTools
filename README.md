# MLTools

module for custom functions used in keras and ML projects

## installation

use `pip install .` from root dir

use `pip install -e .` for symlink (updates immediately accessible)

## tools:

### picklable class-based tokenization and indexing

use the Tokenizer and Indexer in sequence for automatic indexing:

```
from mltools.preprocessing import Tokenizer, Indexer, Pipeline

tokenizer = Tokenizer(max_vocab=100, min_count=1, lower=True, regex=True)
indicizer = Indexer(max_len=10, pad='post', truncate='post',
                    reverse=False, unk_name='UNK', pad_name='PAD')

pipeline = Pipeline([
    ('tokenize', tokenizer),
    ('indicize', indicizer)
])

pipeline.fit(texts)

vects = pipeline.transform(text[:split_idx])
```

this pipeline can be pickled with `sklearn.externals.joblib` `dump()` and `load()`

### mltools.preprocessing

functions for getting (truncated) vocabulary, integer-indexing sequences for keras

decode_sequence, get_vocab, index_sents, onehot_vectorize, dataGenerator

### mltools.embeddings

functions for training `gensim.word2vec` models

### mltools.similarity

class for using cosine similarity of sentence vectors for retrieval-based dialogs
