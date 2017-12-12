# MLTools

module for custom functions used in keras and ML projects

## picklable class-based tokenization and indexing

use the Tokenizer and Indexer in sequence for automatic indexing:

```
from mltools import Tokenizer, Indexer
from sklearn.pipeline import pipeline

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

## mltools.datasets

functions for getting the (truncated) vocabulary

decode_sequence, get_vocab, index_sents, onehot_vectorize, dataGenerator
