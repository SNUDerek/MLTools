import numpy as np
import re
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
from sklearn.base import BaseEstimator, TransformerMixin


class Tokenizer(BaseEstimator, TransformerMixin):
    """
    Space-tokenize a list of (English) sentences.

    Arguments
    ---------
    lower : bool
        Convert all input strings to lowercase with .lower()
    max_vocab : int
        Maximum vocabulary by size
    min_count : int
        Minimum count for vocabulary (alternate size contraint)
    stopwords : list[str]
        Words to remove
    regex : bool
        Use regex to remove all symbols but alphanumerics
    char : bool
        Tokenize by chars

    Attributes
    ----------
    vcounts : dict
        Vocabulary dictionary in form word : count
    """

    def __init__(self, lower=True, max_vocab=10000, min_count=1,
                 stopwords=[], regex=False, char=False, vcounts={},
                 unk_name='UNK'):

        super().__init__()

        self.lower = lower
        self.max_vocab = max_vocab
        self.min_count = min_count
        self.stopwords = stopwords
        self.regex = regex
        self.char = char
        self.vcounts = vcounts
        self.unk_name = unk_name

    def __setstate__(self, state):
        self.__init__(
            state['lower'],
            state['max_vocab'],
            state['min_count'],
            state['stopwords'],
            state['regex'],
            state['char'],
            state['vcounts'],
            state['unk_name'])

    def __getstate__(self):
        state = {
            'lower': self.lower,
            'max_vocab': self.max_vocab,
            'min_count': self.min_count,
            'stopwords': self.stopwords,
            'regex': self.regex,
            'char': self.char,
            'vcounts': self.vcounts,
            'unk_name': self.unk_name
        }

        return state

    # get vocabulary and construct count dictionary
    def _get_vocab(self, sent_toks):

        # get vocab list, cast all as strings
        vocab = [str(word) for sent in sent_toks for word in sent]
        vocab_counts = [t for t in Counter(vocab).most_common()]
        vocab_counts = [t for t in vocab_counts if t[0] not in self.stopwords and t[1] >= self.min_count]
        self.vcounts = dict(vocab_counts)

        return

    # preprocess and space-tokenize
    def _tokenize(self, sentences, pretokenize=False):

        results = []

        for sent in sentences:
            sent = str(sent).strip()
            if self.lower:
                sent = sent.lower()
            if self.regex:
                sent = re.sub(r'[^0-9A-Za-z\s]', '', sent)
                sent = sent.replace('  ', ' ')
            if self.char:
                sent_toks = list(sent)
            else:
                sent_toks = sent.split()

            if pretokenize == False:
                sent_toks = [w if w in self.vcounts.keys() else self.unk_name for w in sent_toks]

            if len(sent_toks) > 0:
                results.append(sent_toks)
            else:
                results.append([self.unk_name])

        return results

    # fit function
    def fit(self, sentences, y=None):

        tokens = self._tokenize(sentences, pretokenize=True)
        self._get_vocab(tokens)

        return self

    # transform
    def transform(self, sentences, y=None):

        # index_sents
        tokens = self._tokenize(sentences)

        return tokens

    def fit_transform(self, sentences, y=None):

        self.fit(sentences)
        tokens = self._tokenize(sentences)

        return tokens

    def inverse_transform(self, sent_toks, y=None):

        return sent_toks


class Indexer(BaseEstimator, TransformerMixin):
    """
    Integer-index a list of tokenized sentences.

    Arguments
    ---------
    max_len : int
        Maximum sequence length. If None, set = max length in data.
    pad : str
        How to pad sequences ('pre' or 'post').
    truncate : str
        How to truncate sequences ('pre' or 'post').
    reverse : bool
        Whether to reverse input sequences (see Sutskever et al.)
    unk_name : str
        Symbol for unknown word (OOV word).
    pad_name : str
        Symbol for pad_value.

    Attributes
    ----------
    idx2word : dict
        Dictionary mapping integer indices to words.
    word2idx : dict
        Dictionary mapping words to integer indices.
    """

    def __init__(self, max_len=None, pad='post', truncate='post',
                 reverse=False, unk_name='UNK', pad_name='PAD',
                 word2idx=None, idx2word=None):

        super().__init__()
        self.max_len = max_len
        self.pad = pad
        self.truncate = truncate
        self.reverse = reverse
        self.unk_name = unk_name
        self.pad_name = pad_name
        self.word2idx = word2idx
        self.idx2word = idx2word

    def __setstate__(self, state):
        self.__init__(
            state['max_len'],
            state['pad'],
            state['truncate'],
            state['reverse'],
            state['unk_name'],
            state['pad_name'],
            state['word2idx'],
            state['idx2word'],
        )

    def __getstate__(self):
        state = {
            'max_len': self.max_len,
            'pad': self.pad,
            'truncate': self.truncate,
            'reverse': self.reverse,
            'unk_name': self.unk_name,
            'pad_name': self.pad_name,
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
        }

        return state

    # pad integer-indexed sequences
    def _zero_pad(self, idx_tokens):

        padded_data = pad_sequences(idx_tokens, maxlen=self.max_len, padding=self.pad, truncating=self.truncate)

        return padded_data

    # get vocabulary and construct dictionaries
    def _get_vocab(self, sent_toks):

        # get vocab list, cast all as strings
        vocab = [str(word) for sent in sent_toks for word in sent]
        vocab_counts = [t for t in Counter(vocab).most_common()]  # get counts of each word, sort
        vocab_counts = [t for t in vocab_counts if t[0] != self.unk_name]
        sorted_vocab = [t[0] for t in vocab_counts]

        # reserve for PAD and UNK
        sorted_vocab = sorted_vocab[:-2]
        self.word2idx = {k: v + 1 for v, k in enumerate(sorted_vocab)}
        self.word2idx[self.unk_name] = len(sorted_vocab) + 1
        self.word2idx[self.pad_name] = 0
        self.idx2word = {v: k for k, v in self.word2idx.items()}

        return

    # integer-index sentences with vocabulary
    def _index_sents(self, sent_tokens):
        vectors = []
        for sent in sent_tokens:
            sent_vect = []
            if self.reverse:
                sent = sent[::-1]
            for word in sent:
                if word in self.word2idx.keys():
                    sent_vect.append(self.word2idx[word])
                else:  # out of max_vocab range or OOV
                    sent_vect.append(self.word2idx[self.unk_name])
            vectors.append(np.asarray(sent_vect))
        vectors = np.asarray(vectors)
        return vectors

    # decode integer-indexed sentences with vocabulary
    def _decode_sents(self, idx_tokens):
        sentences = []
        for sent in idx_tokens:
            sent_toks = []
            if self.reverse:
                sent = sent[::-1]
            for idx in sent:
                if idx in self.idx2word.keys():
                    sent_toks.append(self.idx2word[idx])
                else:  # out of max_vocab range or OOV
                    sent_toks.append(self.idx2word[self.unk_name])
            sentences.append(sent_toks)
        return sentences

    # fit model (vocab, max_len)
    def fit(self, sent_toks, y=None):

        # set max_len if None
        sent_lens = []

        if self.max_len is None:
            self.max_len = np.max(sent_lens)

        # fit vocab
        self._get_vocab(sent_toks)

        return self

    # transform
    def transform(self, sent_toks, y=None):

        # index_sents
        idx_toks = self._index_sents(sent_toks)
        idx_toks = self._zero_pad(idx_toks)

        return idx_toks

    def fit_transform(self, sent_toks, y=None):

        self.fit(sent_toks)
        idx_toks = self.transform(sent_toks)

        return idx_toks

    def inverse_transform(self, idx_toks, y=None):

        sent_toks = self._decode_sents(idx_toks)

        return sent_toks
