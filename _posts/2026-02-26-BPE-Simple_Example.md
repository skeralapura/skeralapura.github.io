---
layout: post
title: Byte-Pair Encoding (BPE) Example
image: "/posts/bpe.png"
tags: [Python, LLMs, BPE, Tokenizer]
---

# Example of building a Byte-Pair Encoding (BPE) algorithm

This project showcases a simple example of how a BPE algorithm works. Tokenization is one of the main building blocks of LLM. Tokenization is a process of converting raw text into vectors that a neural network can understand.

<br>
**Typical LLM Pipeline:** 
Raw data (from internet and other sources) --> Data cleansing --> Tokenizer --> Transformers --> Output probabilities --> Human-Readable Output 

Types of tokenization:
* Word-level
* Character-level
* Subword-level

BPE is a type of subword-level tokenization. The BPE algorithm builds a vocabulary iteratively using the following process:

*   Start with individual characters (each character is a token)
    - Extract all unique characters from corpus to create initial vocab
*   Count all adjacent pairs of tokens in a text corpus
*   Merge most frequent pair into a new token
    - Iterate through 'num_merges' (parameter) times
    - Repeat till desired vocab size

In practice, instead of implementing the algorithm from scratch, we can use a pretrained tokenizer (eg: AutoTokenizer), which was already trained on a large text corpus to build its vocabulary, such as the data used to train GPT-2 or another example would be TikToken library used by OpenAI models.
<br>
___

We start of with a sample corpus.

```python
# Example corpus
corpus = "Too wet to go out, And too cold to play ball. So we sat in the house. We did nothing at all which is too much joy!"
```

___

## Define function for finding most frequent pair and extract frequencies
- Loop through each word (represented as a tuple of chars)
- Extract adjacent pairs of tokens
- Add the word’s frequency to each pair’s count

```python
def get_freq(vocab):
    """Count frequency of all adjacent pairs in the vocabulary."""
    
    pairs = Counter()
    # Iterate over character pairs in the word (represented as tuple of chars)
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            # Extract all adjacent pairs of tokens and add word's freq to each pair's count
            pairs[(word[i], word[i+1])] += freq
    return pairs
```

___

## Define function for merging most frequent pair into a new token in the vocab
- Iterate through frequency and pair combination
- Merge the pair into new token
- Return the new vocab

```python
def merge_pair(vocab, pair, new_token):
    """Merge the most frequent pair into a new token in the vocabulary."""

    new_vocab = {}
    for word, freq in vocab.items():
        # Replace the pair within the word (represented as a list/tuple of characters)
        new_word = []
        i = 0
        while i < len(word) - 1:
            if (word[i], word[i+1]) == pair:
                new_word.append(new_token)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        if i < len(word):
            new_word.append(word[i])
        new_vocab[tuple(new_word)] = freq # Use tuple as key for dict
    return new_vocab
```

___

## Define a main function for BPE training
- Start with individual characters (each character is a token). 
- Extract all unique characters from corpus to create initial vocab. 
- Based on how many merges are needed to reach the desired vocab size, call function to find best pair. Iterate through the num_merges
- Find best (frequent) pair and call function to merge best pair into a new token.

```python
import re
from collections import Counter

def bpe_training(corpus, num_merges):
    """Trains a BPE tokenizer given a text corpus and number of merges."""

    # Start with a character-level vocabulary
    # vocabulary of words represented as character sequences
    words = re.findall(r'\w+|\d+|[^\s\w\d]', corpus.lower()) # Basic word split
    vocab = Counter(tuple(word) for word in words) # Initial character vocab
    merges = {}

    for i in range(num_merges):
        # Finding the most frequent pair
        pairs = get_freq(vocab)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        new_token = "".join(best_pair) # New token is the merged string
        vocab = merge_pair(vocab, best_pair, new_token)
        merges[best_pair] = new_token
        print(f"Merge {i+1}: {best_pair} -> {new_token}")

    return merges, vocab
```

___

# Example tests:

```python
num_merges = 5
merges, vocab = bpe_training(corpus, num_merges)
print("\nFinal Vocabulary:", vocab)
```

### Output:

```md
Merge 1: ('t', 'o') -> to
Merge 2: ('to', 'o') -> too
Merge 3: ('w', 'e') -> we
Merge 4: ('o', 'u') -> ou
Merge 5: ('a', 'l') -> al

Final Vocabulary: {('too',): 3, ('we', 't'): 1, ('to',): 2, ('g', 'o'): 1, ('ou', 't'): 1, (',',): 1, ('a', 'n', 'd'): 1, ('c', 'o', 'l', 'd'): 1, ('p', 'l', 'a', 'y'): 1, ('b', 'al', 'l'): 1, ('.',): 2, ('s', 'o'): 1, ('we',): 2, ('s', 'a', 't'): 1, ('i', 'n'): 1, ('t', 'h', 'e'): 1, ('h', 'ou', 's', 'e'): 1, ('d', 'i', 'd'): 1, ('n', 'o', 't', 'h', 'i', 'n', 'g'): 1, ('a', 't'): 1, ('al', 'l'): 1, ('w', 'h', 'i', 'c', 'h'): 1, ('i', 's'): 1, ('m', 'u', 'c', 'h'): 1, ('j', 'o', 'y'): 1, ('!',): 1}
```
___
