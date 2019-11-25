# Tools

In `./tools/`, you will need to install the following tools:

## Tokenizers

[Moses](https://github.com/moses-smt/mosesdecoder/tree/master/scripts/tokenizer) tokenizer:
```
git clone https://github.com/moses-smt/mosesdecoder
```

Chinese Stanford segmenter:
```
wget https://nlp.stanford.edu/software/stanford-segmenter-2018-10-16.zip
unzip stanford-segmenter-2018-10-16.zip
```

## fastBPE

```
git clone https://github.com/glample/fastBPE
cd fastBPE
g++ -std=c++11 -pthread -O3 fast.cc -o fast
```
