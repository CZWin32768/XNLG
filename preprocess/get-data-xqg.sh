set -e

# data path
MAIN_PATH=$PWD
DATA_PATH=$PWD/data
PYTHONPATH=$MAIN_PATH/xnlg
XQG_PATH=$DATA_PATH/xqg
PROCESSED_PATH=$DATA_PATH/processed/XNLG/eval/xqg
PREPROCESS=$MAIN_PATH/preprocess/preprocess.py
CODES_PATH=$DATA_PATH/codes_xnli_15
VOCAB_PATH=$DATA_PATH/vocab_xnli_15

# tools paths
TOOLS_PATH=$PWD/tools
TOKENIZE=$TOOLS_PATH/tokenize.sh
LOWER_REMOVE_ACCENT=$TOOLS_PATH/lowercase_and_remove_accent.py
FASTBPE=$TOOLS_PATH/fastBPE/fast

mkdir -p $XQG_PATH
mkdir -p $PROCESSED_PATH

gdown https://drive.google.com/uc\?id\=1kZJ0YyvW2vjCJv2vqEoemww2s5ECK7XJ -O $DATA_PATH/xqg.tar.gz

cd $DATA_PATH
tar zxf xqg.tar.gz

# Get BPE codes and vocab
wget -c https://dl.fbaipublicfiles.com/XLM/codes_xnli_15 -P $DATA_PATH
wget -c https://dl.fbaipublicfiles.com/XLM/vocab_xnli_15 -P $DATA_PATH

for split in train dev test; do
  for lang in en zh; do
    for seg in q a e; do
      $FASTBPE applybpe $PROCESSED_PATH/$split.$seg.$lang $XQG_PATH/$split.$seg.$lang.lc $CODES_PATH
      python $PREPROCESS $VOCAB_PATH $PROCESSED_PATH/$split.$seg.$lang
    done
  done
done

# Get Decoding vocabulary
gdown https://drive.google.com/uc\?id\=1jc7osl6XG7Cp3js5ui68idRlSYKzTwHT -O $DATA_PATH/xqg-decoding-vocab.tar.gz
tar zxf xqg-decoding-vocab.tar.gz