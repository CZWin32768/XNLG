set -e

# data path
MAIN_PATH=$PWD
DATA_PATH=$PWD/data
PYTHONPATH=$MAIN_PATH/xnlg
XGIGA_PATH=$DATA_PATH/xgiga
PROCESSED_PATH=$DATA_PATH/processed/XNLG/eval/xgiga
PREPROCESS=$MAIN_PATH/preprocess/preprocess.py
CODES_PATH=$DATA_PATH/codes_xnli_15
VOCAB_PATH=$DATA_PATH/vocab_xnli_15

# tools paths
TOOLS_PATH=$PWD/tools
TOKENIZE=$TOOLS_PATH/tokenize.sh
LOWER_REMOVE_ACCENT=$TOOLS_PATH/lowercase_and_remove_accent.py
FASTBPE=$TOOLS_PATH/fastBPE/fast

mkdir -p $XGIGA_PATH
mkdir -p $PROCESSED_PATH

gdown https://drive.google.com/uc\?id\=1DP6pFPBTkGR1sopcZHY2V-3P8MNYpdsL -O $DATA_PATH/xgiga.tar.gz

cd $DATA_PATH
tar zxf xgiga.tar.gz

# Get BPE codes and vocab
wget -c https://dl.fbaipublicfiles.com/XLM/codes_xnli_15 -P $DATA_PATH
wget -c https://dl.fbaipublicfiles.com/XLM/vocab_xnli_15 -P $DATA_PATH

for split in train dev test; do
  for lang in en fr zh; do
    for seg in x y; do
      $FASTBPE applybpe $PROCESSED_PATH/$split.$seg.$lang $XGIGA_PATH/$split.$seg.$lang $CODES_PATH
      python $PREPROCESS $VOCAB_PATH $PROCESSED_PATH/$split.$seg.$lang
    done
  done
done