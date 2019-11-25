# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

set -e

lg=$1  # input language

# data path
MAIN_PATH=$PWD
DATA_PATH=$PWD/data
PYTHONPATH=$MAIN_PATH/xnlg
WIKI_PATH=$DATA_PATH/wiki
PREPROCESS=$MAIN_PATH/preprocess/preprocess.py
PROCESSED_PATH=$DATA_PATH/processed/XNLG
CODES_PATH=$DATA_PATH/codes_xnli_15
VOCAB_PATH=$DATA_PATH/vocab_xnli_15

# tools paths
TOOLS_PATH=$PWD/tools
TOKENIZE=$TOOLS_PATH/tokenize.sh
LOWER_REMOVE_ACCENT=$TOOLS_PATH/lowercase_and_remove_accent.py
FASTBPE=$TOOLS_PATH/fastBPE/fast

# Wiki data
WIKI_DUMP_NAME=${lg}wiki-latest-pages-articles.xml.bz2
WIKI_DUMP_LINK=https://dumps.wikimedia.org/${lg}wiki/latest/$WIKI_DUMP_NAME

# create Wiki paths
mkdir -p $WIKI_PATH/bz2
mkdir -p $WIKI_PATH/txt
mkdir -p $PROCESSED_PATH

# download Wikipedia dump
echo "Downloading $lg Wikipedia dump from $WIKI_DUMP_LINK ..."
wget -c $WIKI_DUMP_LINK -P $WIKI_PATH/bz2/
echo "Downloaded $WIKI_DUMP_NAME in $WIKI_PATH/bz2/$WIKI_DUMP_NAME"

# extract and tokenize Wiki data
cd $MAIN_PATH
echo "*** Cleaning and tokenizing $lg Wikipedia dump ... ***"
if [ ! -f $WIKI_PATH/txt/$lg.all ]; then
  python $TOOLS_PATH/wikiextractor/WikiExtractor.py $WIKI_PATH/bz2/$WIKI_DUMP_NAME --processes 8 -q -o - \
  | sed "/^\s*\$/d" \
  | grep -v "^<doc id=" \
  | grep -v "</doc>\$" \
  | $TOKENIZE $lg \
  | python $LOWER_REMOVE_ACCENT \
  > $WIKI_PATH/txt/$lg.all
fi
echo "*** Tokenized (+ lowercase + accent-removal) $lg Wikipedia dump to $WIKI_PATH/txt/train.${lg} ***"

# split into train / valid / test
echo "*** Split into train / valid / test ***"
split_data() {
    get_seeded_random() {
        seed="$1"; openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero 2>/dev/null
    };
    NLINES=`wc -l $1  | awk -F " " '{print $1}'`;
    NTRAIN=$((NLINES - 10000));
    NVAL=$((NTRAIN + 5000));
    shuf --random-source=<(get_seeded_random 42) $1 | head -$NTRAIN             > $2;
    shuf --random-source=<(get_seeded_random 42) $1 | head -$NVAL | tail -5000  > $3;
    shuf --random-source=<(get_seeded_random 42) $1 | tail -5000                > $4;
}
split_data $WIKI_PATH/txt/$lg.all $WIKI_PATH/txt/$lg.train $WIKI_PATH/txt/$lg.valid $WIKI_PATH/txt/$lg.test

# Get BPE codes and vocab
wget -c https://dl.fbaipublicfiles.com/XLM/codes_xnli_15 -P $DATA_PATH
wget -c https://dl.fbaipublicfiles.com/XLM/vocab_xnli_15 -P $DATA_PATH

# apply BPE codes and binarize the monolingual corpora
for split in train valid test; do
  $FASTBPE applybpe $PROCESSED_PATH/$lg.$split $WIKI_PATH/txt/$lg.$split $CODES_PATH
  python $PREPROCESS $VOCAB_PATH $PROCESSED_PATH/$lg.$split
done
