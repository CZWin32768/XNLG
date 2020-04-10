# XNLG

Code and dataset for the paper [Cross-Lingual Natural Language Generation via Pre-Training](https://arxiv.org/pdf/1909.10481.pdf) (AAAI-20).

With the pre-trained XNLG, the supervision signals of NLG can be transferred to other languages. For example, finetuning XNLG with English Abstractive Summarization (AS) data and directly performing French AS or even Chinese-French AS.

This repo is based on [XLM](https://github.com/facebookresearch/XLM).

## Dependencies

- numpy
- [nlgeval](https://github.com/Maluuba/nlg-eval) (for calculating BLEU scores)
- pytorch 1.1.0
- fastBPE (generate and apply BPE cpdes)
- Moses (for tokenization)
- apex (for fp16 training)
- tqdm
- gdown (for downloading from Google Drive)
- [pythainlp](https://github.com/PyThaiNLP/pythainlp) 2.0.6

You can install some of the required tools through `bash ./preprocess/install-tools.sh`

## Stage #1: Encoding Pre-Training

### Pre-Trained Models for Stage #1

You can directly use pre-trained [XLM](https://github.com/facebookresearch/XLM) as the pre-trained model for stage #1. 

In the paper, we used the pre-trained model provided by XLM.

| Languages | Layers | Model | BPE codes | Vocabulary |
|-----------|--------|-------|-----------|------------|
|XNLI-15    |12      |[Model](https://dl.fbaipublicfiles.com/XLM/mlm_tlm_xnli15_1024.pth)|[BPE codes](https://dl.fbaipublicfiles.com/XLM/codes_xnli_15)|[Vocabulary](https://dl.fbaipublicfiles.com/XLM/vocab_xnli_15)|

### Training New Models for Stage #1 

#### Preparing Training Data

**Monolingual data** 

In the paper, we use the Wikipedias as the monolingual training data. You can get monolingual training data by `get-data-wiki.sh [lang]`.

E.g., `bash ./preprocess/get-data-wiki.sh en`.

**Parallel data**

In the paper, we use MultiUN as the parallel corpus for en-zh and en-fr.
You can get monolingual training data by `get-data-wiki.sh [lang1-lang2]`.

E.g., `bash ./preprocess/get-data-para.sh en-fr`.

#### Training

```
export NGPU=4; python -m torch.distributed.launch --nproc_per_node=$NGPU python xnlg-train.py
--exp_name stage1_en-zh-fr                 # experiment name
--dump_path ./dump                         # where to store the experiment
--data_path ./data/processed/XNLG          # data location
--lgs 'en-fr-zh'                           # considered languages
--mlm_steps 'en,zh,fr,en-fr,en-zh'         # MLM/XMLM objective
--emb_dim 1024                             # embeddings / model dimension (2048 is big, reduce if only 16Gb of GPU memory)
--n_layers 12                              # number of layers
--n_heads 16                               # number of heads
--dropout 0.1                              # dropout
--attention_dropout 0.1                    # attention dropout
--gelu_activation true                     # GELU instead of ReLU
--batch_size 32                            # sequences per batch
--bptt 256                                 # sequences length  (streams of 256 tokens for MLM)
--optimizer adam,lr=0.0001                 # optimizer (training is quite sensitive to this parameter)
--epoch_size 300000                        # number of sentences per epoch
--max_epoch 100000                         # max number of epochs (~infinite here)
--validation_metrics _valid_mlm_ppl        # validation metric (when to save the best model)
--stopping_criterion _valid_mlm_ppl,25     # stopping criterion (if criterion does not improve 25 times)
--fp16 true  
```

## Stage #2: Decoding Pre-Training

### Pre-Trained Models for Stage #2

We provide the pre-trained XNLG used in the paper:

| Languages | Layers | Validation | Model | BPE codes | Vocabulary |
|-----------|--------|------------|-------|-----------|------------|
|en,zh      |10-6    | en-zh      |[Model](https://drive.google.com/uc?export=download&confirm=_Mzi&id=1uFaHxXGufKWvh27bp9s3dOi0r19_Eg7t)|[BPE codes](https://drive.google.com/uc?authuser=0&id=1nEIWJQlLD26vPt_22pDnEZSJ4mIjssps&export=download)|[Vocabulary](https://drive.google.com/uc?id=1lIERR1ejW7_LV2rL9fFNkEhzmr7jooRx&export=download)|
|en,fr,zh   |10-6    | en-fr      |[Model](https://drive.google.com/uc?export=download&confirm=hhXm&id=1P4iqXi7lGBdLFFPXZIj2SUQTW_1ZLod8)|[BPE codes](https://drive.google.com/uc?authuser=0&id=1nEIWJQlLD26vPt_22pDnEZSJ4mIjssps&export=download)|[Vocabulary](https://drive.google.com/uc?id=1lIERR1ejW7_LV2rL9fFNkEhzmr7jooRx&export=download)|
|en,fr,zh   |10-6    | en-zh      |[Model](https://drive.google.com/uc?export=download&confirm=1bsX&id=1LKXlEHsCOpt1NOEpjXy1XbXqVk3gZGZQ)|[BPE codes](https://drive.google.com/uc?authuser=0&id=1nEIWJQlLD26vPt_22pDnEZSJ4mIjssps&export=download)|[Vocabulary](https://drive.google.com/uc?id=1lIERR1ejW7_LV2rL9fFNkEhzmr7jooRx&export=download)|

### Training New Models for Stage #2

At Stage #2, the model is trained with the same data with #1. 

Notes:

- To load the pre-trained model at Stage #1, use`--reload_model`. `--reload_model [NAME1].pth,[NAME2].pth` means initializing encoder and decoder with `[NAME1]` and `[NAME2]`, respectively.
- In the paper, we used a 10-layer encoder and a 6-layer decoder, so you can use `--n_layers` to set the number of decoder layers and use `--n_enc_layers` to set the number of encoder layers. (When a 10-layer Transformer is loaded from a 12-layer Transformer, it will use the parameters of the first 10 layers of the 12-layer one.)
- During Stage #2, the encoder parameters are frozen, and we only update the decoder parameters. You can use `--train_model_names decoder`.

```
export NGPU=4; python -m torch.distributed.launch --nproc_per_node=4 xnlg-train.py
--exp_name stage2_en-zh-fr
--dump_path ./dump
--data_path ./data/processed/XNLG
--lgs 'ar-bg-de-el-en-es-fr-hi-ru-sw-th-tr-ur-vi-zh'
--mt_steps 'en-zh,zh-en,en-fr,fr-en'
--ae_steps 'en,zh,fr'
--reload_model /path/to/mlm_tlm_xnli15_1024.pth,/path/to/mlm_tlm_xnli15_1024.pth
--emb_dim 1024
--n_layers 6
--n_heads 8
--dropout 0.1
--attention_dropout 0.1
--gelu_activation True
--batch_size 16
--bptt 256
--optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001
--epoch_size 10000
--max_vocab 95000
--encoder_only False
--train_model_names decoder
--stopping_criterion 'valid_en-zh_mt_bleu,25'
--validation_metrics 'valid_en-zh_mt_bleu,valid_en-fr_mt_bleu'
--eval_bleu True
--word_shuffle 3
--word_dropout 0.1
--word_blank 0.1
--lambda_ae 0.5
--n_enc_layers 10
```

## Fine-Tuning for Downstream NLG Tasks

### Question Generation (QG)

#### Preparing Training Data

We use SQuAD 1.1 as the English QG dataset and WebQA as the Chinese QG dataset. You can get our processed the dataset by:
```
bash ./preprocess/get-data-xqg.sh
```

or directly download at [here](https://drive.google.com/open?id=1kZJ0YyvW2vjCJv2vqEoemww2s5ECK7XJ).

When decoding for QG, we use a decoding vocabulary, which can be downloaded at [here](https://drive.google.com/file/d/1jc7osl6XG7Cp3js5ui68idRlSYKzTwHT).

#### Training for Zero-Shot QG

```
python xnlg-ft.py
--exp_name xqg
--dump_path ./dump
--model_path /path/to/pre-trained/XNLG/model
--data_path ./data/processed/XNLG
--transfer_tasks XQG
--optimizer adam,lr=0.000005
--batch_size 16
--n_epochs 200
--epoch_size 4000
--max_len_q 256
--max_len_a 20
--max_len_e 230
--max_vocab 95000
--train_layers 1,10                    # Use `1,10` or `encoder` for zero-shot QG
--vocab_path ./data/xqg-decoding-vocab
--decode_with_vocab True               # When evaluating on Chinese, set True.
--decode_vocab_sizes 95000,95000
--n_enc_layers 10
--n_dec_layers 6
--beam_size 3
--ds_name xqg
--train_directions en-en               
--eval_directions en-en,zh-zh
```

#### Training for Supervised QG

For supervised QG, `--train_layers` should be set as `all`. For supervised Chinese QG, just set `--train_directions` and `--eval_directions` as `zh-zh`.

#### Generating Questions

With a fine-tuned model, you can generate questions in a specific language by controlling the generation direction:

```bash
python qg.py
--vocab_path /path/to/vocab/folder
--data_path ./data/processed/XNLG
--model_dir /path/to/exp
--job_name [exp-index]                      # a hash code like `a23h1yv1`
--direction en-zh                           # en-en, en-zh, zh-en or zh-zh
```

#### Evaluating

Calculate BLEU and METEOR scores:

```
python calc_nlg_scores.py
-i /path/to/generated/questions
--lang zh
--dataset_dir /path/to/eval-dataset
```

**NOTE:** The Chinese training data are stored in format like `中国 商代 最后 一 个 君王 是 谁 ?`. But when evaluation, the Chinese questions in `eval-dataset` should be split character by character like `中 国 商 代 最 后 一 个 君 王 是 谁 ?`.

You can split it by:

`fn=test.q.zh.lc; cat ./data/xqg/$fn | python -u ./tools/zh_split_words.py > ./data/xqg-eval/$fn`

Calculate ROUGE scores for Chinese:

```
python ./xnlg/calc_rouge.py
--ref /path/to/ground_truth
--hyp /path/to/generated_sentences
--zh True
```

Calculate ROUGE scores for other languages:

```
python ./xnlg/calc_rouge.py
--ref /path/to/ground_truth
--hyp /path/to/generated_sentences
```

### Abstractive Summarization (AS)

#### Preparing training data

We use English/French/Chinese Gigaword () processed by extracting the first sentence and the headline of each article, as the source and target sentence. You can get our processed the dataset by:

```bash
bash ./preprocess/get-data-xsumm.sh
```

or directly download at [here](https://drive.google.com/open?id=1DP6pFPBTkGR1sopcZHY2V-3P8MNYpdsL).

#### Training for Zero-Shot AS

```
python xnlg-ft.py
--exp_name xsumm
--dump_path ./dump
--model_path /path/to/pre-trained/XNLG/model
--data_path ./data/processed/XNLG
--transfer_tasks XSumm
--optimizer adam,lr=0.000005
--batch_size 32
--n_epochs 200
--epoch_size 4000
--max_len 120
--max_vocab 95000
--train_layers 1,10
--decode_with_vocab False
--n_enc_layers 10
--n_dec_layers 6
--beam_size 3
--ds_name xgiga
--train_directions en-en
--eval_directions zh-zh
```

#### Training for Supervised AS

For supervised AS, `--train_layers` should be set as `all`. For supervised French AS, just set `--train_directions` and `--eval_directions` as `fr-fr`.

#### Generating Summaries

```
python summ.py
--data_path ./data/processed/XNLG
--model_dir /path/to/exp
--job_name [exp-index]                      # a hash code like `a23h1yv1`
--direction en-fr                           # en-en/fr-fr/zh-zh/en-zh/fr-en/...
```

## References

Please cite the paper [Cross-Lingual Natural Language Generation via Pre-Training](https://arxiv.org/abs/1909.10481) if you found the resources in the repository useful.

```
@article{xnlg,
  title={Cross-Lingual Natural Language Generation via Pre-Training},
  author={Chi, Zewen and Dong, Li and Wei, Furu and Wang, Wenhui and Mao, Xian-Ling and Huang, Heyan},
  journal={arXiv preprint arXiv:1909.10481},
  year={2019}
}
```

