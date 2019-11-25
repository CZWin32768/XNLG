import os
import io
import argparse
import torch
import copy
import sys

import nltk
nltk.download('punkt')


from src.utils import bool_flag, initialize_exp, AttrDict
from src.evaluation.glue import GLUE
from src.evaluation.xnli import XNLI
from src.evaluation.xqg import XQG_v3
from src.evaluation.xsumm import XSumm
from src.model.transformer import TransformerModel
from src.data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD


GLUE_TASKS = ['MNLI-m', 'MNLI-mm', 'QQP', 'QNLI', 'SST-2', 'CoLA', 'MRPC', 'RTE', 'STS-B', 'WNLI', 'AX_MNLI-m']
XNLI_TASKS = ['XNLI']
XNLG_TASKS = ["XQG", "XSumm"]
TASKS = GLUE_TASKS + XNLI_TASKS + XNLG_TASKS

def get_params():

  # parse parameters
  parser = argparse.ArgumentParser(description='Train on GLUE or XNLI or XNLG')

  # main parameters
  parser.add_argument("--exp_name", type=str, default="",
                      help="Experiment name")
  parser.add_argument("--dump_path", type=str, default="",
                      help="Experiment dump path")
  parser.add_argument("--exp_id", type=str, default="",
                      help="Experiment ID")

  # evaluation task / pretrained model
  parser.add_argument("--transfer_tasks", type=str, default="",
                      help="Transfer tasks, example: 'MNLI-m,RTE,XNLI' ")
  parser.add_argument("--model_path", type=str, default="",
                      help="Model location")

  # data
  parser.add_argument("--data_path", type=str, default="",
                      help="Data path")
  parser.add_argument("--ds_name", type=str, default="xgiga",
                      help="name of dataset: xsumm or xgiga")
  parser.add_argument("--max_vocab", type=int, default=-1,
                      help="Maximum vocabulary size (-1 to disable)")
  parser.add_argument("--min_count", type=int, default=0,
                      help="Minimum vocabulary count")

  # batch parameters
  parser.add_argument("--max_len", type=int, default=256,
                      help="Maximum length of sentences (after BPE)")
  parser.add_argument("--max_len_q", type=int, default=256,
                    help="Maximum length of sentences (after BPE)")
  parser.add_argument("--max_len_a", type=int, default=256,
                    help="Maximum length of sentences (after BPE)")
  parser.add_argument("--max_len_e", type=int, default=256,
                    help="Maximum length of sentences (after BPE)")
  parser.add_argument("--group_by_size", type=bool_flag, default=False,
                      help="Sort sentences by size during the training")
  parser.add_argument("--batch_size", type=int, default=32,
                      help="Number of sentences per batch")
  parser.add_argument("--max_batch_size", type=int, default=0,
                      help="Maximum number of sentences per batch (used in combination with tokens_per_batch, 0 to disable)")
  parser.add_argument("--tokens_per_batch", type=int, default=-1,
                      help="Number of tokens per batch")

  # model / optimization
  parser.add_argument("--finetune_layers", type=str, default='0:_1',
                      help="Layers to finetune. 0 = embeddings, _1 = last encoder layer")
  parser.add_argument("--weighted_training", type=bool_flag, default=False,
                      help="Use a weighted loss during training")
  parser.add_argument("--dropout", type=float, default=0,
                      help="Fine-tuning dropout")
  parser.add_argument("--optimizer_e", type=str, default="adam,lr=0.0001",
                      help="Embedder (pretrained model) optimizer")
  parser.add_argument("--optimizer_p", type=str, default="adam,lr=0.0001",
                      help="Projection (classifier) optimizer")
  parser.add_argument("--optimizer", type=str, default="adam,lr=0.0001",
                      help="Projection (classifier) optimizer")                    
  parser.add_argument("--n_epochs", type=int, default=100,
                      help="Maximum number of epochs")
  parser.add_argument("--epoch_size", type=int, default=-1,
                      help="Epoch size (-1 for full pass over the dataset)")

  # debug
  parser.add_argument("--debug_train", type=bool_flag, default=False,
                      help="Use valid sets for train sets (faster loading)")
  parser.add_argument("--debug_slurm", type=bool_flag, default=False,
                      help="Debug multi-GPU / multi-node within a SLURM job")
  parser.add_argument("--sample_alpha", type=float, default=0,
                      help="Exponent for transforming word counts to probabilities (~word2vec sampling)")
  parser.add_argument("--word_pred", type=float, default=0.15,
                      help="Fraction of words for which we need to make a prediction")

  parser.add_argument("--max_dec_len", type=int, default=80,
                      help="Maximum length of target sentence (after BPE)")

  # decode with vocab

  parser.add_argument("--decode_with_vocab", type=bool_flag, default=False,
                      help="Decode with vocab")
  parser.add_argument("--decode_vocab_sizes", type=str, default="26000,20000",
                      help="decode_vocab_sizes")
  parser.add_argument("--vocab_path", type=str, default="",
                      help="vocab_path")

  # multi-gpu
  parser.add_argument("--local_rank", type=int, default=-1,
                      help="Multi-GPU - Local rank")
  parser.add_argument("--multi_gpu", type=bool_flag, default=False,
                      help="multi-gpu")

  parser.add_argument("--train_layers", type=str, default="",
                      help="train layers of encoder") 
  parser.add_argument("--n_enc_layers", type=int, default=0,
                      help="") 
  parser.add_argument("--n_dec_layers", type=int, default=0,
                      help="") 
  parser.add_argument("--fixed_embeddings", type=bool_flag, default=False,
                    help="fixed_embeddings")
  parser.add_argument("--fixed_position_embeddings", type=bool_flag, default=False,
                      help="fixed_position_embeddings")
  parser.add_argument("--fixed_lang_embeddings", type=bool_flag, default=False,
                      help="fixed_lang_embeddings")
  parser.add_argument("--fixed_task_embeddings", type=bool_flag, default=False,
                      help="fixed_task_embeddings")
  parser.add_argument("--beam_size", type=int, default=1,
                      help="")
  parser.add_argument("--no_init", type=str, default="None",
                      help="dont init with pretrained models")
  
  parser.add_argument("--train_directions", type=str, default="en-en",
                      help="")
  parser.add_argument("--eval_directions", type=str, default="",
                      help="")
  parser.add_argument("--emb_dim", type=int, default=-1,
                      help="Number of sentences per batch")
  parser.add_argument("--reload_emb", type=str, default="",
                      help="path to .vec produced by fasttext")
  parser.add_argument("--cut_dataset", type=int, default=-1,
                      help="Number of sentences in dataset. -1 for full dataset.")
  params = parser.parse_args()

  return params


def read_txt_embeddings(logger, path):
  """
  Reload pretrained embeddings from a text file.
  """
  import numpy as np
  word2id = {}
  vectors = []

  # load pretrained embeddings
  # _emb_dim_file = params.emb_dim
  _emb_dim_file = 0
  with io.open(path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
    for i, line in enumerate(f):
      if i == 0:
        split = line.split()
        assert len(split) == 2
        _emb_dim_file = int(split[1])
        continue
      word, vect = line.rstrip().split(' ', 1)
      vect = np.fromstring(vect, sep=' ')
      if word in word2id:
        logger.warning("Word \"%s\" found twice!" % word)
        continue
      if not vect.shape == (_emb_dim_file,):
        logger.warning("Invalid dimension (%i) for word \"%s\" in line %i."
                        % (vect.shape[0], word, i))
        continue
      assert vect.shape == (_emb_dim_file,)
      word2id[word] = len(word2id)
      vectors.append(vect[None])

  assert len(word2id) == len(vectors)
  logger.info("Loaded %i pretrained word embeddings from %s" % (len(vectors), path))

  # compute new vocabulary / embeddings
  embeddings = np.concatenate(vectors, 0)
  embeddings = torch.from_numpy(embeddings).float()

  # assert embeddings.size() == (len(word2id), params.emb_dim)
  return word2id, embeddings


def load_bin_embeddings(logger, path):
  """
  Reload pretrained embeddings from a fastText binary file.
  """
  import fasttext
  import numpy as np
  model = fasttext.load_model(path)
  words = model.get_labels()
  logger.info("Loaded binary model from %s" % path)

  # compute new vocabulary / embeddings
  embeddings = np.concatenate([model.get_word_vector(w)[None] for w in words], 0)
  embeddings = torch.from_numpy(embeddings).float()
  word2id = {w: i for i, w in enumerate(words)}
  logger.info("Generated embeddings for %i words." % len(words))

  return word2id, embeddings


def set_pretrain_emb(logger, model, dico, word2id, embeddings):
  """
  Pretrain word embeddings.
  """
  n_found = 0
  with torch.no_grad():
    for i in range(len(dico)):
      idx = word2id.get(dico[i], None)
      if idx is None:
        continue
      n_found += 1
      model.embeddings.weight[i] = embeddings[idx].cuda()
      try:
        model.pred_layer.proj.weight[i] = embeddings[idx].cuda()
      except AttributeError:
        pass
  logger.info("Pretrained %i/%i words (%.3f%%)."
              % (n_found, len(dico), 100. * n_found / len(dico)))


def str_to_class(str):
  return getattr(sys.modules[__name__], str)


def run_xnlg():
  params = get_params()

  # initialize the experiment / build sentence embedder
  logger = initialize_exp(params)

  if params.tokens_per_batch > -1:
      params.group_by_size = True

  # check parameters
  assert os.path.isdir(params.data_path)
  assert os.path.isfile(params.model_path)

  # tasks
  params.transfer_tasks = params.transfer_tasks.split(',')
  assert len(params.transfer_tasks) > 0
  assert all([task in TASKS for task in params.transfer_tasks])

  reloaded = torch.load(params.model_path)
  model_params = AttrDict(reloaded['params'])
  logger.info(
    "Supported languages: %s" % ", ".join(model_params.lang2id.keys()))
  params.n_langs = model_params['n_langs']
  params.id2lang = model_params['id2lang']
  params.lang2id = model_params['lang2id']

  
  if "enc_params" in reloaded:
    encoder_model_params = AttrDict(reloaded["enc_params"])
  elif params.n_enc_layers == model_params.n_layers or params.n_enc_layers == 0:
    encoder_model_params = model_params
  else:
    encoder_model_params = AttrDict(reloaded['params'])
    encoder_model_params.n_layers = params.n_enc_layers
    assert model_params.n_layers is not encoder_model_params.n_layers
  
  if "dec_params" in reloaded:
    decoder_model_params = AttrDict(reloaded["dec_params"])
  elif params.n_dec_layers == model_params.n_layers or params.n_dec_layers == 0:
    decoder_model_params = model_params
  else:
    decoder_model_params = AttrDict(reloaded['params'])
    decoder_model_params.n_layers = params.n_dec_layers
    assert model_params.n_layers is not decoder_model_params.n_layers
  
  params.encoder_model_params = encoder_model_params
  params.decoder_model_params = decoder_model_params

  if params.emb_dim != -1:
    encoder_model_params.emb_dim = params.emb_dim
    decoder_model_params.emb_dim = params.emb_dim
  
  # build dictionary / build encoder / build decoder / reload weights
  dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])

  for p in [params, encoder_model_params, decoder_model_params]:
    p.n_words = len(dico)
    p.bos_index = dico.index(BOS_WORD)
    p.eos_index = dico.index(EOS_WORD)
    p.pad_index = dico.index(PAD_WORD)
    p.unk_index = dico.index(UNK_WORD)
    p.mask_index = dico.index(MASK_WORD)

  encoder = TransformerModel(encoder_model_params, dico, is_encoder=True, with_output=False)
  decoder = TransformerModel(decoder_model_params, dico, is_encoder=False, with_output=True)

  def _process_state_dict(state_dict):
    return {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

  if params.no_init == "all":
    logger.info("All Models will not load state dict.!!!")
  elif params.reload_emb != "":
    logger.info("Reloading embedding from %s ..." % params.reload_emb)
    word2id, embeddings = read_txt_embeddings(logger, params.reload_emb)
    set_pretrain_emb(logger, encoder, dico, word2id, embeddings)
    set_pretrain_emb(logger, decoder, dico, word2id, embeddings)
  else:
    if "model" in reloaded:
      if params.no_init != "encoder":
        encoder.load_state_dict(_process_state_dict(reloaded['model']), strict=False)
      if params.no_init != "decoder":
        decoder.load_state_dict(_process_state_dict(reloaded['model']), strict=False)
    else:
      if params.no_init != "encoder":
        encoder.load_state_dict(_process_state_dict(reloaded['encoder']), strict=False)
      if params.no_init != "decoder":
        decoder.load_state_dict(_process_state_dict(reloaded['decoder']))
  
  scores = {}

  # run
  for task in params.transfer_tasks:
      if task == "XQG":
        XQG_v3(encoder, decoder, scores, dico, params).run()
      elif task == "XSumm":
        XSumm(encoder, decoder, scores, dico, params).run()
        

if __name__ == "__main__":
  run_xnlg()