import os
import torch
import argparse

from src.evaluation.xqg import XQG_v3, tokens2words, XQG_LANGS
from src.utils import to_cuda, AttrDict, bool_flag
from src.model.transformer import TransformerModel
from src.data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
from tqdm import tqdm

class XQGEvalOnly(XQG_v3):

  def __init__(self, encoder, decoder, scores, dico, params):
    super().__init__(encoder, decoder, scores, dico, params)

    params = self.params
    self.encoder.cuda()
    self.decoder.cuda()
    self.data = {}
    if self.params.decode_with_vocab: self.setup_vocab_mask(self.dico)

  
  def qg4dataset(self, direction, split="test"):
    direction = direction.split("-")
    params = self.params
    encoder = self.encoder
    decoder = self.decoder
    encoder.eval()
    decoder.eval()
    dico = self.dico

    src_lang, trg_lang = direction
    print("Performing %s-%s-xsumm" % (src_lang, trg_lang))

    results = []

    trg_lang_id = params.lang2id[trg_lang]
    vocab_mask=self.vocab_mask[trg_lang] if params.decode_with_vocab else None
    
    for batch in tqdm(self.get_iterator_v2(
      "test", ae_lang=src_lang, q_lang=src_lang, ds_name=params.ds_name)):

      # (sent_q, len_q), (sent_a, len_a), (sent_e, len_e), _ = batch
      x, lens, _, src_langs, _, _ = self.concat_qae_batch(
        batch, src_lang[-2:], use_task_emb=False, is_test=True)
      x, lens, src_langs = to_cuda(x, lens, src_langs)
      max_len = params.max_dec_len

      with torch.no_grad():
        encoded = encoder("fwd", x=x, lengths=lens, langs=src_langs, causal=False)
        encoded = encoded.transpose(0, 1)

        if params.beam_size == 1:
          decoded, _ = decoder.generate(
            encoded, lens, trg_lang_id, max_len=max_len, vocab_mask=vocab_mask)
        else:
          decoded, _ = decoder.generate_beam(
            encoded, lens, trg_lang_id, beam_size=params.beam_size,
            length_penalty=0.9, early_stopping=False,
            max_len=max_len, vocab_mask=vocab_mask)
    
      for j in range(decoded.size(1)):
    
        sent = decoded[:, j]
        delimiters = (sent == params.eos_index).nonzero().view(-1)
        assert len(delimiters) >= 1 and delimiters[0].item() == 0
        sent = sent[1:] if len(delimiters) == 1  else sent[1: delimiters[1]]

        trg_tokens = [dico[sent[k].item()] for k in range(len(sent))]
        trg_words = tokens2words(trg_tokens)
        if trg_lang == "zh": results.append(" ".join("".join(trg_words)))
        else: results.append(" ".join(trg_words))

    return results

  def qg(self, direction, out_fn=None):
    print("%s-qg to %s" % (direction, out_fn))
    # self.set_data(src_lang, trg_lang)
    results = self.qg4dataset(direction)
    if out_fn is None: return results
    with open(out_fn, "w") as fp:
      for line in results:
        fp.write(line + "\n")

def load_model(params):
  # check parameters
  assert os.path.isdir(params.data_path)
  assert os.path.isfile(params.model_path)
  reloaded = torch.load(params.model_path)

  encoder_model_params = AttrDict(reloaded['enc_params'])
  decoder_model_params = AttrDict(reloaded['dec_params'])

  dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])

  params.n_langs = encoder_model_params['n_langs']
  params.id2lang = encoder_model_params['id2lang']
  params.lang2id = encoder_model_params['lang2id']
  params.n_words = len(dico)
  params.bos_index = dico.index(BOS_WORD)
  params.eos_index = dico.index(EOS_WORD)
  params.pad_index = dico.index(PAD_WORD)
  params.unk_index = dico.index(UNK_WORD)
  params.mask_index = dico.index(MASK_WORD)

  encoder = TransformerModel(encoder_model_params, dico, is_encoder=True, with_output=False)
  decoder = TransformerModel(decoder_model_params, dico, is_encoder=False, with_output=True)

  def _process_state_dict(state_dict):
    return {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

  encoder.load_state_dict(_process_state_dict(reloaded['encoder']))
  decoder.load_state_dict(_process_state_dict(reloaded['decoder']))

  return encoder, decoder, dico


def get_params():
  parser = argparse.ArgumentParser(description='Evaluation')
  parser.add_argument("--model_dir", type=str, default="",
                      help="Model location")
  # parser.add_argument("--model_path", type=str, default="",
  #                     help="Model location")
  parser.add_argument("--data_path", type=str, default="",
                      help="Data path")
  parser.add_argument("--max_len_q", type=int, default=256,
                    help="Maximum length of sentences (after BPE)")
  parser.add_argument("--max_len_a", type=int, default=20,
                    help="Maximum length of sentences (after BPE)")
  parser.add_argument("--max_len_e", type=int, default=230,
                      help="Maximum length of sentences (after BPE)")
  parser.add_argument("--batch_size", type=int, default=16,
                      help="Number of sentences per batch")
  parser.add_argument("--max_batch_size", type=int, default=0,
                      help="Maximum number of sentences per batch (used in combination with tokens_per_batch, 0 to disable)")
  parser.add_argument("--group_by_size", type=bool_flag, default=False,
                      help="Sort sentences by size during the training")
  parser.add_argument("--tokens_per_batch", type=int, default=-1,
                      help="Number of tokens per batch")
  parser.add_argument("--max_dec_len", type=int, default=80,
                      help="Maximum length of target sentence (after BPE)")
  parser.add_argument("--beam_size", type=int, default=3,
                      help="Maximum length of sentences (after BPE)")
  parser.add_argument("--decode_with_vocab", type=bool_flag, default=True,
                      help="Decode with vocab")
  parser.add_argument("--decode_vocab_sizes", type=str, default="95000,95000",
                      help="decode_vocab_sizes")
  parser.add_argument("--vocab_path", type=str, default="",
                      help="vocab_path")
  parser.add_argument("--debug_train", type=bool_flag, default=False,
                      help="Use valid sets for train sets (faster loading)")
  parser.add_argument("--max_vocab", type=int, default=95000,
                      help="Maximum vocabulary size (-1 to disable)")
  parser.add_argument("--min_count", type=int, default=0,
                      help="Minimum vocabulary count")

  parser.add_argument("--ds_name", type=str, default="xqg",
                      help="path to output file of qg results")

  parser.add_argument("--out_dir", type=str, default="",
                      help="path to output file of qg results")
  parser.add_argument("--out_fn", type=str, default="",
                      help="path to output file of qg results")
  parser.add_argument("--direction", type=str, default="",
                      help="direction", required=True)
  parser.add_argument("--job_name", type=str, default="",
                      help="Model location", required=True)
  params = parser.parse_args()

  model_name = "best_%s_Bleu_4.pth" % params.trg_lang
  model_path = os.path.join(params.model_dir, params.job_name, model_name)
  params.model_path = model_path
  print("use model from", model_path)

  params.lang_pair = "%s-%s" % (params.src_lang, params.trg_lang)
  os.makedirs(os.path.join(params.out_dir, params.job_name), exist_ok=True)
  if params.out_fn == "":
    params.out_fn = os.path.join(
      params.out_dir, params.job_name, params.lang_pair)

  return params


if __name__ == "__main__":
  params = get_params()

  assert params.src_lang in ["en", "zh", "en2zh", "zh2en"]
  assert params.trg_lang in ["en", "zh"]

  encoder, decoder, dico = load_model(params)
  scores = {}
  xqgeo = XQGEvalOnly(encoder, decoder, scores, dico, params)
  xqgeo.qg(params.direction, params.out_fn)


