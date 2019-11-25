import os
import torch
import argparse

# from src.evaluation.xqg import XQG_v2, tokens2words, XQG_LANGS
from src.evaluation.xsumm import XSumm, XSumm_LANGS, tokens2words
from src.qg import load_model
from src.utils import to_cuda, AttrDict, bool_flag
from src.model.transformer import TransformerModel
from src.data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
from tqdm import tqdm
# from nlgeval import NLGEval


class XSummEvalOnly(XSumm):

  def __init__(self, encoder, decoder, scores, dico, params):
    super().__init__(encoder, decoder, scores, dico, params)
    self.data = {}
    self.encoder.cuda()
    self.decoder.cuda()
    if self.params.decode_with_vocab: self.setup_vocab_mask(self.dico)

  def summ4dataset(self, direction):
    direction = direction.split("-")
    params = self.params
    encoder = self.encoder
    decoder = self.decoder
    encoder.eval()
    decoder.eval()
    dico = self.dico

    x_lang, y_lang = direction
    print("Performing %s-%s-xsumm" % (x_lang, y_lang))

    X, Y = [], []
    x_lang_id = params.lang2id[x_lang[-2:]]
    y_lang_id = params.lang2id[y_lang[-2:]]
    vocab_mask=self.vocab_mask[y_lang[-2:]] if params.decode_with_vocab else None

    for batch in tqdm(self.get_iterator("test", x_lang, y_lang)):
      (sent_x, len_x), (sent_y, len_y), _ = batch
      lang_x = sent_x.clone().fill_(x_lang_id)
      # lang_y = sent_y.clone().fill_(y_lang_id)

      sent_x, len_x, lang_x = to_cuda(sent_x, len_x, lang_x)

      with torch.no_grad():
        encoded = encoder(
          "fwd", x=sent_x, lengths=len_x, langs=lang_x, causal=False)
        encoded = encoded.transpose(0, 1)

        if params.beam_size == 1:
          decoded, _ = decoder.generate(
            encoded, len_x, y_lang_id, max_len=params.max_dec_len,
            vocab_mask=vocab_mask)
        else:
          decoded, _ = decoder.generate_beam(
            encoded, len_x, y_lang_id, beam_size=params.beam_size,
            length_penalty=0.9, early_stopping=False,
            max_len=params.max_dec_len, vocab_mask=vocab_mask)
    
      for j in range(decoded.size(1)):
        sent = decoded[:, j]
        delimiters = (sent == params.eos_index).nonzero().view(-1)
        assert len(delimiters) >= 1 and delimiters[0].item() == 0
        sent = sent[1:] if len(delimiters) == 1  else sent[1: delimiters[1]]

        trg_tokens = [dico[sent[k].item()] for k in range(len(sent))]
        trg_words = tokens2words(trg_tokens)
        if y_lang.endswith("zh"): Y.append(" ".join("".join(trg_words)))
        else: Y.append(" ".join(trg_words))

    return Y
  
  def summ(self, direction, out_fn):
    print("%s-summ to %s" % (direction, out_fn))
    results = self.summ4dataset(direction)
    with open(out_fn, "w") as fp:
      for line in results:
        fp.write(line + "\n")
  
def get_params():
  parser = argparse.ArgumentParser(description='Evaluation')
  parser.add_argument("--model_dir", type=str, default="",
                      help="Model location")
  # parser.add_argument("--model_path", type=str, default="",
  #                     help="Model location")
  parser.add_argument("--data_path", type=str, default="",
                      help="Data path")
  parser.add_argument("--ds_name", type=str, default="",
                      help="Data path")
  parser.add_argument("--max_len", type=int, default=256,
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
  parser.add_argument("--decode_with_vocab", type=bool_flag, default=False,
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


  parser.add_argument("--out_dir", type=str, default="",
                      help="path to output file of qg results")
  parser.add_argument("--out_fn", type=str, default="",
                      help="path to output file of qg results")
  parser.add_argument("--direction", type=str, default="",
                      help="src_lang", required=True)
  parser.add_argument("--job_name", type=str, default="",
                      help="Model location", required=True)
  parser.add_argument("--cut_dataset", type=int, default=-1,
                      help="Model location")
  params = parser.parse_args()

  # model_name = "best_%s_rouge-l.pth" % params.direction
  model_name = "best_%s_Bleu_4.pth" % params.direction
  model_path = os.path.join(params.model_dir, params.job_name, model_name)
  params.model_path = model_path
  print("use model from", model_path)

  os.makedirs(os.path.join(params.out_dir, params.job_name), exist_ok=True)
  if params.out_fn == "":
    params.out_fn = os.path.join(
      params.out_dir, params.job_name, params.direction)

  return params

if __name__ == "__main__":
  params = get_params()

  encoder, decoder, dico = load_model(params)
  scores = {}
  xs = XSummEvalOnly(encoder, decoder, scores, dico, params)
  xs.summ(params.direction, params.out_fn)
