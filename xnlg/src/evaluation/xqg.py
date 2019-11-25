"""
XQG Class

Author: Zewen
Date: 7/17/2019
"""

from logging import getLogger
import os
import copy
import time
import json
from collections import OrderedDict, defaultdict

import torch
from torch import nn
import torch.nn.functional as F
from nlgeval import NLGEval
from tqdm import tqdm

from ..optim import get_optimizer
from ..utils import concat_batches, truncate, to_cuda, mask_out_v2
from ..data.dataset import TripleDataset
from ..data.loader import load_binarized, set_dico_parameters

XQG_LANGS = ["en", "zh"]
XQG_DATASETS = ["en", "zh", "en2zh", "zh2en"]
# XQG_LANGS = ["zh"]
logger = getLogger()
nlgeval = NLGEval(
  no_skipthoughts=True,no_glove=True,metrics_to_omit=['CIDEr'])

def get_parameters(model, train_layers_str):
  ret = []

  fr, to = map(int, train_layers_str.split(","))
  assert fr >= 0
  if fr == 0:
    # add embeddings
    ret += model.embeddings.parameters()
    logger.info("Adding embedding parameters")
    ret += model.position_embeddings.parameters()
    logger.info("Adding positional embedding parameters")
    ret += model.lang_embeddings.parameters()
    logger.info("Adding language embedding parameters")
    fr = 1
  assert fr <= to
  # add attention layers
  # NOTE cross attention is not added
  for i in range(fr, to + 1):
    ret += model.attentions[i-1].parameters()
    ret += model.layer_norm1[i-1].parameters()
    ret += model.ffns[i-1].parameters()
    ret += model.layer_norm2[i-1].parameters()
    logger.info("Adding layer-%s parameters to optimizer" % i)

  return ret

def tokens2words(toks):
  words = []
  for tok in toks:
    if len(words) > 0 and words[-1].endswith("@@"):
      words[-1] = words[-1][:-2] + tok
    else:
      words.append(tok)
  return words


class XQG(object):

  def __init__(self, embedder, scores, params):
    """Initialize XQA trainer / evaluator"""
    self._embedder = embedder
    self.params = params
    self.scores = scores
    self.iter_cache = {}
  
  def setup_vocab_mask(self, dico=None):
    if dico is None: dico = self.embedder.dico
    n_words = len(dico)
    params = self.params

    self.vocab_mask = {}

    decode_vocab_sizes = [int(s) for s in params.decode_vocab_sizes.split(",")]
    assert len(decode_vocab_sizes) == len(XQG_LANGS)

    for lang, sz in zip(XQG_LANGS, decode_vocab_sizes):
      
      fn = os.path.join(params.vocab_path, lang + ".vocab")
      assert os.path.isfile(fn)

      mask = torch.ByteTensor(n_words)
      mask.fill_(0)
      assert mask.sum() == 0
      mask[dico.eos_index] = 1
      # TODO generate unk?
      mask[dico.unk_index] = 1
      count = 0
      with open(fn) as fp:
        for line, _ in zip(fp, range(sz)):
          tok = line.strip().split("\t")[0].split(" ")[0]
          if tok not in dico.word2id:
            # logger.warn("Token %s not in dico" % tok)
            count += 1
          else: mask[dico.word2id[tok]] = 1
      
      logger.warn("%d tokens not in dico" % count)
      self.vocab_mask[lang] = mask
  
  def gen_references(self, dico):
    self.references = {}
    for lang in XQG_LANGS:
      refs = []
      for batch in self.get_iterator("test", lang):
        (sent_q, len_q), _, _, _ = batch
        for j in range(len(len_q)):
          ref_sent = sent_q[1:len_q[j]-1,j]
          ref_toks = [dico[ref_sent[k].item()] for k in range(len(ref_sent))]
          ref_words = tokens2words(ref_toks)

          if lang == "zh": refs.append(" ".join("".join(ref_words)))
          else: refs.append(" ".join(ref_words))

      self.references[lang] = refs
  
  def gen_references_v2(self, dico, eval_directions):
    self.references = {}
    for split in ["dev", "test"]:
      self.references[split] = {}
      for qg_direction in eval_directions:
        ae_lang, q_lang = qg_direction
        if q_lang in self.references: continue
        refs = []
        for batch in self.get_iterator_v2(
          split, ae_lang=ae_lang, q_lang=q_lang, ds_name=self.params.ds_name):
          (sent_q, len_q), _, _, _ = batch
          for j in range(len(len_q)):
            ref_sent = sent_q[1:len_q[j]-1,j]
            ref_toks = [dico[ref_sent[k].item()] for k in range(len(ref_sent))]
            ref_words = tokens2words(ref_toks)

            #zh or en2zh
            if q_lang.endswith("zh"): refs.append(" ".join("".join(ref_words)))
            else: refs.append(" ".join(ref_words))

        self.references[split][q_lang] = refs


  def run(self):
    """Run XQG training /evaluation"""
    params = self.params
    self.data = self.load_data()
    if not self.data["dico"] == self._embedder.dico:
      raise Exception(
        ("dico different between pre-trained model and current data"))
    
    self.embedder = copy.deepcopy(self._embedder)
    self.embedder.cuda()
    # TODO embedder doesn't reload pred layer by default

    self.optimizer = get_optimizer(
      self.embedder.get_parameters(params), params.optimizer)
    
    # distributed
    if params.multi_gpu:
      logger.info("Using nn.parallel.DistributedDataParallel ...")
      self.embedder.parallel(params)
    
    if params.overfitting_test:
      self.single_batch_overfitting_test()
    
    # generate references
    self.gen_references(self.embedder.dico)
    
    # decode with vocab
    if self.params.decode_with_vocab: self.setup_vocab_mask()

    self.best_scores = defaultdict(float)
    
    for epoch in range(params.n_epochs):
      self.epoch = epoch
      logger.info("XQG - Training epoch %d ..." % epoch)
      self.train()
      logger.info("XQG - Evaluating epoch %d ..." % epoch)
      self.eval()

  def concat_qae_batch(self, batch, lang, use_task_emb=False, is_test=False):
    """Concat q, a, e batch."""

    params = self.params
    batch_q, (sent_a, len_a), (sent_e, len_e), idx = batch
    if not is_test: sent_q, len_q = batch_q

    # NOTE positions will be reset, MUST add task embedding
    ea_lens = len_e + len_a
    lens = ea_lens if is_test else ea_lens + len_q
    slen, bs = lens.max().item(), lens.size(0)
    tasks = None

    lang_id = params.lang2id[lang]
    pad_index = params.pad_index
    eos_index = params.eos_index
    if use_task_emb:
      enc_task_index = params.enc_task_index
      dec_task_index = params.dec_task_index

    x = sent_e.new(slen, bs).fill_(pad_index)
    x[:len_e.max().item()].copy_(sent_e)
    
    positions = torch.arange(slen)[:, None].repeat(1, bs).to(sent_e.device)
    langs = sent_e.new(slen, bs).fill_(lang_id)
    if use_task_emb:
      tasks = sent_e.new(slen, bs).fill_(enc_task_index)

    for i in range(bs):
      # copy answer to the batch
      x[len_e[i]: ea_lens[i], i].copy_(sent_a[:len_a[i], i])
      # copy question to the batch
      if not is_test:
        x[ea_lens[i]: lens[i], i].copy_(sent_q[:len_q[i], i])
        if use_task_emb:
          positions[ea_lens[i]:, i] -= ea_lens[i]
          tasks[ea_lens[i]:, i] = dec_task_index
    
    return x, lens, positions, langs, tasks, ea_lens

  def train(self):
    """Finetune for 1 epoch on XQG English training set"""
    
    params = self.params
    # self.embedder.train()
    model = self.embedder.model
    if params.multi_gpu: model = model.module
    model.train()

    # training variables
    losses = []
    ns = 0  # number of sentences
    nw = 0  # number of words
    t = time.time()

    for batch in self.get_iterator("train", "en"):
      # TODO use task embedding?
      # use_task_emb = params.use_task_emb
      # NOTE issue #2 hard code for task embedding, fixed
      x, lens, positions, langs, tasks, ea_lens = self.concat_qae_batch(
        batch, "en", use_task_emb=params.use_task_emb)
      
      # mask out
      x, y, pred_mask = mask_out_v2(params, x, lens, ea_lens)
      x, y, pred_mask, lens, positions, langs, tasks, ea_lens = to_cuda(
        x, y, pred_mask, lens, positions, langs, tasks, ea_lens)
      
      # NOTE nothing masked, continue
      if len(y) == 0: continue

      # forward / loss
      tensor = model(
        'fwd_v2', x=x, lengths=lens, positions=positions,
        langs=langs, mask_type="s2s_training", tasks=tasks, src_lens=ea_lens)
      _, loss = model(
        'predict', tensor=tensor, pred_mask=pred_mask, y=y, get_scores=False)
      
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      bs = len(lens)
      ns += bs
      nw += lens.sum().item()
      losses.append(loss.item())

      # log
      if ns % (100 * bs) < bs:
        logger.info(
          "XQG - Epoch %i - Train iter %7i - %.1f words/s - Loss: %.4f" % (
            self.epoch, ns, nw / (time.time() - t), sum(losses) / len(losses)))
        nw, t = 0, time.time()
        losses = []
  
  def eval(self):
    """Evaluate on XQG test sets, for all languages."""
    params = self.params
    # self.embedder.eval()
    model = self.embedder.model
    if params.multi_gpu: model = model.module
    model.eval()
    dico = self.embedder.dico

    debug = True
    debug_num = 0
    
    scores = OrderedDict({"epoch": self.epoch})
    best_scores = self.best_scores

    # self.iter_data_test()

    for lang in XQG_LANGS:
      
      debug_num = 0

      results = []
      if debug:
        evidences = []
        ans = []

      lang_id = params.lang2id[lang]
      # for batch in tqdm(self.get_iterator("test", lang), total=len(self.references[lang])//8):
      for batch in self.get_iterator("test", lang):
        (sent_q, len_q), (sent_a, len_a), (sent_e, len_e), _ = batch
        x, lens, _, _, _, _ = self.concat_qae_batch(
          batch, lang, use_task_emb=params.use_task_emb, is_test=True)
        # x, lens, mask_type, max_len=100,
                  # lang_ids=None, task_ids=None
        x, lens = to_cuda(x, lens)
        # NOTE issue#1 fixed
        max_len = params.max_dec_len
        # greedy decode
        # decoded: (trg_len, bs)
        decoded, dec_lens = model.generate_with_vocab(
          x=x,
          lens=lens,
          mask_type="s2s_decoding",
          max_len=max_len,
          lang_ids=(lang_id, lang_id),
          task_ids=(
            params.enc_task_index, params.dec_task_index
          ) if params.use_task_emb else None,
          vocab_mask=self.vocab_mask[lang] if params.decode_with_vocab else None
        )
        # convert sentences to words
        # for jth sentence in batch
        for j in range(decoded.size(1)):
          
          sent = decoded[:, j]
          delimiters = (sent == params.eos_index).nonzero().view(-1)
          assert len(delimiters) >= 1 and delimiters[0].item() == 0
          sent = sent[1:] if len(delimiters) == 1  else sent[1: delimiters[1]]

          trg_tokens = [dico[sent[k].item()] for k in range(len(sent))]
          trg_words = tokens2words(trg_tokens)
          if lang == "zh": results.append(" ".join("".join(trg_words)))
          else: results.append(" ".join(trg_words))

          if len(evidences) < 5:
            e_sent = sent_e[1:len_e[j], j]
            e_toks = [dico[e_sent[k].item()] for k in range(len(e_sent))]
            e_words = tokens2words(e_toks)
            evidences.append(e_words)
            a_sent = sent_a[1:len_a[j], j]
            a_toks = [dico[a_sent[k].item()] for k in range(len(a_sent))]
            a_words = tokens2words(a_toks)
            ans.append(a_words)
  
        debug_num += 1
        # if debug and debug_num >= 10: break

      # calculate bleu
      # print(len(self.references[lang]), len(results))
      if debug:
        # print(len(evidences))
        logger.info("%d res %d ref" % (
          len(results), len(self.references[lang])))
        for i in range(5):
          logger.info("%d Evidence: %s\nAnswer: %s\nGenerated: %s\nReference: %s\n" % (
              i, 
              " ".join(evidences[i]),
              " ".join(ans[i]),
              results[i], 
              self.references[lang][i])
          )
      eval_res = nlgeval.compute_metrics([self.references[lang][:len(results)]], results)
      # bleu1, bleu2
      best_scores[lang] = max(best_scores[lang], eval_res["Bleu_4"])
      logger.info("XQG - %s - Epoch %d - Best BLEU-4: %.5f - scores: %s" % (
        lang, self.epoch, best_scores[lang], eval_res))


  
  def get_iterator(self, splt, lang):
    # NOTE lang could be a str like "zh" or a tuple like ("en", "en")
    # assert splt == "test" or splt == "train" and lang == "en"
    if type(lang) == tuple:
      assert len(lang) == 2
      lang1, lang2 = lang
      if lang1 == lang2: lang = lang1
    if type(lang) == str:
      assert splt == "test" or splt == "train" and lang != "zh"

    return self.data[lang][splt].get_iterator(
      shuffle=(splt == 'train'),
      # shuffle=True, # TODO debug
      group_by_size=self.params.group_by_size,
      return_indices=True)
  
  def iter_data_test(self):
    """test for iterating data"""
    self.data = self.load_data()
    iterator = self.get_iterator("train", "en")
    num = 10
    dico = self.embedder.dico
    for batch in iterator:
      (sent_q, len_q), (sent_a, len_a), (sent_e, len_e), idx = batch
      # print("q:", sent_q, "len_q:", len_q)
      ref_sent = sent_q[:len_q[0],0]
      ref_words = " ".join([dico[sent_q[k][0].item()] for k in range(len(ref_sent))])
      print("Q sentence:\n", ref_words)
      print("============================")
      # print("a:", sent_a, "len_a:", len_a)
      ref_sent = sent_a[:len_a[0],0]
      ref_words = " ".join([dico[sent_a[k][0].item()] for k in range(len(ref_sent))])
      print("A sentence:\n", ref_words)
      print("============================")
      # print("e:", sent_e, "len_e:", len_e)
      ref_sent = sent_e[:len_e[0],0]
      ref_words = " ".join([dico[sent_e[k][0].item()] for k in range(len(ref_sent))])
      print("E sentence:\n", ref_words)
      print("============================")
      num -= 1
      if num <= 0: break
  
  def single_batch_overfitting_test(self):

    params = self.params
    # self.embedder.train()
    model = self.embedder.model
    dico = self.embedder.dico

    # training variables

    itr = self.get_iterator("train", "en")
    batch = next(itr)
    lang_id = params.lang2id["en"]
    epoch = 0
    while True:
      losses = []
      ns = 0  # number of sentences
      nw = 0  # number of words
      t = time.time()
      # train
      model.train()

      res = []
      ref = []

      for i in range(30):
        x, lens, positions, langs, tasks, ea_lens = self.concat_qae_batch(
        batch, "en", use_task_emb=True)
        x, y, pred_mask = mask_out_v2(params, x, lens, ea_lens)
        x, y, pred_mask, lens, positions, langs, tasks, ea_lens = to_cuda(
          x, y, pred_mask, lens, positions, langs, tasks, ea_lens)
        # NOTE nothing masked, continue
        if len(y) == 0: continue
        tensor = model(
        'fwd_v2', x=x, lengths=lens, positions=positions,
        langs=langs, mask_type="s2s_training", tasks=tasks, src_lens=ea_lens)
        scores, loss = model(
          'predict', tensor=tensor, pred_mask=pred_mask, y=y, get_scores=True)
        top_indices = torch.topk(scores, 1)[1].squeeze(1)
        # print(y[:10], top_indices[:10])
        print("#mask:%d #correct:%d" % (len(y), (top_indices == y).sum()))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        bs = len(lens)
        ns += bs
        nw += lens.sum().item()
        losses.append(loss.item())
      epoch += 1
      logger.info(
        "XQG - Epoch %i - Train iter %7i - %.1f words/s - Loss: %.4f" % (
          epoch, ns, nw / (time.time() - t), sum(losses) / len(losses)))
      nw, t = 0, time.time()
      losses = []

      # eval
      model.eval()
      (sent_q, len_q), _, _, _ = batch
      x, lens, _, _, _, _ = self.concat_qae_batch(
        batch, "en", use_task_emb=params.use_task_emb, is_test=True)
      x, lens = to_cuda(x, lens)
      # NOTE issue#1 fixed
      max_len = params.max_dec_len
      # greedy decode
      # decoded: (trg_len, bs)
      decoded, dec_lens = model.generate_v2(
        x=x,
        lens=lens,
        mask_type="s2s_decoding",
        max_len=max_len,
        lang_ids=(lang_id, lang_id),
        task_ids=(
          params.enc_task_index, params.dec_task_index
        ) if params.use_task_emb else None
      )
      # convert sentences to words
      # for jth sentence in batch
      for j in range(decoded.size(1)):
        
        sent = decoded[:, j]
        delimiters = (sent == params.eos_index).nonzero().view(-1)
        assert len(delimiters) >= 1 and delimiters[0].item() == 0
        sent = sent[1:] if len(delimiters) == 1  else sent[1: delimiters[1]]

        trg_words = [dico[sent[k].item()] for k in range(len(sent))]
        trg_words = tokens2words(trg_words)

        ref_sent = sent_q[1:len_q[j]-1,j]
        ref_words = [dico[ref_sent[k].item()] for k in range(len(ref_sent))]
        ref_words = tokens2words(ref_words)
        print("========================================")
        print("Generated sentence: ", " ".join(trg_words), "\nReference sentence:", " ".join(ref_words))

        res.append(trg_words)
        ref.append([ref_words])
      
      print("BLEU:", corpus_bleu(ref, res))

  def get_iterator_v2(self, splt, lang=None, ae_lang=None, q_lang=None, ds_name='xqg'):
    assert lang != None or ae_lang != None and q_lang != None
    if lang != None: ae_lang = q_lang = lang
    ae_lang = self._parse_lang(ae_lang)
    q_lang = self._parse_lang(q_lang)
    assert ae_lang[0] == q_lang[0]
    logger.info("Getting iterator -- ae_lang: (%s, %s), q_lang: (%s, %s)" % (
      ae_lang[0], ae_lang[1], q_lang[0], q_lang[1]))
    return self.get_or_load_data(ae_lang, q_lang, splt, ds_name).get_iterator(
      shuffle=(splt == 'train'),
      group_by_size=self.params.group_by_size,
      return_indices=True)
  
  def next_batch(self, splt, ae_lang, q_lang, ds_name='xqg'):
    
    key = (splt, ae_lang, q_lang)
    if key not in self.iter_cache:
      self.iter_cache[key] = self.get_iterator_v2(
        splt, ae_lang=ae_lang, q_lang=q_lang, ds_name=ds_name)
    try:
      ret = next(self.iter_cache[key])
    except StopIteration:
      self.iter_cache[key] = self.get_iterator_v2(
        splt, ae_lang=ae_lang, q_lang=q_lang, ds_name=ds_name)
      ret = next(self.iter_cache[key])
    return ret

  def _parse_lang(self, lang):
    if type(lang) == tuple:
      assert len(lang) == 2
      lang1, lang2 = lang
      assert lang1 in XQG_LANGS
      assert lang2 in XQG_LANGS
      return (lang1, lang2)
    if type(lang) == str:
      if lang in XQG_LANGS:
        return (lang, lang)
      else:
        lang1, lang2 = lang.split("2")
        assert lang1 in XQG_LANGS
        assert lang2 in XQG_LANGS
        return (lang1, lang2)
  
  def lang2str(self, lang):
    lang1, lang2 = lang
    if lang1 == lang2: return lang1
    return "%s-%s" % (lang1, lang2)

  def get_or_load_data(self, ae_lang, q_lang, splt, ds_name='xqg'):
    params = self.params
    data = self.data

    lang = (ae_lang, q_lang)
    if lang in self.data:
      if splt in self.data[lang]:
        return self.data[lang][splt]
    else:
      self.data[lang] = {}

    dpath = os.path.join(params.data_path, 'eval', ds_name)
    
    q = load_binarized(os.path.join(dpath, "%s.q.%s.pth" % (
      splt, self.lang2str(q_lang))), params)
    a = load_binarized(os.path.join(dpath, "%s.a.%s.pth" % (
      splt, self.lang2str(ae_lang))), params)
    e = load_binarized(os.path.join(dpath, "%s.e.%s.pth" % (
      splt, self.lang2str(ae_lang))), params)
    data["dico"] = data.get("dico", q["dico"])
    set_dico_parameters(params, data, q["dico"])
    set_dico_parameters(params, data, a["dico"])
    set_dico_parameters(params, data, e["dico"])

    data[lang][splt] = TripleDataset(
      q["sentences"], q["positions"],
      a["sentences"], a["positions"],
      e["sentences"], e["positions"],
      params)
    data[lang][splt].remove_empty_sentences()
    data[lang][splt].cut_long_sentences(
      params.max_len_q,
      params.max_len_a,
      params.max_len_e)
    
    return self.data[lang][splt]

  def load_data(self):
    """Load XQG data."""

    params = self.params
    data = {lang: {
      splt: {} for splt in ["train", "test"]} for lang in XQG_LANGS}
    dpath = os.path.join(params.data_path, 'eval', 'xqg')

    for splt in ["train", "test"]:
      for lang in XQG_LANGS:

        if splt == "train" and lang != "en":
          del data[lang]['train']
          continue
        q = load_binarized(os.path.join(dpath, "%s.q.%s.pth" % (
          splt, lang)), params)
        a = load_binarized(os.path.join(dpath, "%s.a.%s.pth" % (
          splt, lang)), params)
        e = load_binarized(os.path.join(dpath, "%s.e.%s.pth" % (
          splt, lang)), params)
        data["dico"] = data.get("dico", q["dico"])
        set_dico_parameters(params, data, q["dico"])
        set_dico_parameters(params, data, a["dico"])
        set_dico_parameters(params, data, e["dico"])

        # create dataset
        data[lang][splt] = TripleDataset(
          q["sentences"], q["positions"],
          a["sentences"], a["positions"],
          e["sentences"], e["positions"],
          params)

        # if splt == "train":
        #   data[lang][splt].remove_empty_sentences()
        #   data[lang][splt].remove_long_sentences(params.max_len)
        # TODO remove long sen with 3 different max len, ref: 479
        # TODO for test, if a sen is too long, just clip it.
        data[lang][splt].remove_empty_sentences()
        # data[lang][splt].remove_long_sentences(params.max_len)
        data[lang][splt].cut_long_sentences(
          params.max_len_q,
          params.max_len_a,
          params.max_len_e)

    return data
  
  def load_translated_qg_data(self):
    """Load translated xqg dataset."""

    params = self.params
    data = {}
    dpath = os.path.join(params.data_path, 'eval', 'xqg-trans')
    for lang1 in XQG_LANGS:
      for lang2 in XQG_LANGS:
        if lang1 == lang2: continue

        # TODO NOTE because en-zh currently is not processed
        # break it for walkaround
        if lang1 == "en" and lang2 == "zh":
          break

        data[(lang1, lang2)] = {}
        for splt in ["train", "test"]:
          """
          NOTE lang1-lang2 is just the translation direction,
          rather than qg direction!!!!!
          """
          q = load_binarized(os.path.join(dpath, "%s.q.%s-%s.pth" % (
            splt, lang1, lang2)), params)
          a = load_binarized(os.path.join(dpath, "%s.a.%s-%s.pth" % (
            splt, lang1, lang2)), params)
          e = load_binarized(os.path.join(dpath, "%s.e.%s-%s.pth" % (
            splt, lang1, lang2)), params)
          data["dico"] = data.get("dico", q["dico"])
          set_dico_parameters(params, data, q["dico"])
          set_dico_parameters(params, data, a["dico"])
          set_dico_parameters(params, data, e["dico"])

          # create dataset
          data[(lang1, lang2)][splt] = TripleDataset(
            q["sentences"], q["positions"],
            a["sentences"], a["positions"],
            e["sentences"], e["positions"],
            params)

          # if splt == "train":
          #   data[lang][splt].remove_empty_sentences()
          #   data[lang][splt].remove_long_sentences(params.max_len)
          # TODO remove long sen with 3 different max len, ref: 479
          # TODO for test, if a sen is too long, just clip it.
          data[(lang1, lang2)][splt].remove_empty_sentences()
          # data[lang][splt].remove_long_sentences(params.max_len)
          data[(lang1, lang2)][splt].cut_long_sentences(
            params.max_len_q,
            params.max_len_a,
            params.max_len_e)

    return data


class XQG_v2(XQG):
  """
  Finetuning Encoder-Decoder XLM for XQG
  But only for en-en-qg, zh-zh-qg setting.
  """

  def __init__(self, encoder, decoder, scores, dico, params):
    self.encoder = encoder
    self.decoder = decoder
    self.params = params
    self.scores = scores
    self.dico = dico
  
  def run(self):
    params = self.params

    # train_direction = params.train_direction.split("-")
    # eval_directions = [d.split("-") for d in params.eval_directions]

    self.data = self.load_data()
    # self.data = {}
    if not self.data["dico"] == self.dico:
      raise Exception(
        ("dico different between pre-trained model and current data"))
    
    self.encoder.cuda()
    self.decoder.cuda()
    parameters = []
    if params.train_layers == "all":
      parameters.extend([_ for _ in self.encoder.parameters()])
      parameters.extend([_ for _ in self.decoder.parameters()])
    elif params.train_layers == "decoder":
      parameters = self.decoder.parameters()
    elif params.train_layers == "encoder":
      parameters = self.encoder.parameters()
    else:
      parameters = get_parameters(self.encoder, params.train_layers)
    self.optimizer = get_optimizer(parameters, params.optimizer)

    # self.gen_references_v2(self.dico, eval_directions)
    self.gen_references(self.dico)
    if self.params.decode_with_vocab: self.setup_vocab_mask(self.dico)

    self.best_scores = defaultdict(float)
    
    for epoch in range(params.n_epochs):
      self.epoch = epoch
      logger.info("XQG - Training epoch %d ..." % epoch)
      self.train()
      logger.info("XQG - Evaluating epoch %d ..." % epoch)
      self.eval()
  
  def train(self):
    params = self.params
    encoder = self.encoder
    decoder = self.decoder
    
    encoder.train()
    decoder.train()

    # training variables
    losses = []
    ns = 0  # number of sentences
    nw = 0  # number of words
    t = time.time()
    lang_id = params.lang2id["en"]

    # ae_lang, q_lang = train_direction
    # for batch in self.get_iterator_v2("train", ae_lang=ae_lang, q_lang=q_lang):
    for batch in self.get_iterator("train", "en"):
      
      (sent_q, len_q), _, _, _ = batch
      x1, len1, positions, lang1, _, _ = self.concat_qae_batch(
        batch, "en", use_task_emb=False, is_test=True)
      lang2 = sent_q.clone().fill_(lang_id)
      alen = torch.arange(len_q.max(), dtype=torch.long, device=len_q.device)
      pred_mask = alen[:, None] < len_q[None] - 1
      y = sent_q[1:].masked_select(pred_mask[:-1])
      assert len(y) == (len_q-1).sum().item()

      x1, len1, lang1, sent_q, len_q, lang2, y = to_cuda(
        x1, len1, lang1, sent_q, len_q, lang2, y)
      
      enc1 = self.encoder('fwd', x=x1, lengths=len1, langs=lang1, causal=False)
      enc1 = enc1.transpose(0, 1)

      dec2 = self.decoder(
        'fwd', x=sent_q, lengths=len_q, langs=lang2,
        causal=True, src_enc=enc1, src_len=len1)
      
      _, loss = self.decoder('predict', tensor=dec2, pred_mask=pred_mask, y=y, get_scores=False)
      
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      bs = len(len_q)
      ns += bs
      nw += len_q.sum().item()
      losses.append(loss.item())

      # log
      if ns % (100 * bs) < bs:
        logger.info(
          "XQG - Epoch %i - Train iter %7i - %.1f words/s - Loss: %.4f" % (
            self.epoch, ns, nw / (time.time() - t), sum(losses) / len(losses)))
        nw, t = 0, time.time()
        losses = []
  
  def eval(self):
    params = self.params
    encoder = self.encoder
    decoder = self.decoder
    encoder.eval()
    decoder.eval()
    dico = self.dico
    best_scores = self.best_scores

    for lang in XQG_LANGS:
      
      debug_num = 0

      results = []
      evidences = []
      ans = []

      lang_id = params.lang2id[lang]
      vocab_mask=self.vocab_mask[lang] if params.decode_with_vocab else None
      # for batch in tqdm(self.get_iterator("test", lang), total=len(self.references[lang])//8):
      for batch in self.get_iterator("test", lang):
        (sent_q, len_q), (sent_a, len_a), (sent_e, len_e), _ = batch
        x, lens, _, langs, _, _ = self.concat_qae_batch(
          batch, lang, use_task_emb=False, is_test=True)
        x, lens, langs = to_cuda(x, lens, langs)
        max_len = params.max_dec_len

        with torch.no_grad():
          encoded = encoder("fwd", x=x, lengths=lens, langs=langs, causal=False)
          encoded = encoded.transpose(0, 1)

          if params.beam_size == 1:
            decoded, _ = decoder.generate(
              encoded, lens, lang_id, max_len=max_len, vocab_mask=vocab_mask)
          else:
            # TODO config for length_penalty and early_stopping
            decoded, _ = decoder.generate_beam(
              encoded, lens, lang_id, beam_size=params.beam_size,
              length_penalty=0.9, early_stopping=False,
              max_len=max_len, vocab_mask=vocab_mask)

        for j in range(decoded.size(1)):
          
          sent = decoded[:, j]
          delimiters = (sent == params.eos_index).nonzero().view(-1)
          assert len(delimiters) >= 1 and delimiters[0].item() == 0
          sent = sent[1:] if len(delimiters) == 1  else sent[1: delimiters[1]]

          trg_tokens = [dico[sent[k].item()] for k in range(len(sent))]
          trg_words = tokens2words(trg_tokens)
          if lang == "zh": results.append(" ".join("".join(trg_words)))
          else: results.append(" ".join(trg_words))

          if len(evidences) < 5:
            e_sent = sent_e[1:len_e[j], j]
            e_toks = [dico[e_sent[k].item()] for k in range(len(e_sent))]
            e_words = tokens2words(e_toks)
            evidences.append(e_words)
            a_sent = sent_a[1:len_a[j], j]
            a_toks = [dico[a_sent[k].item()] for k in range(len(a_sent))]
            a_words = tokens2words(a_toks)
            ans.append(a_words)
        
        debug_num += 1
        # if debug_num >= 10: break

      # calculate bleu
      # print(len(self.references[lang]), len(results))
      logger.info("%d res %d ref" % (
          len(results), len(self.references[lang])))
      for i in range(5):
        logger.info("%d Evidence: %s\nAnswer: %s\nGenerated: %s\nReference: %s\n" % (
            i, 
            " ".join(evidences[i]),
            " ".join(ans[i]),
            results[i], 
            self.references[lang][i])
        )
      eval_res = nlgeval.compute_metrics([self.references[lang][:len(results)]], results)
      # bleu1, bleu2
      # best_scores[lang] = max(best_scores[lang], eval_res["Bleu_4"])
      if eval_res["Bleu_4"] > best_scores[lang]:
        logger.info("New best Bleu_4 score! Saving model...")
        best_scores[lang] = eval_res["Bleu_4"]
        self.save("best_%s_Bleu_4" % lang)

      logger.info("XQG - %s - Epoch %d - Best BLEU-4: %.5f - scores: %s" % (
        lang, self.epoch, best_scores[lang], eval_res))

  def save(self, name):
    path = os.path.join(self.params.dump_path, "%s.pth" % name)
    logger.info("Saving %s to %s ..." % (name, path))
    data = {
      "epoch": getattr(self, "epoch", 0),
      "encoder": self.encoder.state_dict(),
      "decoder": self.decoder.state_dict(),
      "enc_params": {
        k: v for k, v in self.params.encoder_model_params.__dict__.items()},
      "dec_params": {
        k: v for k, v in self.params.decoder_model_params.__dict__.items()},
      "dico_id2word": self.dico.id2word,
      "dico_word2id": self.dico.word2id,
      "dico_counts": self.dico.counts,
      "params": {k: v for k, v in self.params.__dict__.items()}
    }
    torch.save(data, path)


class XQG_v3(XQG_v2):
  """ 
  Enable for (en/zh/en2zh/zh2en)-(en/zh/en2zh/zh2en)-qg 
  for training and eval 
  """

  def __init__(self, encoder, decoder, scores, dico, params):
    self.encoder = encoder
    self.decoder = decoder
    self.params = params
    self.scores = scores
    self.dico = dico
    self.iter_cache = {}
  
  def run(self):
    params = self.params

    # train_direction = params.train_direction.split("-")
    train_directions = [d.split("-") for d in params.train_directions.split(",")]
    eval_directions = [d.split("-") for d in params.eval_directions.split(",")]

    self.data = {}
    
    self.encoder.cuda()
    self.decoder.cuda()
    parameters = []
    if params.train_layers == "all":
      parameters.extend([_ for _ in self.encoder.parameters()])
      parameters.extend([_ for _ in self.decoder.parameters()])
    elif params.train_layers == "decoder":
      parameters = self.decoder.parameters()
    elif params.train_layers == "encoder":
      parameters = self.encoder.parameters()
    else:
      parameters = get_parameters(self.encoder, params.train_layers)
    self.optimizer = get_optimizer(parameters, params.optimizer)

    self.gen_references_v2(self.dico, eval_directions)
    if self.params.decode_with_vocab: self.setup_vocab_mask(self.dico)

    self.best_scores = defaultdict(float)
    
    for epoch in range(params.n_epochs):
      self.epoch = epoch
      logger.info("XQG - Training epoch %d ..." % epoch)
      self.train(train_directions)
      logger.info("XQG - Evaluating epoch %d ..." % epoch)
      self.eval(eval_directions, "dev", True)
      self.eval(eval_directions, "test", False)
  
  def train(self, train_directions):
    params = self.params
    encoder = self.encoder
    decoder = self.decoder
    
    encoder.train()
    decoder.train()

    # training variables
    losses = []
    ns = 0  # number of sentences
    nw = 0  # number of words
    t = time.time()

    # ae_lang, q_lang = train_direction
    # lang2_id = params.lang2id[q_lang[-2:]]
    n_train_drt = len(train_directions)

    for step_idx in range(params.epoch_size):
      
      ae_lang, q_lang = train_directions[step_idx % n_train_drt]
      lang2_id = params.lang2id[q_lang[-2:]]

      batch = self.next_batch("train", ae_lang, q_lang, params.ds_name)
      (sent_q, len_q), _, _, _ = batch

      x1, len1, positions, lang1, _, _ = self.concat_qae_batch(
        batch, ae_lang[-2:], use_task_emb=False, is_test=True)
      lang2 = sent_q.clone().fill_(lang2_id)
      alen = torch.arange(len_q.max(), dtype=torch.long, device=len_q.device)
      pred_mask = alen[:, None] < len_q[None] - 1
      y = sent_q[1:].masked_select(pred_mask[:-1])
      assert len(y) == (len_q-1).sum().item()

      x1, len1, lang1, sent_q, len_q, lang2, y = to_cuda(
        x1, len1, lang1, sent_q, len_q, lang2, y)
      
      enc1 = self.encoder('fwd', x=x1, lengths=len1, langs=lang1, causal=False)
      enc1 = enc1.transpose(0, 1)

      dec2 = self.decoder(
        'fwd', x=sent_q, lengths=len_q, langs=lang2,
        causal=True, src_enc=enc1, src_len=len1)
      
      _, loss = self.decoder('predict', tensor=dec2, pred_mask=pred_mask, y=y, get_scores=False)
      
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      bs = len(len_q)
      ns += bs
      nw += len_q.sum().item()
      losses.append(loss.item())

      # log
      if ns % (100 * bs) < bs:
        logger.info(
          "XQG - Epoch %i - Train iter %7i - %.1f words/s - Loss: %.4f" % (
            self.epoch, ns, nw / (time.time() - t), sum(losses) / len(losses)))
        nw, t = 0, time.time()
        losses = []

  def eval(self, eval_directions, split="test", save=True):
    params = self.params
    encoder = self.encoder
    decoder = self.decoder
    encoder.eval()
    decoder.eval()
    dico = self.dico
    best_scores = self.best_scores

    for qg_direction in eval_directions:
      ae_lang, q_lang = qg_direction
      logger.info("Evaluating %s-%s-qg on %s set" % (ae_lang, q_lang, split))
      
      debug_num = 0

      results = []
      evidences = []
      ans = []

      trg_lang_id = params.lang2id[q_lang[-2:]]
      vocab_mask=self.vocab_mask[q_lang[-2:]] if params.decode_with_vocab else None
      # for batch in tqdm(self.get_iterator("test", lang), total=len(self.references[lang])//8):
      for batch in self.get_iterator_v2(
        split, ae_lang=ae_lang, q_lang=q_lang, ds_name=params.ds_name):
        (sent_q, len_q), (sent_a, len_a), (sent_e, len_e), _ = batch
        x, lens, _, langs, _, _ = self.concat_qae_batch(
          batch, ae_lang[-2:], use_task_emb=False, is_test=True)
        x, lens, langs = to_cuda(x, lens, langs)
        max_len = params.max_dec_len

        with torch.no_grad():
          encoded = encoder("fwd", x=x, lengths=lens, langs=langs, causal=False)
          encoded = encoded.transpose(0, 1)

          if params.beam_size == 1:
            decoded, _ = decoder.generate(
              encoded, lens, trg_lang_id, max_len=max_len, vocab_mask=vocab_mask)
          else:
            # TODO config for length_penalty and early_stopping
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
          if q_lang.endswith("zh"): results.append(" ".join("".join(trg_words)))
          else: results.append(" ".join(trg_words))

          if len(evidences) < 5:
            e_sent = sent_e[1:len_e[j], j]
            e_toks = [dico[e_sent[k].item()] for k in range(len(e_sent))]
            e_words = tokens2words(e_toks)
            evidences.append(e_words)
            a_sent = sent_a[1:len_a[j], j]
            a_toks = [dico[a_sent[k].item()] for k in range(len(a_sent))]
            a_words = tokens2words(a_toks)
            ans.append(a_words)
        
        debug_num += 1
        # if debug_num >= 10: break

      # calculate bleu
      # print(len(self.references[lang]), len(results))
      logger.info("%d res %d ref" % (
          len(results), len(self.references[split][q_lang])))
      for i in range(5):
        logger.info("%d Evidence: %s\nAnswer: %s\nGenerated: %s\nReference: %s\n" % (
            i, 
            " ".join(evidences[i]),
            " ".join(ans[i]),
            results[i], 
            self.references[split][q_lang][i])
        )
      eval_res = nlgeval.compute_metrics([self.references[split][q_lang][:len(results)]], results)
      # bleu1, bleu2
      # best_scores[lang] = max(best_scores[lang], eval_res["Bleu_4"])
      qg_direction_str = "-".join(qg_direction)
      if save:
        if eval_res["Bleu_4"] > best_scores[qg_direction_str]:
          logger.info("New best Bleu_4 score! Saving model...")
          best_scores[qg_direction_str] = eval_res["Bleu_4"]
          self.save("best_%s_Bleu_4" % qg_direction_str)

      logger.info("XQG - %s - Epoch %d - Best BLEU-4: %.5f - scores: %s" % (
        qg_direction_str, self.epoch, best_scores[qg_direction_str], eval_res))
