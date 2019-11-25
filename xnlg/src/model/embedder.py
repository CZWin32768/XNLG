# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import torch

from torch import nn
from .transformer import TransformerModel
from ..data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
from ..utils import AttrDict


logger = getLogger()


class SentenceEmbedder(object):

    @staticmethod
    def reload(path, params, cls_name=TransformerModel):
        """
        Create a sentence embedder from a pretrained model.
        """
        # reload model
        reloaded = torch.load(path)
        state_dict = reloaded['model']

        # handle models from multi-GPU checkpoints
        if 'checkpoint' in path:
            state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

        # reload dictionary and model parameters
        dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])
        pretrain_params = AttrDict(reloaded['params'])
        pretrain_params.n_words = len(dico)
        pretrain_params.bos_index = dico.index(BOS_WORD)
        pretrain_params.eos_index = dico.index(EOS_WORD)
        pretrain_params.pad_index = dico.index(PAD_WORD)
        pretrain_params.unk_index = dico.index(UNK_WORD)
        pretrain_params.mask_index = dico.index(MASK_WORD)

        # if "n_nlu_layers" in params: 
        #     pretrain_params.n_nlu_layers = params.n_nlu_layers
        # if "n_task_layers" in params: 
        #     pretrain_params.n_task_layers = params.n_task_layers
        # if "n_lang_layers" in params: 
        #     pretrain_params.n_lang_layers = params.n_lang_layers
        
        # TODO config n layers to load

        # build model and reload weights
        model = cls_name(pretrain_params, dico, True, True, params.use_task_emb)
        # model = cls_name(params, dico, True, True, params.use_task_emb)
        # NOTE task embedding is not included in the Facebook XLM15
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        # adding missing parameters
        params.max_batch_size = 0

        return SentenceEmbedder(model, dico, pretrain_params)
        # return SentenceEmbedder(model, dico, params)

    def __init__(self, model, dico, pretrain_params):
        """
        Wrapper on top of the different sentence embedders.
        Returns sequence-wise or single-vector sentence representations.
        """
        self.pretrain_params = {k: v for k, v in pretrain_params.__dict__.items()}
        self.model = model
        self.dico = dico
        self.n_layers = model.n_layers
        self.out_dim = model.dim
        self.n_words = model.n_words

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def cuda(self):
        self.model.cuda()
    
    def parallel(self, params):
        self.model =  nn.parallel.DistributedDataParallel(
            self.model, device_ids=[params.local_rank],
            output_device=params.local_rank, broadcast_buffers=False)

    def get_parameters(self, params):

        layer_range = params.finetune_layers

        s = layer_range.split(':')
        assert len(s) == 2
        i, j = int(s[0].replace('_', '-')), int(s[1].replace('_', '-'))

        # negative indexing
        i = self.n_layers + i + 1 if i < 0 else i
        j = self.n_layers + j + 1 if j < 0 else j

        # sanity check
        assert 0 <= i <= self.n_layers
        assert 0 <= j <= self.n_layers

        if i > j:
            return []

        parameters = []

        # embeddings
        if i == 0:
            # embeddings
            if not params.fixed_embeddings:
                parameters += self.model.embeddings.parameters()
                logger.info("Adding embedding parameters to optimizer")
            # positional embeddings
            if self.pretrain_params['sinusoidal_embeddings'] is False \
                and not params.fixed_position_embeddings:
                parameters += self.model.position_embeddings.parameters()
                logger.info("Adding positional embedding parameters to optimizer")
            # language embeddings
            if hasattr(self.model, 'lang_embeddings') and \
                not params.fixed_lang_embeddings:
                parameters += self.model.lang_embeddings.parameters()
                logger.info("Adding language embedding parameters to optimizer")
            # task embeddings
            if hasattr(self.model, "task_embeddings") and \
                not params.fixed_task_embeddings:
                parameters += self.model.task_embeddings.parameters()
                logger.info("Adding task embedding parameters to optimizer")
            parameters += self.model.layer_norm_emb.parameters()
        # layers
        for l in range(max(i - 1, 0), j):
            parameters += self.model.attentions[l].parameters()
            parameters += self.model.layer_norm1[l].parameters()
            parameters += self.model.ffns[l].parameters()
            parameters += self.model.layer_norm2[l].parameters()
            logger.info("Adding layer-%s parameters to optimizer" % (l + 1))

        logger.info("Optimizing on %i Transformer elements." % sum([p.nelement() for p in parameters]))

        return parameters

    def get_embeddings(self, x, lengths, positions=None, langs=None):
        """
        Inputs:
            `x`        : LongTensor of shape (slen, bs)
            `lengths`  : LongTensor of shape (bs,)
        Outputs:
            `sent_emb` : FloatTensor of shape (bs, out_dim)
        With out_dim == emb_dim
        """
        slen, bs = x.size()
        assert lengths.size(0) == bs and lengths.max().item() == slen

        # get transformer last hidden layer
        tensor = self.model('fwd', x=x, lengths=lengths, positions=positions, langs=langs, causal=False)
        assert tensor.size() == (slen, bs, self.out_dim)

        # single-vector sentence representation (first column of last layer)
        return tensor[0]
