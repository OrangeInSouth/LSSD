
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor
from fairseq.models.transformer import (
    TransformerConfig,
)
from fairseq.modules.transformer_layer import TransformerEncoderLayerBase


class TransformerReorderEncoderLayer(TransformerEncoderLayerBase):
    """Encoder layer block equipped with reorder mechanism.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, cfg):
        super().__init__()

        # Added by Eachan
        self.reordering_embedding_layer = ReorderingLayer(args)


    def forward(
        self,
        x,
        encoder_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor] = None,
        position_embedding = None
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
        )

        # Added by Eachan: calculate the reordering embedding
        reordering_embedding = self.reordering_embedding_layer(x, residual, position_embedding)
        x = x + reordering_embedding

        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


class TransformerLSReorderEncoderLayer(TransformerEncoderLayerBase):
    """Encoder layer block equipped with language-specific reorder mechanism.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, cfg):
        super().__init__()

        # Added by Eachan
        self.LS_reordering_embedding_layer = LSReorderingLayer(args)

    def forward(
            self,
            x,
            encoder_padding_mask: Optional[Tensor],
            attn_mask: Optional[Tensor] = None,
            position_embedding = None,
            langs=None
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
        )

        # Added by Eachan: calculate the language-specific reordering embedding
        LS_reordering_embedding = self.LS_reordering_embedding_layer(x, residual, position_embedding=position_embedding,
                                                                     langs=langs)
        x = x + LS_reordering_embedding

        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


class ReorderingLayer(nn.Module):
    """
    Implementation of reordering sub-layer.
    Used to calculate the reordering embedding.
    """
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.reordering_V = nn.Linear(self.embed_dim, self.embed_dim)
        self.reordering_W1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.reordering_W2 = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x, layer_input, position_embedding):
        # Eachan: calculate the reordering embedding
        # print("eachan print:==================================")
        # print("residual shape:", residual.shape)
        # print("x shape:", x.shape)
        # print("position embedding shape", position_embedding.shape)
        position_penalty = torch.tanh(self.reordering_W1(layer_input) + self.reordering_W2(x))
        position_penalty = torch.sigmoid(self.reordering_V(position_penalty))
        # print("position penalty shape:", position_penalty.shape)
        reordering_embedding = position_penalty * position_embedding
        # print("reordering embedding shape:", reordering_embedding.shape)
        return reordering_embedding


class LSReorderingLayer(nn.Module):
    """
    (x)Implementation of language-pair-specific reordering sub-layer.
    Implementation of language-specific reordering sub-layer, i.e., build a reordering embedding sub-layer
    for each source language (in M2O) or target language (in O2M)

    args:
    lang2id: a dict used to map the lang token to id in the dict
    """
    def __init__(self, args):
        super().__init__()

        # get all IDs of src tokens and target tokens in the dict
        # print("eachan print lang_pairs:", args.lang_pairs)
        self.args = args

        # create a reordering_embedding for each language pair
        self.reordering_embedding_dict = nn.ModuleDict()
        for langpair in args.lang_pairs:
            if args.encoder_langtok == "src":
                lang_token = langpair.split("-")[0]
            elif args.encoder_langtok == "tgt":
                lang_token = langpair.split("-")[1]

            lang_token = '__{}__'.format(lang_token)
            lang_id = args.lang2id[lang_token]

            # 我这里简单点弄（我累了
            assert lang_id != 3
            self.reordering_embedding_dict[str(lang_id)] = ReorderingLayer(args)

    def forward(self, x, layer_input, position_embedding=None, langs=None):
        """
        x: (Length, Batch, Hidden_state)
        layer_input: (Length, Batch, Hidden_state)
        src_token: (Length, Batch)
        position_embedding: don't care, just parse to reordering_embedding
        """
        # 1. get the language of each sentence according to src_tokens.
        # Now I implemented this in TransformerReorderingEncoder
        # print("eachan print langs")
        # print(langs)
        #
        # print("eachan pring lang2id", self.args.lang2id)

        # # 2. calculate the reordering embedding for each sentence.
        # print("====================================================")
        # print("eachan print: shape of x", x.shape)
        # print("eachan print: shape of langs", langs.shape)
        assert len(langs.unique()) == 1

        if len(langs.unique()) == 1:
            lang_id = str(langs[0].item())
            all_reordering_embeddings = self.reordering_embedding_dict[lang_id](x, layer_input,
                                                                                        position_embedding)
        else:
            all_reordering_embeddings = []
            for i in range(len(langs)):
                lang_id = str(langs[i].item())
                all_reordering_embeddings.append(self.reordering_embedding_dict[lang_id](x[:,i,:].unsqueeze(1),
                                                                                          layer_input[:,i,:].unsqueeze(1),
                                                                                          position_embedding[:,i,:].unsqueeze(1)))

            # 3. stack all reordering embeddings to return.
            all_reordering_embeddings = torch.stack(all_reordering_embeddings, dim=1).squeeze(2)

        assert all_reordering_embeddings.shape == x.shape
        return all_reordering_embeddings