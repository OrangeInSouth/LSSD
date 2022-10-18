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
from fairseq.modules.transformer_layer import TransformerEncoderLayerBase, TransformerDecoderLayerBase


class TransformerReorderEncoderLayer(TransformerEncoderLayerBase):
    """Encoder layer block equipped with reorder mechanism.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        # Added by Eachan
        print("********************Reorder Mechanism is Used in Encoder********************")
        self.reordering_embedding_layer = ReorderingLayer(cfg)
        # self.reordering_embedding_layer = ExReorderingLayer(cfg)

    def forward(
            self,
            x,
            encoder_padding_mask: Optional[Tensor],
            attn_mask: Optional[Tensor] = None,
            position_embedding=None
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
            position_embedding (Tensor): shape `(seq_len, batch, emb_dim)`


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
        # print("eachan print x:")
        # print(x.shape)
        # print("eachan print residual:")
        # print(residual.shape)
        # print("eachan print position_embedding:")
        # print(position_embedding.shape)
        reordering_embedding = self.reordering_embedding_layer(x, residual, position_embedding, encoder_padding_mask)
        # reordering_embedding = self.dropout_module(reordering_embedding)
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


class TransformerReorderDecoderLayer(TransformerDecoderLayerBase):
    """
    Decoder layer bock equipped with reorder mechanism.
    """

    def __init__(
            self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__(
            TransformerConfig.from_namespace(args),
            no_encoder_attn=no_encoder_attn,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        self.args = args

        # Added by Eachan
        print("********************Reorder Mechanism is Used in Decoder********************")
        self.reordering_embedding_layer = ReorderingLayer(args)

    def forward(
            self,
            x,
            encoder_out: Optional[torch.Tensor] = None,
            encoder_padding_mask: Optional[torch.Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            prev_self_attn_state: Optional[List[torch.Tensor]] = None,
            prev_attn_state: Optional[List[torch.Tensor]] = None,
            self_attn_mask: Optional[torch.Tensor] = None,
            self_attn_padding_mask: Optional[torch.Tensor] = None,
            need_attn: bool = False,
            need_head_weights: bool = False,
            position_embedding=None,
    ):
        """
        Re-implemented by Eachan

        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
                incremental_state is not None
                and _self_attn_input_buffer is not None
                and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            # Added by Eachan: insert a reordering embedding layer here.
            reordering_embedding = self.reordering_embedding_layer(x, residual, position_embedding)
            x = x + reordering_embedding

            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

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
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None


class TransformerLSReorderEncoderLayer(TransformerEncoderLayerBase):
    """Encoder layer block equipped with language-specific reorder mechanism.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        print("********************Language-Specific Reorder Mechanism is Used********************")
        # Added by Eachan
        self.LS_reordering_embedding_layer = LSReorderingLayer(cfg)

    def forward(
            self,
            x,
            encoder_padding_mask: Optional[Tensor],
            attn_mask: Optional[Tensor] = None,
            position_embedding=None,
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
        LS_reordering_embedding = self.LS_reordering_embedding_layer(x, residual, encoder_padding_mask,
                                                                     position_embedding=position_embedding,
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


class TransformerLSReorderDecoderLayer(TransformerDecoderLayerBase):
    """
        Decoder layer bock equipped with reorder mechanism.
        """

    def __init__(
            self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__(
            TransformerConfig.from_namespace(args),
            no_encoder_attn=no_encoder_attn,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        self.args = args

        # Added by Eachan
        print("********************Reorder Mechanism is Used in Decoder********************")
        self.LS_reordering_embedding_layer = LSReorderingLayer(args)

    def forward(
            self,
            x,
            encoder_out: Optional[torch.Tensor] = None,
            encoder_padding_mask: Optional[torch.Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            prev_self_attn_state: Optional[List[torch.Tensor]] = None,
            prev_attn_state: Optional[List[torch.Tensor]] = None,
            self_attn_mask: Optional[torch.Tensor] = None,
            self_attn_padding_mask: Optional[torch.Tensor] = None,
            need_attn: bool = False,
            need_head_weights: bool = False,
            position_embedding=None,
            langs=None
    ):
        """
        Re-implemented by Eachan

        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
                incremental_state is not None
                and _self_attn_input_buffer is not None
                and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            # Added by Eachan: insert a reordering embedding layer here.
            reordering_embedding = self.LS_reordering_embedding_layer(x, residual, position_embedding, langs)
            x = x + reordering_embedding

            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

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
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None


# class ReorderingLayer(nn.Module):
#     """
#     Implementation of reordering sub-layer.
#     Used to calculate the reordering embedding.
#     """
#
#     def __init__(self, args):
#         self.args = args
#         self.LA_reorder = getattr(args, "LA_reorder", False)
#         super().__init__()
#         self.embed_dim = args.encoder_embed_dim
#         self.reordering_V = nn.Linear(self.embed_dim, self.embed_dim)
#         self.reordering_W1 = nn.Linear(self.embed_dim, self.embed_dim)
#         self.reordering_W2 = nn.Linear(self.embed_dim, self.embed_dim)
#
#         # print("eachan print: LA_reorder is", self.LA_reorder)
#         if self.LA_reorder:
#             self.Lang_W = nn.Linear(self.embed_dim, self.embed_dim)
#
#         # 在考虑要不要加个二值化的gate
#         self.gate = 0
#         # if args.reorder_gate:
#         #     self.reordering_gate_W = nn.Linear(self.embed_dim, 1)
#
#     def forward(self, x, layer_input, position_embedding, encoder_mask):
#         """
#         The shape of 'x', 'layer_input', 'position_embedding' is all:
#             (seq_len, batch, embed_dim)
#
#         The shape of 'encoder_mask' is:
#             (batch, seq_len)
#         and the element pad will indicate '1'.
#         Note: !!! encoder_mask is None when no pad exists.
#
#         """
#
#         # 1. get real length of each sentence and language token index in each sentence.
#         seq_length, batch_size, dim = x.shape
#         sentence_length = (1 - encoder_mask).sum(dim=-1)  # shape: (batch)
#         langtok_index = seq_length - sentence_length  # (batch)
#
#         # 2. predict a position for each token.
#         predicted_position = torch.sigmoid(self.reordering_W(x)).squeeze(-1)  # shape: (seq_length, batch, 1)
#         predicted_position = predicted_position * sentence_length  # shape: (seq_length, batch)
#         print("DEBUG | predicted position:")
#         print(predicted_position.shape)
#
#         # 3. According to the predicted position, calculating the reordering embedding which is the weighted sum of
#         # sin/cos PE. These weights are negatively related to the distance between all possible positions to the
#         # predicted position.
#
#         # 3.1 calculating the distances between each possible positions and predicted position.
#         # 3.1.1 generating all possible positions.
#         all_positions = torch.range(0, seq_length - 1).unsqueeze(-1).repeat(1, batch_size)
#         all_positions.required_grad = False
#         all_positions = all_positions - langtok_index  # (seq_length, batch)
#
#         # 3.1.2 calculate the distances
#         distances = predicted_position.unsqueeze(-1).repeat(1, 1, seq_length) - \
#                     all_positions.unsqueeze(-1).repeat(1, 1, seq_length)
#         distances = distances ** 2  # (seq_length, batch, seq_length)
#
#         # 3.2 converting the distances into probabilities.
#         position_probability = distances ** -1
#         position_probability *= (1 - encoder_mask)  # deal with [PAD]
#         position_probability /= position_probability.sum(dim=-1).unsqueeze(-1)  # normalize
#
#         # 3.3 calculating the reordering embedding by aggregate all possible positions.
#         reordering_embedding = torch.bmm(
#             position_probability.transpose(0, 1),
#             position_embedding.transpose(0, 1)
#         ).transpose(0, 1)
#
#         return reordering_embedding

class ReorderingLayer(nn.Module):
    """
    Implementation of reordering sub-layer.
    Used to calculate the reordering embedding.
    """
    def __init__(self, args):
        self.args = args
        self.LA_reorder = getattr(args, "LA_reorder", False)
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.reordering_V = nn.Linear(self.embed_dim, self.embed_dim)
        self.reordering_W1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.reordering_W2 = nn.Linear(self.embed_dim, self.embed_dim)

        # print("eachan print: LA_reorder is", self.LA_reorder)
        if self.LA_reorder:
            self.Lang_W = nn.Linear(self.embed_dim, self.embed_dim)

        # 在考虑要不要加个二值化的gate
        self.gate = 0
        # if args.reorder_gate:
        #     self.reordering_gate_W = nn.Linear(self.embed_dim, 1)

    def forward(self, x, layer_input, position_embedding, encoder_mask):
        """
        The shape of 'x', 'layer_input', 'position_embedding' is all:
            (seq_len, batch, embed_dim)

        The shape of 'encoder_mask' is:
            (batch, seq_len)
        and the element pad will indicate '1'.
        Note: !!! encoder_mask is None when no pad exists.

        """
        # Eachan test: What will happen if we just add position embedding for each layer?
        # return position_embedding

        # according to 'encoder_mask', get the indices of langtok
        # reorder_mask = torch.zeros_like(x, requires_grad=False)
        #
        # if encoder_mask is None:
        #     reorder_mask[0] = 1
        # else:
        #     for i in range(encoder_mask.shape[0]):
        #         for j in range(encoder_mask.shape[1]):
        #             if encoder_mask[i][j] == 0:
        #                 reorder_mask[j][i] = 1  # langtok
        #                 reorder_mask[j + 1][i] = 1  # [BOS]
        #                 reorder_mask[-1][i] = 1  # [EOS]
        #                 break
        # print("eachan print | reorder_mask:", reorder_mask.mean(dim=-1))

        if self.LA_reorder:
            lang_embedding = layer_input[0].repeat(x.shape[0], 1, 1)
            position_penalty = torch.sigmoid(self.reordering_W1(layer_input) + self.reordering_W2(x)
                                             + self.Lang_W(lang_embedding))
            reordering_embedding = position_penalty * position_embedding
        else:
            position_penalty = torch.tanh(self.reordering_W1(layer_input) + self.reordering_W2(x))
            # 试着在这dropout
            position_penalty = torch.sigmoid(self.reordering_V(position_penalty))
            reordering_embedding = position_penalty * position_embedding

            # set the position penalty of langtok, [BOS] and [EOS] to 1, which indicate that
            # their reordering embeddings are equal to position embedding.
            # position_penalty = position_penalty * (1 - reorder_mask)
            # position_penalty = position_penalty + reorder_mask

        # print("eachan print | position_penalty:", position_penalty.mean(dim=[0, 2]))
        # reordering_embedding = position_penalty


        # if self.args.reorder_gate:
        #     # gate for reordering embedding:
        #     # 1. calculate the representation of sentence. (batch, hidden_state)
        #     sentence_embedding = x.mean(dim=0)
        #
        #     # 2. calculate reordering gate for sentence representation. (batch)
        #     gate = torch.sigmoid(self.reordering_gate_W(sentence_embedding))
        #     gate = gate.unsqueeze(0)
        #     gate = gate.repeat([x.shape[0], 1, 1])
        #     gate = gate.repeat([1, 1, x.shape[-1]])
        #
        #     # 3. apply the gate in reordering embedding
        #     # reordering_embedding = reordering_embedding * gate
        #     # 1. 手动设置每种语言的gate
        #     # 2. 将gate设为0.5
        #     # 3. 考虑句法、词性
        #     reordering_embedding = reordering_embedding * gate + position_embedding * (1 - gate)
        # reordering_embedding = reordering_embedding * self.gate + position_embedding * (1 - self.gate)

        return reordering_embedding

class ExReorderingLayer(nn.Module):
    """
    Implementation of explicit reordering sub-layer.
    Used to calculate the explicit reordering embedding.
    """

    def __init__(self, args):
        self.args = args
        self.LA_reorder = getattr(args, "LA_reorder", False)
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.reordering_W = nn.Linear(self.embed_dim, 1)

    def forward(self, x, layer_input, position_embedding, encoder_mask):
        """
        The shape of 'x', 'layer_input', 'position_embedding' is all:
            (seq_len, batch, embed_dim)

        The shape of 'encoder_mask' is:
            (batch, seq_len)
        and the element pad will indicate '1'.
        Note: !!! encoder_mask is None when no pad exists.

        """
        seq_length, batch_size, dim = x.shape
        if encoder_mask is None:
            encoder_mask = torch.zeros([batch_size, seq_length], device=x.device, requires_grad=False).bool()

        # 1. get real length of each sentence and language token index in each sentence.
        sentence_length = (~encoder_mask).sum(dim=-1)  # shape: (batch)
        langtok_index = seq_length - sentence_length  # (batch)

        # 2. predict a position for each token.
        predicted_position = torch.sigmoid(self.reordering_W(x)).squeeze(-1)  # shape: (seq_length, batch, 1)
        # print("DEBUG | predicted position representation:")
        # print(predicted_position)
        predicted_position = predicted_position * sentence_length  # shape: (seq_length, batch)
        # print("DEBUG | sentence_length:")
        # print(sentence_length)
        print("DEBUG | predicted position:")
        print(predicted_position)

        # 3. According to the predicted position, calculating the reordering embedding which is the weighted sum of
        # sin/cos PE. These weights are negatively related to the distance between all possible positions to the
        # predicted position.

        # 3.1 calculating the distances between each possible positions and predicted position.
        # 3.1.1 generating all possible positions.
        all_positions = torch.range(0, seq_length - 1, device=x.device, requires_grad=False).half().\
            unsqueeze(-1).repeat(1, batch_size)
        all_positions = all_positions - langtok_index  # (seq_length, batch)

        # 3.1.2 calculate the distances
        distances = predicted_position.unsqueeze(-1).repeat(1, 1, seq_length) - \
                    all_positions.repeat(seq_length, 1).reshape([seq_length, seq_length, batch_size]).transpose(1, 2)
        distances = distances ** 2  # (seq_length, batch, seq_length)

        # 3.2 converting the distances into probabilities.
        position_probability = (distances + 0.001) ** -1
        position_probability *= (~encoder_mask)  # deal with [PAD]
        position_probability /= position_probability.sum(dim=-1).unsqueeze(-1)  # normalize

        # 3.3 calculating the reordering embedding by aggregate all possible positions.
        reordering_embedding = torch.bmm(
            position_probability.transpose(0, 1),
            position_embedding.transpose(0, 1)
        ).transpose(0, 1)

        # 3.4 mask reordering embedding for PAD
        reordering_embedding *= (~encoder_mask).transpose(0, 1).unsqueeze(-1).repeat(1, 1, dim)

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

        # create language-family-specific reordering layer
        # lang_family_module_map = {}
        # lang_family_module_map["family-1"] = ReorderingLayer(args)
        # lang_family_module_map["family-2"] = ReorderingLayer(args)
        # lang_family_module_map["family-3"] = ReorderingLayer(args)
        # lang_family_module_map["family-4"] = ReorderingLayer(args)

        # create a reordering_embedding for each language pair
        self.reordering_embedding_dict = nn.ModuleDict()
        # print(args)
        for langpair in args.lang_pairs.split(','):
            if args.encoder_langtok == "src":
                lang = langpair.split("-")[0]
            elif args.encoder_langtok == "tgt":
                lang = langpair.split("-")[1]

            lang_token = '__{}__'.format(lang)
            lang_id = args.lang2id[lang_token]

            # 我这里简单点弄（我累了，应该不是UNK
            assert lang_id != 3
            self.reordering_embedding_dict[str(lang_id)] = ReorderingLayer(args)
            # lang_family = self.get_lang_family(lang)
            # self.reordering_embedding_dict[str(lang_id)] = lang_family_module_map[lang_family]
            # self.reordering_embedding_dict[str(lang_id)].gate = self.get_language_specific_reordering_gate(lang)

    def get_language_specific_reordering_gate(self, lang):
        """
        Set reordering gate for each language.
        This gate determine to use whether reordering embedding or positional embedding.
        """
        low_resources = ['bos', 'mar', 'hin', 'mkd']
        high_resources = ['ell', 'bul', 'fra', 'kor']
        if lang in low_resources:
            return 1
        elif lang in high_resources:
            return 0
        else:
            return None

    def get_lang_family(self, lang):
        """
        map language to language family.
        """
        lang_family_map = {
            "aze": "family-1",
            "bel": "family-1",
            "glg": "family-2",
            "slk": "family-2",
            "tur": "family-3",
            "rus": "family-3",
            "por": "family-4",
            "ces": "family-4",
        }
        return lang_family_map[lang]

    def forward(self, x, layer_input, encoder_padding_mask, position_embedding=None, langs=None):
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
        # print("eachan print: shape of position_embedding", position_embedding.shape)
        assert len(langs.unique()) == 1

        if len(langs.unique()) == 1:
            lang_id = str(langs[0].item())
            all_reordering_embeddings = self.reordering_embedding_dict[lang_id](x, layer_input,
                                                                                position_embedding,
                                                                                encoder_padding_mask)
        else:  # for mix-batch
            all_reordering_embeddings = []
            for i in range(len(langs)):
                lang_id = str(langs[i].item())
                tmp_x = x[:, i, :].unsqueeze(1)
                tmp_layer_input = layer_input[:, i, :].unsqueeze(1)
                tmp_position_embedding = position_embedding[:, i, :].unsqueeze(1)
                all_reordering_embeddings.append(self.reordering_embedding_dict[lang_id](tmp_x,
                                                                                         tmp_layer_input,
                                                                                         tmp_position_embedding))

            # 3. stack all reordering embeddings to return.
            all_reordering_embeddings = torch.stack(all_reordering_embeddings, dim=1).squeeze(2)

        assert all_reordering_embeddings.shape == x.shape
        return all_reordering_embeddings
