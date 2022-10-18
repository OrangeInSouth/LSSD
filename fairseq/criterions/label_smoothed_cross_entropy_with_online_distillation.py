# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import logging
import pdb
from fairseq import metrics, utils
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask
from fairseq.criterions import register_criterion

from .label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig,
)

from dataclasses import dataclass, field

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@dataclass
class LabelSmoothedCrossEntropyCriterionWithOnlineDistillationConfig(
    LabelSmoothedCrossEntropyCriterionConfig
):
    online_distillation_weight: float = field(
        default=1, metadata={"help": "weight for the online distillation loss"}
    )
    selective_online_distillation: str = field(
        default="none",
        metadata={"help": "choice: (hard, soft, top5, none)"},
    )
    selective_online_distillation_level: str = field(
        default="token",
        metadata={"help": "choice: (token, sentence, batch)"}
    )
    language_aware_online_distillation: bool = field(
        default=False, metadata={"help": "whether to use Language-Aware Online Distillation"}
    )


@register_criterion(
    "label_smoothed_cross_entropy_with_online_distillation",
    dataclass=LabelSmoothedCrossEntropyCriterionWithOnlineDistillationConfig,
)
class LabelSmoothedCrossEntropyCriterionWithOnlineDistillation(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(self, task, sentence_avg, label_smoothing, online_distillation_weight, selective_online_distillation,
                 language_aware_online_distillation,
                 selective_online_distillation_level
                 ):
        super().__init__(task, sentence_avg, label_smoothing)
        self.online_distillation_weight = online_distillation_weight
        self.selective_online_distillation = selective_online_distillation
        self.language_aware_online_distillation = language_aware_online_distillation
        self.selective_online_distillation_level = selective_online_distillation_level
        logger.debug(f"online_distillation_weight: {self.online_distillation_weight}")
        logger.debug(f"selective_online_distillation: {self.selective_online_distillation}")
        logger.debug(f"language_aware_online_distillation: {self.language_aware_online_distillation}")
        logger.debug(f"selective_online_distillation_level: {self.selective_online_distillation_level}")

        # only distilling target words in one language each time.
        if task.MNMT_mode == "O2M" and language_aware_online_distillation:
            language_aware_target_mask = {}
            for lang in task.target_langs:
                # 1. loading language-specific dicts
                LS_dict = TranslationMultiSimpleEpochTask.load_dictionary(f"{task.args.data}/LS_dict.{lang}.txt")
                LS_dict.add_symbol(f"__{lang}__")

                # 2. constructing a mask vector to zero the probs in other languages.
                target_mask = []
                tgt_dict = task.dicts[lang]
                for i in range(len(tgt_dict)):
                    target_mask.append(tgt_dict[i] in LS_dict)
                target_mask = torch.tensor(target_mask, requires_grad=False)
                language_aware_target_mask[lang] = target_mask
            self.language_aware_target_mask = language_aware_target_mask

    def forward(self, student, sample, teacher=None, need_online_distillation=False, reduce=True, epoch=-1):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        4) if need online distillation and sample is not DUMMY, calculating Online_Distillation_loss
            and adding into logging_output
        """
        student_output = student(**sample["net_input"])
        loss, nll_loss = self.compute_loss(student, student_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(student, student_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)

        # Compute alignment loss only for training set and non dummy batches.
        if need_online_distillation and sample != "DUMMY":
            with torch.no_grad():
                teacher_output = teacher(**sample["net_input"])
            online_distillation_loss = self.compute_online_distillation_loss(student, student_output,
                                                                             teacher, teacher_output,
                                                                             sample,
                                                                             reduce,
                                                                             epoch=epoch)

            # logging_output["online_distillation_loss"] = utils.item(online_distillation_loss.data)

            # logger.debug(f"online_distillation_weight: {self.online_distillation_weight}")
            loss = (loss + self.online_distillation_weight * online_distillation_loss) / 2

        return loss, sample_size, logging_output

    def compute_online_distillation_loss(self, student, student_output, teacher, teacher_output, sample, reduce=True,
                                         epoch=-1):
        # 1. get lprobs of student_output and prob of teacher_output and target
        student_lprobs, target = self.get_lprobs_and_target(student, student_output, sample)
        # logger.debug(f"student_lprobs.shape: {student_lprobs.shape}")  # (batch_size, vacab_size)
        # logger.debug(f"target.shape: {target.shape}")                  # (batch_size)
        teacher_probs = self.get_probs(teacher, teacher_output)
        # logger.debug(f"teacher_probs.shape: {teacher_probs.shape}")    # (batch_size, vocab_size)

        # Insert: language-aware distillation
        if self.language_aware_online_distillation:
            batch_lang = self.task.get_batch_lang(sample['net_input']['src_tokens'])
            if self.language_aware_target_mask[batch_lang].device != teacher_probs.device:
                self.language_aware_target_mask[batch_lang] = self.language_aware_target_mask[batch_lang].to(teacher_probs.device)
            teacher_probs = teacher_probs * self.language_aware_target_mask[batch_lang]
            teacher_probs = teacher_probs / teacher_probs.sum(dim=-1).unsqueeze(dim=-1)

        # Insert: top-K distillation
        if len(self.selective_online_distillation) > 3 and self.selective_online_distillation[:3] == "top":
            K = int(self.selective_online_distillation[3:])
            value_top_K_teacher_probs, index_top_K_teacher_probs = teacher_probs.topk(K, dim=-1)  # (abtch_size, K)
            # value_top_K_teacher_probs = value_top_K_teacher_probs.softmax(dim=-1)
            filtered_student_lprobs = student_lprobs.gather(dim=-1, index=index_top_K_teacher_probs)
            loss = -value_top_K_teacher_probs * filtered_student_lprobs
        else:
            # 2. compute cross entropy between teacher_output and student_output
            loss = -teacher_probs * student_lprobs  # (batch_size, vocab_size)
        loss = loss.sum(dim=-1)  # (batch_size)

        # 3. selective online distillation
        if self.selective_online_distillation != "none":

            if self.selective_online_distillation == "hard":
                # 3.1.1 calculate the probability over target word for teacher and student,
                student_probs = student_lprobs.exp()
                online_distillation_gate = self.calculate_hard_gate(student_probs, teacher_probs, target, sample, epoch)
                loss *= online_distillation_gate
            elif self.selective_online_distillation == "soft":
                student_probs = student_lprobs.exp()
                online_distillation_gate = self.calculate_soft_gate(student_probs, teacher_probs, target, sample, epoch)
                loss *= online_distillation_gate

            elif len(self.selective_online_distillation) > 3 and self.selective_online_distillation[:3] == "top":
                pass
            else:
                logger.debug(f"self.selective_online_distillation: {self.selective_online_distillation}")
                raise Exception("for selective-online-distillation, only 'hard', 'soft', 'none' is allowed")

        if self.padding_idx is not None:
            pad_mask = target.eq(self.padding_idx)
            loss.masked_fill_(pad_mask, 0.0)
        else:
            loss = loss.squeeze(-1)
        if reduce:
            loss = loss.sum()

        return loss

    def get_probs(self, model, net_output):
        probs = model.get_normalized_probs(net_output, log_probs=False)
        if self.ignore_prefix_size > 0:
            if getattr(probs, "batch_first", False):
                probs = probs[:, self.ignore_prefix_size :, :].contiguous()
            else:
                probs = probs[self.ignore_prefix_size :, :, :].contiguous()
        return probs.view(-1, probs.size(-1))

    def calculate_hard_gate(self, student_probs, teacher_probs, target, sample, epoch):
        student_probs_on_target = student_probs.gather(dim=-1, index=target.unsqueeze(-1)).detach()  # (batch_size, 1)
        teacher_probs_on_target = teacher_probs.gather(dim=-1, index=target.unsqueeze(-1)).detach()  # (batch_size, 1)

        if self.padding_idx is not None:
            pad_mask = target.eq(self.padding_idx).unsqueeze(dim=-1)
            student_probs_on_target.masked_fill_(pad_mask, 0.0)
            teacher_probs_on_target.masked_fill_(pad_mask, 0.0)

        # convert token-level gate into sentence-level gate if specified.
        if self.selective_online_distillation_level == "sentence":
            # (1) get seq_len
            batch_size = sample['net_input']['src_tokens'].shape[0]
            seq_len = int(target.shape[0] / batch_size)
            # (2) reshape student_probs_on_target and teacher_probs_on_target as (batch_size, seq_len)
            student_probs_on_target = student_probs_on_target.reshape(batch_size, seq_len)  # (batch_size, seq_len)
            teacher_probs_on_target = teacher_probs_on_target.reshape(batch_size, seq_len)  # (batch_size, seq_len)
            # (3) aggregate into sentence-level confidence (batch_size, 1). Be careful of [PAD]
            student_probs_on_target = student_probs_on_target.sum(dim=-1)  # (batch_size)
            teacher_probs_on_target = teacher_probs_on_target.sum(dim=-1)  # (batch_size)
            # (4) calculate sentence-level distillation gate.
            sentence_gate = teacher_probs_on_target > student_probs_on_target  # (batch_size)
            # (5) convert sentence-level gate back into consistent token gate.
            token_gate = sentence_gate.unsqueeze(dim=-1).repeat(1, seq_len)  # (batch_size * seq_len)
            token_gate = token_gate.reshape(batch_size * seq_len, 1)

            # print sentence gate
            print(f"Sentence-level Hard Selective Online Distillation | {epoch},"
                  f"{self.task.get_batch_lang(sample['net_input']['src_tokens'])},"
                  f"{sentence_gate.sum().item()},"
                  f"{batch_size}")
        elif self.selective_online_distillation_level == "token":
            token_gate = teacher_probs_on_target > student_probs_on_target  # (batch_size * seq_len, 1)
            # print sentence gate
            print(f"Token-level Hard Selective Online Distillation | {epoch},"
                  f"{self.task.get_batch_lang(sample['net_input']['src_tokens'])},"
                  f"{token_gate.sum().item()},"
                  f"{sample['ntokens']}")
            assert token_gate.sum().item() <= sample['ntokens']
        else:
            raise Exception("for selective-online-distillation-level, only 'sentence', 'token' is allowed")

        # 3.1.3 We set the weight to 0.2 times of vanilla distillation if the student is better than teacher.
        online_distillation_gate = token_gate + (~token_gate) * 0.2  # (batch_size * seq_len, 1)
        online_distillation_gate = online_distillation_gate.squeeze(dim=-1)  # (batch_size * seq_len)
        return online_distillation_gate

    def calculate_soft_gate(self, student_probs, teacher_probs, target, sample, epoch):
        student_probs_on_target = student_probs.gather(dim=-1, index=target.unsqueeze(-1)).detach()  # (batch_size, 1)
        teacher_probs_on_target = teacher_probs.gather(dim=-1, index=target.unsqueeze(-1)).detach()  # (batch_size, 1)

        if self.selective_online_distillation_level == "sentence":
            if self.padding_idx is not None:
                pad_mask = target.eq(self.padding_idx).unsqueeze(dim=-1)
                student_probs_on_target.masked_fill_(pad_mask, 0.0)
                teacher_probs_on_target.masked_fill_(pad_mask, 0.0)

            # (1) get seq_len
            batch_size = sample['net_input']['src_tokens'].shape[0]
            seq_len = int(target.shape[0] / batch_size)  # 要注意是target端的seq_len
            # (2) reshape student_probs_on_target and teacher_probs_on_target as (batch_size, seq_len)
            student_probs_on_target = student_probs_on_target.reshape(batch_size, seq_len)  # (batch_size, seq_len)
            teacher_probs_on_target = teacher_probs_on_target.reshape(batch_size, seq_len)  # (batch_size, seq_len)
            # (3) aggregate into sentence-level confidence (batch_size, 1). Be careful of [PAD]
            student_probs_on_target = student_probs_on_target.sum(dim=-1)  # (batch_size)
            teacher_probs_on_target = teacher_probs_on_target.sum(dim=-1)  # (batch_size)
            pad_mask = target.eq(self.padding_idx).reshape(batch_size, seq_len)
            target_seq_len = (~pad_mask).sum(dim=-1)
            student_probs_on_target /= target_seq_len
            teacher_probs_on_target /= target_seq_len
            # (4) calculate sentence-level distillation gate.
            sentence_gate = teacher_probs_on_target / student_probs_on_target  # (batch_size)
            # (5) convert sentence-level gate back into consistent token gate.
            token_gate = sentence_gate.unsqueeze(dim=-1).repeat(1, seq_len)  # (batch_size * seq_len)
            token_gate = token_gate.reshape(batch_size * seq_len, 1)

            # print sentence gate
            if sentence_gate.device.index == 0:
                print(f"Sentence-level Soft Selective Online Distillation | {epoch},"
                      f"{self.task.get_batch_lang(sample['net_input']['src_tokens'])},"
                      f"{sentence_gate.mean().item()},"
                      f"{teacher_probs_on_target.mean().item()},"
                      f"{student_probs_on_target.mean().item()}")
        elif self.selective_online_distillation_level == "token":
            if self.padding_idx is not None:
                pad_mask = target.eq(self.padding_idx).unsqueeze(dim=-1)
                student_probs_on_target.masked_fill_(pad_mask, 0.0)
                teacher_probs_on_target.masked_fill_(pad_mask, 0.01)  # Note!!! This line is different with sentence-level selective online distillation

            token_gate = teacher_probs_on_target / student_probs_on_target  # (batch_size * seq_len, 1)
            print(f"Token-level Soft Selective Online Distillation | {epoch},"
                  f"{self.task.get_batch_lang(sample['net_input']['src_tokens'])},"
                  f"{(token_gate.sum() / sample['ntokens']).item()}")
        else:
            raise Exception("for selective-online-distillation-level, only 'sentence', 'token' is allowed")

        # normalize
        # online_distillation_gate = 2 / (1 + (1 - token_gate).exp())  # I found this is irrational
        online_distillation_gate = torch.min(token_gate, torch.tensor([1.], device=token_gate.device))  # trunc
        online_distillation_gate = online_distillation_gate.squeeze(dim=-1)

        return online_distillation_gate
    #
    # @staticmethod
    # def reduce_metrics(logging_outputs) -> None:
    #     """Aggregate logging outputs from data parallel training."""
    #     loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
    #     nll_loss_sum = utils.item(
    #         sum(log.get("nll_loss", 0) for log in logging_outputs)
    #     )
    #     alignment_loss_sum = utils.item(
    #         sum(log.get("alignment_loss", 0) for log in logging_outputs)
    #     )
    #     ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
    #     sample_size = utils.item(
    #         sum(log.get("sample_size", 0) for log in logging_outputs)
    #     )
    #
    #     metrics.log_scalar(
    #         "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
    #     )
    #     metrics.log_scalar(
    #         "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
    #     )
    #     metrics.log_scalar(
    #         "alignment_loss",
    #         alignment_loss_sum / sample_size / math.log(2),
    #         sample_size,
    #         round=3,
    #     )
    #     metrics.log_derived(
    #         "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
    #     )
    #
    # @staticmethod
    # def logging_outputs_can_be_summed() -> bool:
    #     """
    #     Whether the logging outputs returned by `forward` can be summed
    #     across workers prior to calling `reduce_metrics`. Setting this
    #     to True will improves distributed training speed.
    #     """
    #     return True
