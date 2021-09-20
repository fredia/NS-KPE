import math
import torch
import logging
import traceback
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import CrossEntropyLoss
from ..transformers import BertPreTrainedModel, RobertaModel

from itertools import repeat
from torch._six import container_abcs

logger = logging.getLogger()


# -------------------------------------------------------------------------------------------
# CnnGram Extractor
# -------------------------------------------------------------------------------------------

class CNNGramer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.cnn_1 = nn.Conv1d(in_channels=input_size,
                               out_channels=hidden_size,
                               kernel_size=1)

        self.cnn_2 = nn.Conv1d(in_channels=input_size,
                               out_channels=hidden_size,
                               kernel_size=2)

        self.cnn_3 = nn.Conv1d(in_channels=input_size,
                               out_channels=hidden_size,
                               kernel_size=3)

        self.cnn_4 = nn.Conv1d(in_channels=input_size,
                               out_channels=hidden_size,
                               kernel_size=4)

        self.cnn_5 = nn.Conv1d(in_channels=input_size,
                               out_channels=hidden_size,
                               kernel_size=5)

    def forward(self, x):
        x = x.transpose(1, 2)

        gram_1 = self.cnn_1(x)
        gram_2 = self.cnn_2(x)
        gram_3 = self.cnn_3(x)
        gram_4 = self.cnn_4(x)
        gram_5 = self.cnn_5(x)

        outputs = torch.cat([gram_1.transpose(1, 2), gram_2.transpose(1, 2),
                             gram_3.transpose(1, 2), gram_4.transpose(1, 2), gram_5.transpose(1, 2)], 1)
        return outputs


# -------------------------------------------------------------------------------------------
# Inherit BertPreTrainedModel
# -------------------------------------------------------------------------------------------
class RobertaForCnnGramClassification7(BertPreTrainedModel):

    def __init__(self, config):
        super(RobertaForCnnGramClassification7, self).__init__(config)

        cnn_output_size = 512

        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.sampling_rate = config.sampling_rate
        self.mask_rate = config.mask_rate

        self.cnn2gram = CNNGramer(input_size=config.hidden_size, hidden_size=cnn_output_size)

        self.classifier = nn.Linear(cnn_output_size, config.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()


# -------------------------------------------------------------------------------------------
# RobertaForCnnGramExtractor
# -------------------------------------------------------------------------------------------

class RobertaForCnnGramExtractor7(RobertaForCnnGramClassification7):

    def forward(self, input_ids, attention_mask, valid_ids, active_mask, valid_output, labels=None):

        # --------------------------------------------------------------------------------
        # Bert Embedding Outputs
        outputs = self.roberta(input_ids=input_ids,
                               attention_mask=attention_mask)

        sequence_output = outputs[0]

        # --------------------------------------------------------------------------------
        # Valid Outputs : get first token vector
        batch_size = sequence_output.size(0)
        for i in range(batch_size):
            valid_num = sum(valid_ids[i]).item()

            vectors = sequence_output[i][valid_ids[i] == 1]
            valid_output[i, :valid_num].copy_(vectors)

        # --------------------------------------------------------------------------------
        # Dropout
        sequence_output = self.dropout(valid_output)

        # --------------------------------------------------------------------------------
        # CNN Outputs
        cnn_outputs = self.cnn2gram(sequence_output)  # shape = (batch_size, max_gram_num, 512)

        # classifier (512, 2)
        logits = self.classifier(cnn_outputs)

        if labels is not None:
            # 负采样
            # 把label为1的位置的gram全都取出来作为被标记了的样本(label=1 activate_mask一定为1)
            # 把label为0且activate_mask为1的取出来作为 未被标记的样本
            # mask: 把部分标注样本当做负例
            device = torch.device("cuda", cnn_outputs.get_device())
            max_gram_num = cnn_outputs.size(1)
            expand_label = labels.unsqueeze(-1).expand(batch_size, max_gram_num, 2)
            expand_activate_mask = active_mask.unsqueeze(-1).expand(batch_size, max_gram_num, 2)
            # print('expand label', expand_label.size())
            positive_mask = (expand_label == 1)
            negative_mask = (expand_label == 0) & (expand_activate_mask == 1)
            # print('nega mask', negative_mask.size())
            positive_logits = torch.masked_select(logits, positive_mask).view([-1, 2])
            positive_num = positive_logits.size(0)
            # mask掉部分标注样本
            positive_perm = torch.randperm(positive_num).to(device)
            remain_num = int(np.ceil((1 - self.mask_rate) * positive_num))
            remain_positivate_logits = torch.index_select(positive_logits, 0, positive_perm[:remain_num])
            mask_positivate_logits = torch.index_select(positive_logits, 0, positive_perm[remain_num:])
            negative_logits = torch.masked_select(logits, negative_mask).view([-1, 2])
            merge_negative_logits = torch.cat([negative_logits, mask_positivate_logits], dim=0)
            negative_num = merge_negative_logits.size(0)

            merge_logits = torch.cat([remain_positivate_logits, merge_negative_logits], dim=0).view(-1, self.num_labels)
            merge_lables = torch.cat([labels.data.new_ones([remain_num]), labels.data.new_zeros([negative_num])],
                                     dim=0)

            loss_fct = CrossEntropyLoss(reduction='mean')
            loss = loss_fct(merge_logits, merge_lables)
            return loss
        else:
            active_loss = active_mask.view(-1) == 1  # [False, True, ...] # batch_size * max_len
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            return active_logits
