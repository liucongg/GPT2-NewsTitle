# -*- coding:utf-8 -*-
# @project: GPT2-NewsTitle
# @filename: model.py
# @author: 刘聪NLP
# @contact: logcongcong@gmail.com
# @time: 2020/12/16 16:26
"""
    文件说明：
    GPT2模型文件，主要对transformers包中GPT2LMHeadModel的重写，修改计算loss部分，只计算预测title部分的loss
"""

from torch.nn import CrossEntropyLoss
import torch.nn as nn
from transformers.modeling_gpt2 import GPT2PreTrainedModel, GPT2Model


class GPT2LMHeadModel(GPT2PreTrainedModel):
    """GPT2模型"""
    def __init__(self, config):
        """
        初始化函数
        Args:
            config: 配置参数
        """
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.init_weights()

    def forward(self, input_ids=None, past=None, token_type_ids=None, labels=None, title_id=None):
        """
        前向函数，计算GPT2预测结果值
        Args:
            input_ids: 输入序列在词表中的索引序列，size:[batch_size, sequence_length]
            past: 包含由模型预先计算好的隐藏状态，一般使用在预测阶段，用于加速顺序解码，防止重复计算前面计算过的token
            token_type_ids: 用于区分输入序列中content和title的分隔符序列，size:[batch_size, sequence_length]
            labels: 标签序列，size:[batch_size, sequence_length]，一般情况下，与input_ids相同
            title_id: title部分分隔符的id
        Returns:

        """
        # 获取GPT2模型的输出结果
        transformer_outputs = self.transformer(input_ids, past=past, token_type_ids=token_type_ids)
        # 获取GPT2模型的最后一层的隐层节点状态，size:[batch_size, sequence_length, config.n_embd]
        hidden_states = transformer_outputs[0]
        # 预测隐层节点状态中的每一个token的下一个token，size:[batch_size, sequence_length, config.vocab_size]
        lm_logits = self.lm_head(hidden_states)
        # 拼接输出结果
        outputs = (lm_logits,) + transformer_outputs[1:]
        # 如果labels不为None时，计算损失值loss，并拼接到输出结果中
        if labels is not None:
            # 计算loss时，title_id不可以为None，因为需要title_id找到title的部分
            if title_id is None or token_type_ids is None:
                raise Exception("当labels不为None时， title_id和token_type_ids均不可以为None。")
            # 获取mask值，如果token_type_ids中等于title_id的部分需要计算loss，标记为1；否则为0。
            # size:[batch_size, sequence_length]
            mask = (token_type_ids == title_id).long()
            # 获取新的标签，size:[batch_size, sequence_length]
            labels = labels * mask
            # 对预测结果和标签进行偏移操作
            # GPT2的生成机制为通过前面的token，预测下一个token；并且labels与input_ids相同，
            # 因此input_ids中的第一个token的预测结果，实际上是标签中的第二个token，以此类推，最终仅计算sequence_length-1个token的loss
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # 定义损失函数CrossEntropyLoss，并且设置忽略计算loss的索引，以及返回loss的形式
            # 忽略shift_labels中为0的loss，也就是仅计算title部分的损失值
            # 对loss的计算方式设为sum，由于我们仅计算了itle部分的损失值，如果使用mean，会使loss变小（实际除的是sequence_length-1，不是title部分的真实长度）
            loss_fct = CrossEntropyLoss(ignore_index=0, reduction="sum")
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            # 获取title部分的真实长度，并计算真实loss
            num = shift_labels.ne(0).long().sum().item()
            loss = loss / num
            outputs = (loss,) + outputs
        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)

