# -*- coding:utf-8 -*-
# @project: GPT2-NewsTitle
# @filename: generate_title.py
# @author: 刘聪NLP
# @contact: logcongcong@gmail.com
# @time: 2020/12/16 16:29
"""
    文件说明：
    根据训练好的模型，进行新闻标题生成，预测文件
"""

import torch
import os
import argparse
from model import GPT2LMHeadModel
from transformers import BertTokenizer
import torch.nn.functional as F
import copy


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, help='设置预测时使用的显卡')
    parser.add_argument('--output_dir', default='output_dir/', type=str, help='模型输出路径')
    parser.add_argument('--vocab_path', default='vocab/vocab.txt', type=str, help='词表，该词表为小词表，并增加了一些新的标记')
    parser.add_argument('--batch_size', default=5, type=int, help='生成标题的个数')
    parser.add_argument('--generate_max_len', default=32, type=int, help='生成标题的最大长度')
    parser.add_argument('--repetition_penalty', default=1.2, type=float, help='重复处罚率')
    parser.add_argument('--top_k', default=3, type=float, help='解码时保留概率最高的多少个标记')
    parser.add_argument('--top_p', default=0.999, type=float, help='解码时保留概率累加大于多少的标记')
    parser.add_argument('--max_len', type=int, default=512, help='输入模型的最大长度，要比config中n_ctx小')
    return parser.parse_args()


def top_k_top_p_filtering(logits, top_k, top_p, filter_value=-float("Inf")):
    assert logits.dim() == 2
    top_k = min(top_k, logits[0].size(-1))
    if top_k > 0:
        for logit in logits:
            indices_to_remove = logit < torch.topk(logit, top_k)[0][..., -1, None]
            logit[indices_to_remove] = filter_value
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)  # 对logits进行递减排序
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for index, logit in enumerate(logits):
            indices_to_remove = sorted_indices[index][sorted_indices_to_remove[index]]
            logit[indices_to_remove] = filter_value
    return logits


def predict_one_sample(model, tokenizer, device, args, content):
    content_tokens = tokenizer.tokenize(content)
    if len(content_tokens) > args.max_len - 3 - args.generate_max_len:
        content_tokens = content_tokens[:args.max_len - 3 - args.generate_max_len]
    content_id = tokenizer.convert_tokens_to_ids("[Content]")
    title_id = tokenizer.convert_tokens_to_ids("[Title]")
    unk_id = tokenizer.convert_tokens_to_ids("[UNK]")
    sep_id = tokenizer.convert_tokens_to_ids("[SEP]")
    content_tokens = ["[CLS]"] + content_tokens + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(content_tokens)
    input_ids = [copy.deepcopy(input_ids) for _ in range(args.batch_size)]
    token_type_ids = [[content_id]*len(content_tokens) for _ in range(args.batch_size)]
    input_tensors = torch.tensor(input_ids).long().to(device)
    token_type_tensors = torch.tensor(token_type_ids).long().to(device)
    next_token_type = torch.tensor([[title_id] for _ in range(args.batch_size)]).long().to(device)
    generated = []
    finish_set = set()
    with torch.no_grad():
        for _ in range(args.generate_max_len):
            outputs = model(input_ids=input_tensors, token_type_ids=token_type_tensors)
            next_token_logits = outputs[0][:, -1, :]
            for index in range(args.batch_size):
                for token_id in set([token_ids[index] for token_ids in generated]):
                    next_token_logits[index][token_id] /= args.repetition_penalty
            for next_token_logit in next_token_logits:
                next_token_logit[unk_id] = -float("Inf")
            filter_logits = top_k_top_p_filtering(next_token_logits, top_k=args.top_k, top_p=args.top_p)
            next_tokens = torch.multinomial(F.softmax(filter_logits, dim=-1), num_samples=1)
            for index, token_id in enumerate(next_tokens[:, 0]):
                if token_id == sep_id:
                    finish_set.add(index)
            finish_flag = True
            for index in range(args.batch_size):
                if index not in finish_set:
                    finish_flag = False
                    break
            if finish_flag:
                break
            generated.append([token.item() for token in next_tokens[:, 0]])
            input_tensors = torch.cat((input_tensors, next_tokens), dim=-1)
            token_type_tensors = torch.cat((token_type_tensors, next_token_type), dim=-1)

        candidate_responses = []
        for index in range(args.batch_size):
            responses = []
            for token_index in range(len(generated)):
                if generated[token_index][index] != sep_id:
                    responses.append(generated[token_index][index])
                else:
                    break
            candidate_responses.append("".join(tokenizer.convert_ids_to_tokens(responses)).replace("##", "").replace("[Space]", " "))
    return candidate_responses


def main():
    args = set_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICE"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() and int(args.device)>=0 else "cpu")
    tokenizer = BertTokenizer(vocab_file=args.voca_path)
    model = GPT2LMHeadModel.from_pretrained(args.output_dir)
    model.to(device)
    model.eval()
    print('开始对新闻生成标题，输入CTRL + Z，则退出')
    while True:
        content = input("输入的新闻正文为:")
        titles = predict_one_sample(model, tokenizer, device, args, content)
        for i, title in enumerate(titles):
            print("生成的第{}个标题为：{}".format(i, title))


if __name__ == '__main__':
    main()

