# -*- coding:utf-8 -*-
# @project: GPT2-NewsTitle
# @filename: data_set.py
# @author: 刘聪NLP
# @contact: logcongcong@gmail.com
# @time: 2020/12/16 16:25
"""
    文件说明：
    数据类文件，定义模型所需的数据类，方便模型训练使用
"""

import torch
import json
import os
from tqdm import tqdm
from torch.utils.data import Dataset


class GPT2NewsTitleDataSet(Dataset):
    def __init__(self, tokenizer, max_len, data_dir, data_set_name, path_file=None, is_overwrite=False):
        self.tokenizer = tokenizer
        self.content_id = self.tokenizer.convert_tokens_to_ids("[Content]")
        self.title_id = self.tokenizer.convert_tokens_to_ids("[Title]")
        self.space_id = self.tokenizer.convert_tokens_to_ids("[Space]")
        self.max_len = max_len
        cached_feature_file = os.path.join(data_dir, "cached_{}_{}".format(data_set_name, max_len))
        if os.path.exists(cached_feature_file) and not is_overwrite:
            self.data_set = torch.load(cached_feature_file)["data_set"]
        else:
            self.data_set = self.load_data(path_file)
            torch.save({"data_set": self.data_set}, cached_feature_file)

    def load_data(self, path_file):
        self.data_set = []
        with open(path_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            for idx, sample in enumerate(tqdm(data, desc="iter", disable=False)):
                input_ids, token_type_ids = self.convert_feature(sample)
                self.data_set.append({"input_ids": input_ids, "token_type_ids": token_type_ids})
        return self.data_set

    def convert_feature(self, sample):
        input_ids = []
        token_type_ids = []
        content_tokens = self.tokenizer.tokenize(sample["content"])
        title_tokens = self.tokenizer.tokenize(sample["title"].replace(" ", "[space]"))
        if len(content_tokens) > self.max_len - len(title_tokens) - 3:
            content_tokens = content_tokens[:self.max_len - len(title_tokens) - 3]
        input_ids.append(self.tokenizer.cls_token_id)
        token_type_ids.append(self.content_id)
        input_ids.extend(self.tokenizer.convert_tokens_to_ids(content_tokens))
        token_type_ids.extend([self.content_id] * len(content_tokens))
        input_ids.append(self.tokenizer.sep_token_id)
        token_type_ids.append(self.content_id)
        input_ids.extend(self.tokenizer.convert_tokens_to_ids(title_tokens))
        token_type_ids.extend([self.title_id] * len(title_tokens))
        input_ids.append(self.tokenizer.sep_token_id)
        token_type_ids.append(self.title_id)
        assert len(input_ids) == token_type_ids
        assert len(input_ids) <= self.max_len
        return input_ids, token_type_ids

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, item):
        instance = self.data_set[item]
        return instance


def collate_func(batch_data):
    batch_size = len(batch_data)
    if batch_size == 0:
        return {}
    input_ids_list, token_type_ids_list = [], []
    max_len = max([len(instance["input_ids"]) for instance in batch_data])
    for instance in batch_data:
        input_ids_temp = instance["input_ids"].extend([0]*(max_len-len(instance["input_ids"])))
        token_type_ids_temp = instance["token_type_ids"].extend([0] * (max_len - len(instance["token_type_ids"])))
        input_ids_list.append(torch.tensor(input_ids_temp, dtype=torch.long))
        token_type_ids_list.append(torch.tensor(token_type_ids_temp, dtype=torch.long))
    return {"input_ids": input_ids_list,
            "token_type_ids": token_type_ids_list}

