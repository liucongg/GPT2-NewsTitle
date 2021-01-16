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
import logging
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


class GPT2NewsTitleDataSet(Dataset):
    """新闻标题生成模型所需要的数据类"""
    def __init__(self, tokenizer, max_len, title_max_len, data_dir, data_set_name, path_file=None, is_overwrite=False):
        """
        初始化函数
        Args:
            tokenizer: 分词器
            max_len: 数据的最大长度
            title_max_len: 生成标题的最大长度
            data_dir: 保存缓存文件的路径
            data_set_name: 数据集名字
            path_file: 原始数据文件
            is_overwrite: 是否重新生成缓存文件
        """
        self.tokenizer = tokenizer
        # content_id和title_id分别对应新闻的正文和标题，为了在模型中区分的更明显
        self.content_id = self.tokenizer.convert_tokens_to_ids("[Content]")
        self.title_id = self.tokenizer.convert_tokens_to_ids("[Title]")
        # space_id表示空格标记，由于一些标题中带有空格，如果直接使用tokenizer进行分词，会导致空格消失，会显得标题很奇怪
        # 但是又不方便同一替换成任意一个标点，因此将其用[Space]替换。
        self.space_id = self.tokenizer.convert_tokens_to_ids("[Space]")
        self.max_len = max_len
        self.title_max_len = title_max_len
        cached_feature_file = os.path.join(data_dir, "cached_{}_{}".format(data_set_name, max_len))
        # 判断缓存文件是否存在，如果存在，则直接加载处理后数据
        if os.path.exists(cached_feature_file) and not is_overwrite:
            logger.info("已经存在缓存文件{}，直接加载".format(cached_feature_file))
            self.data_set = torch.load(cached_feature_file)["data_set"]
        # 如果缓存数据不存在，则对原始数据进行数据处理操作，并将处理后的数据存成缓存文件
        else:
            logger.info("不存在缓存文件{}，进行数据预处理操作".format(cached_feature_file))
            self.data_set = self.load_data(path_file)
            logger.info("数据预处理操作完成，将处理后的数据存到{}中，作为缓存文件".format(cached_feature_file))
            torch.save({"data_set": self.data_set}, cached_feature_file)

    def load_data(self, path_file):
        """
        加载原始数据，生成数据处理后的数据
        Args:
            path_file: 原始数据路径

        Returns:

        """
        self.data_set = []
        with open(path_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            for idx, sample in enumerate(tqdm(data, desc="iter", disable=False)):
                # 使用convert_feature函数，对新闻正文和标题进行索引化，生成模型所需数据格式
                input_ids, token_type_ids = self.convert_feature(sample)
                self.data_set.append({"input_ids": input_ids, "token_type_ids": token_type_ids})
        return self.data_set

    def convert_feature(self, sample):
        """
        数据处理函数
        Args:
            sample: 一个字典，包含新闻的正文和新闻的标题，格式为{"content": content, "title": title}

        Returns:

        """
        input_ids = []
        token_type_ids = []
        # 对新闻正文进行tokenizer.tokenize分词
        content_tokens = self.tokenizer.tokenize(sample["content"])
        # 对新闻标题进行tokenizer.tokenize分词，注意tokenizer中已经将[Space]作为一个分隔符，不会切割成多个字符
        title_tokens = self.tokenizer.tokenize(sample["title"].replace(" ", "[Space]"))
        # 判断如果标题过长，进行截断
        if len(title_tokens) > self.title_max_len:
            title_tokens = title_tokens[:self.title_max_len]
        # 判断如果正文过长，进行截断
        if len(content_tokens) > self.max_len - len(title_tokens) - 3:
            content_tokens = content_tokens[:self.max_len - len(title_tokens) - 3]
        # 生成模型所需的input_ids和token_type_ids
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
        # 判断input_ids与token_type_ids长度是否一致
        assert len(input_ids) == len(token_type_ids)
        # 判断input_ids长度是否小于等于最大长度
        assert len(input_ids) <= self.max_len
        return input_ids, token_type_ids

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        instance = self.data_set[idx]
        return instance


def collate_func(batch_data):
    """
    DataLoader所需的collate_fun函数，将数据处理成tensor形式
    Args:
        batch_data: batch数据

    Returns:

    """
    batch_size = len(batch_data)
    # 如果batch_size为0，则返回一个空字典
    if batch_size == 0:
        return {}
    input_ids_list, token_type_ids_list = [], []
    for instance in batch_data:
        # 按照batch中的最大数据长度,对数据进行padding填充
        input_ids_temp = instance["input_ids"]
        token_type_ids_temp = instance["token_type_ids"]
        # 将input_ids_temp和token_type_ids_temp添加到对应的list中
        input_ids_list.append(torch.tensor(input_ids_temp, dtype=torch.long))
        token_type_ids_list.append(torch.tensor(token_type_ids_temp, dtype=torch.long))
    # 使用pad_sequence函数，会将list中所有的tensor进行长度补全，补全到一个batch数据中的最大长度，补全元素为padding_value
    return {"input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=0),
            "token_type_ids": pad_sequence(token_type_ids_list, batch_first=True, padding_value=0)}

