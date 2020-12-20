# -*- coding:utf-8 -*-
# @project: GPT2-NewsTitle
# @filename: data_helper.py
# @author: 刘聪NLP
# @contact: logcongcong@gmail.com
# @time: 2020/12/16 16:25
"""
    文件说明：
    数据预处理文件，将数据进行简单的清洗
    数据来源于新浪微博
    数据链接：https://www.jianshu.com/p/8f52352f0748?tdsourcetag=s_pcqq_aiomsg
"""

import re
from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import json
import random


def clean_weibo_title(title: str):
    """
    对微博数据中的标题内容（待生成）进行清洗
    Args:
        title: 标题

    Returns:

    """
    # 去除##符号（一般为微博数据的话题标记）
    title = re.sub(r"#", "", title)
    # 去除[]中间的文字（一般为微博数据中的表情）
    title = re.sub(r"(\[{1,2})(.*?)(\]{1,2})", "", title)
    # 合并标题中过多的空格
    title = re.sub(r"\s+", " ", title)
    return title


def clean_weibo_content(content: str):
    """
    对微博数据中的文本内容进行清洗
    Args:
        content: 文本

    Returns:

    """
    # 去除网址
    content = re.sub(r"(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b", "", content)
    # 合并正文中过多的空格
    content = re.sub(r"\s+", " ", content)
    # 去除\u200b字符
    content = content.replace("\u200b", "")
    return content


def clean_data(sample):
    """
    整体清洗函数，为了方便多线程使用
    Args:
        sample: 一个元组，包含正文内容和标题内容

    Returns:

    """
    (content, title) = sample
    sample = dict()
    # 清洗数据
    sample["title"] = clean_weibo_title(title.strip())
    sample["content"] = clean_weibo_content(content.strip())
    return sample


def build_news_data(content_path, title_path, train_save_path, test_save_path):
    """
    对微博数据进行清洗，构建训练集和测试集
    Args:
        content_path: 正文内容文件路径
        title_path: 标题内容文件路径
        train_save_path: 训练集文件路径
        test_save_path: 测试集文件路径

    Returns:

    """
    # 打开文件，并将其zip成一个文件
    content_data = open(content_path, "r", encoding="utf-8")
    title_data = open(title_path, "r", encoding="utf-8")
    data = zip(content_data.readlines(), title_data.readlines())
    # 使用多进程处理数据
    threads = min(8, cpu_count())
    with Pool(threads) as p:
        annoate_ = partial(clean_data)
        data = list(tqdm(p.imap(annoate_, data, chunksize=8),
                         desc="build data"
                         )
                    )
    # 对数据进行过滤，去除重复数据、正文内容字长小于100的数据和标题内容字长小于100的数据
    data_set = set()
    data_new = []
    for d in data:
        if d["content"] in data_set or len(d["content"]) < 100 or len(d["title"]) < 2:
            continue
        else:
            data_set.add(d["content"])
            data_new.append(d)
    # 分割数据，构建训练集和测试集
    random.shuffle(data_new)
    train_data = data_new[:-3000]
    test_data = data_new[-3000:]
    fin = open(train_save_path, "w", encoding="utf-8")
    fin.write(json.dumps(train_data, indent=4, ensure_ascii=False))
    fin.close()
    fin = open(test_save_path, "w", encoding="utf-8")
    fin.write(json.dumps(test_data, indent=4, ensure_ascii=False))
    fin.close()


if __name__ == '__main__':
    content_path_dir = "data_dir/train_text.txt"
    title_path_dir = "data_dir/train_label.txt"
    train_save_path_dir = "data_dir/train_data.json"
    test_save_path_dir = "data_dir/test_data.json"
    build_news_data(content_path_dir, title_path_dir, train_save_path_dir, test_save_path_dir)
