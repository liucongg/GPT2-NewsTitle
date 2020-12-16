# -*- coding:utf-8 -*-
# @project: GPT2-NewsTitle
# @filename: data_helper.py
# @author: 刘聪NLP
# @contact: logcongcong@gmail.com
# @time: 2020/12/16 16:25
"""
    文件说明：
    数据预处理文件，将数据进行简单的清洗
    数据来源于两处：（1）新浪微博和（2）NLPCC2017
    （1）数据链接：https://www.jianshu.com/p/8f52352f0748?tdsourcetag=s_pcqq_aiomsg
    （2）数据链接：http://tcci.ccf.org.cn/conference/2017/taskdata.php
"""

import re


def clean_weibo_title(title: str):
    pattern = re.compile(r'(#[1-3])(.*?)(#[1-3])')
    pattern = ""
    pattern = re.compile(r'(\[)(.*?)(\])')

    return pattern.sub(r'', pattern.sub(r'', title))


if __name__ == '__main__':
    print(clean_weibo_title("#111111#222[22]22#33333333#"))

