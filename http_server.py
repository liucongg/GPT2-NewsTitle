# -*- coding:utf-8 -*-
# @project: GPT2-NewsTitle
# @filename: http_server.py
# @author: 刘聪NLP
# @contact: logcongcong@gmail.com
# @time: 2020/12/19 20:49
"""
    文件说明:
    构建web服务文件
"""

from gevent import monkey
monkey.patch_all()
from flask import Flask, request, render_template
import argparse
from gevent import wsgi
from generate_title import predict_one_sample
import torch
from model import GPT2LMHeadModel
from transformers import BertTokenizer
import os


def set_args():
    """设置所需参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, help='设置预测时使用的显卡,使用CPU设置成-1即可')
    parser.add_argument('--output_dir', default='output_dir/checkpoint-111844', type=str, help='模型文件路径')
    parser.add_argument('--vocab_path', default='vocab/vocab.txt', type=str, help='词表，该词表为小词表，并增加了一些新的标记')
    parser.add_argument('--batch_size', default=3, type=int, help='生成标题的个数')
    parser.add_argument('--generate_max_len', default=32, type=int, help='生成标题的最大长度')
    parser.add_argument('--repetition_penalty', default=1.2, type=float, help='重复处罚率')
    parser.add_argument('--top_k', default=5, type=float, help='解码时保留概率最高的多少个标记')
    parser.add_argument('--top_p', default=0.95, type=float, help='解码时保留概率累加大于多少的标记')
    parser.add_argument('--max_len', type=int, default=512, help='输入模型的最大长度，要比config中n_ctx小')
    parser.add_argument('--http_id', type=str, default="0.0.0.0", help='ip地址')
    parser.add_argument('--port', type=int, default=5555, help='端口号')
    return parser.parse_args()


def start_sever():
    args = set_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICE"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")
    # 实例化tokenizer和model
    tokenizer = BertTokenizer.from_pretrained(args.vocab_path, do_lower_case=True)
    model = GPT2LMHeadModel.from_pretrained(args.output_dir)
    model.to(device)
    model.eval()
    print("load model ending!")
    app = Flask(__name__)

    @app.route('/')
    def index():
        return "This is News Title Generate Model Server"

    @app.route('/news-title-generate', methods=['Get', 'POST'])
    def response_request():
        if request.method == 'POST':
            content = request.form.get('content')
            titles = predict_one_sample(model, tokenizer, device, args, content)
            title_str = ""
            for i, t in enumerate(titles):
                title_str += "生成的第{}个标题为：{}\n".format(i+1, t)
            return render_template("index_ok.html", content=content, titles=title_str)
        return render_template("index.html")
    server = wsgi.WSGIServer((str(args.http_id), args.port), app)
    server.serve_forever()


if __name__ == '__main__':
    start_sever()

