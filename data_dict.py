from typing import List

import torch
from transformers import BatchEncoding

from graph_utils import get_graph_manager
from utils import csv_to_column_dict

relation_dict = {}
token_dict = {}


def data_prepare(data_dir, max_hop_path=5):
    global relation_dict
    get_graph_manager(data_dir + 'igraph.pkl')
    rel = csv_to_column_dict(data_dir + 'relations.csv')
    inverse = 'inverse_relation_'
    for i in rel:
        most_rel = rel[i]['most_relation'].split('\t')
        relation_dict[i] = most_rel[0]
        relation_dict[inverse + i] = most_rel[1]
    path_generator(data_dir, max_hop_path)


def path_generator(data_dir, max_hop_path):
    total_path = [[] for i in range(max_hop_path + 2)]
    with open(data_dir + 'paths.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            hop_nums = int(len(line) / 2 - 1)
            if hop_nums == 1:
                if line[0] == line[2]:
                    hop_nums = 0
            total_path[hop_nums].append(line)
    tokenize_data(total_path)


def tokenize_data(path_data):
    print(len(path_data))
    for paths in path_data:
        print(len(paths))
        for path in paths:
            pass
        break


def merge_batches(batches: List[BatchEncoding]) -> BatchEncoding:
    """
    合并多个 BatchEncoding 对象为一个 BatchEncoding

    Args:
        batches (List[BatchEncoding]): 包含多个 BatchEncoding 的列表

    Returns:
        BatchEncoding: 合并后的 BatchEncoding
    """
    if not batches:
        return BatchEncoding({})  # 空列表返回空对象，或可改为 raise ValueError

    # 检查所有 BatchEncoding 的字段是否一致
    keys = batches[0].keys()
    for batch in batches[1:]:
        if batch.keys() != keys:
            raise ValueError("BatchEncoding 字段不一致，无法合并")

    merged = {}
    for key in keys:


        # 收集所有张量
        tensors = [batch[key] for batch in batches]
        # 沿批量维度拼接
        merged[key] = torch.cat(tensors, dim=0)

    return BatchEncoding(merged)


def get_relation_dict():
    return relation_dict


def get_token_dict():
    return token_dict


data_prepare('D:\PycharmProjects\XPR-KGC\data\WN18RR/')
