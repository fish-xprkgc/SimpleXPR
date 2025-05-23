import time

from torch import nn
from torch.nn.utils.rnn import pad_sequence
from graph_utils import get_graph_manager
from utils import csv_to_column_dict, _concat_name_desc
from config import args
import torch
from transformers import BatchEncoding, AutoTokenizer
from typing import List

relation_dict = {}
token_dict = {}
total_path = []


def data_prepare(data_dir, max_hop_path=5):
    global relation_dict
    gm = get_graph_manager(data_dir + 'igraph.pkl')
    rel = csv_to_column_dict(data_dir + 'relations.csv')
    inverse = 'inverse_relation_'
    for i in rel:
        most_rel = rel[i]['most_relation'].split('\t')
        if args.use_llm_relation:
            relation_dict[i] = most_rel[0]
            relation_dict[inverse + i] = most_rel[1]
        else:
            relation_dict[i] = i
            relation_dict[inverse + i] = inverse + i
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    token_dict['query_flag'] = tokenizer('query: ', return_tensors='pt', padding=True, truncation=True, max_length=50)
    token_dict['path_flag'] = tokenizer(',path: ', return_tensors='pt', padding=True, truncation=True, max_length=50)
    token_dict['connect_flag'] = tokenizer('>', return_tensors='pt', padding=True, truncation=True, max_length=50)
    token_dict['end_flag'] = tokenizer('end of path', return_tensors='pt', padding=True, truncation=True, max_length=50)
    for key in relation_dict:
        token_dict[key] = tokenizer(relation_dict[key], return_tensors='pt', padding=True, truncation=True,
                                    max_length=50)
    node_ids = gm.get_all_entities()
    for node_id in node_ids:
        node_info = gm.get_node_info(node_id)
        node_str = _concat_name_desc(node_info['name'], node_info['description'])
        token_dict[node_id] = tokenizer(node_str, padding=True, return_tensors='pt', truncation=True, max_length=50)

    path_generator(data_dir, max_hop_path)


def path_generator(data_dir, max_hop_path):
    global total_path
    total_path = [[] for i in range(max_hop_path + 3)]
    num = 0
    start_time = time.time()
    with open(data_dir + 'sample_paths.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            num += 1
            if num % 1000 == 0:
                print(num / 1000)
                print(time.time() - start_time)
            line = line.strip().split('\t')
            hop_nums = int(len(line) / 2 - 1)
            if line[0] != line[2] and hop_nums == 1:
                temp_line = line[0:2]
                temp_line.append(line[0])
                temp_line.append(line[-1])
                total_path[0].append(temp_line)
            if hop_nums == 1 and line[0] == line[2]:
                hop_nums = 0
            if hop_nums > 1:
                for k in range(1, hop_nums):
                    total_path[k].append(line[0:2 * k + 2])
                temp_line = line[0:2]
                temp_line.append(line[0])
                temp_line.append(line[-1])
                total_path[0].append(temp_line)
            total_path[hop_nums].append(line)
            if hop_nums == 0:
                total_path[2].append(line)
            else:
                total_path[hop_nums+1].append(line)

def process_path(path_arr, query_hop=1):
    path_arr = [token_dict[i] for i in path_arr]
    path_head = path_arr[0:query_hop * 2]
    if len(path_arr) == query_hop * 2:
        path_tail = [token_dict['connect_flag'], token_dict['end_flag']]
    else:
        path_tail = [token_dict['connect_flag'], path_arr[query_hop * 2], token_dict['connect_flag'],
                     path_arr[query_hop * 2 + 1]]
    query = [token_dict['query_flag'], path_head[0], token_dict['path_flag']]
    for i in range(1, query_hop * 2):
        query.append(path_head[i])
    head, tail = merge_path(query), merge_path(path_tail)
    return head, tail


def merge_path(batches: List[BatchEncoding]) -> BatchEncoding:
    if not batches:
        return BatchEncoding({})
    # 检查字段一致性
    keys = batches[0].keys()
    for batch in batches[1:]:
        if batch.keys() != keys:
            raise ValueError("BatchEncoding 字段不一致")
    merged = {}
    for key in keys:
        combined_sequence = []
        num_batches = len(batches)

        for i, batch in enumerate(batches):
            tensor = batch[key]  # 输入形状 [1, seq_len]
            # 验证输入维度
            if tensor.dim() != 2 or tensor.size(0) != 1:
                raise ValueError(f"无效的输入维度 {tensor.shape}，应满足 [1, N]")

            # 切片逻辑（保留原始逻辑）
            if num_batches == 1:
                return batches[0]
            else:
                if i == 0:
                    sliced = tensor[:, :-1]  # 第一个去掉尾部 [1, seq_len-1]
                elif i == num_batches - 1:
                    sliced = tensor[:, 1:]  # 最后去掉头部 [1, seq_len-1]
                else:
                    sliced = tensor[:, 1:-1]  # 中间去头尾 [1, seq_len-2]

            combined_sequence.append(sliced)
        # 核心修改：沿序列维度拼接（dim=1）
        merged[key] = torch.cat(combined_sequence, dim=1)  # 结果形状 [1, total_sliced_len]

    return BatchEncoding(merged)


def merge_batches(batches):
    """合并多个batch的数据，自动处理填充和维度对齐

    Args:
        batches: List[Dict[str, Tensor]]，多个batch的数据列表，
                 每个batch的每个字段张量形状为 (1, seq_len)

    Returns:
        BatchEncoding: 合并后的批次数据，每个字段形状为 (batch_size, max_seq_len)
    """
    # 处理空输入
    if not batches:
        return BatchEncoding()

    merged = {}
    for key in batches[0].keys():
        # 提取所有张量并去除首维的batch维度（原形状需为 (1, L)）
        # 使用生成器表达式减少内存占用
        tensors = (batch[key].squeeze(0) for batch in batches)

        # 核心改进：对attention_mask使用1填充（其他字段用0）
        padding_value = 1 if key == "attention_mask" else 0

        merged[key] = pad_sequence(
            tensors,
            batch_first=True,
            padding_value=padding_value
        )

    # 将合并后的字典转换为BatchEncoding对象返回
    return BatchEncoding(merged)




def get_total_path():
    return total_path


def get_relation_dict():
    return relation_dict


def get_token_dict():
    return token_dict

