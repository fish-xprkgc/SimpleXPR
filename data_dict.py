from torch.nn.utils.rnn import pad_sequence
from graph_utils import get_graph_manager
from utils import csv_to_column_dict, _concat_name_desc
from config import args
import torch
from transformers import BatchEncoding, AutoTokenizer
from typing import List

relation_dict = {}
token_dict = {}


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
    token_dict['query_flag'] = tokenizer('query: ', padding=True, truncation=True, max_length=50)
    token_dict['path_flag'] = tokenizer(',path: ', padding=True, truncation=True, max_length=50)
    token_dict['connect_flag'] = tokenizer('>', padding=True, truncation=True, max_length=50)
    print(token_dict['connect_flag'] )
    token_dict['end_flag'] = tokenizer('end of path', padding=True, truncation=True, max_length=50)
    for key in relation_dict:
        token_dict[key] = tokenizer(relation_dict[key], padding=True, truncation=True, max_length=50)
    node_ids = gm.get_all_entities()
    for node_id in node_ids:
        node_info = gm.get_node_info(node_id)
        node_str = _concat_name_desc(node_info['name'], node_info['description'])
        token_dict[node_id] = tokenizer(node_str, padding=True, truncation=True, max_length=50)
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
                # 特殊处理单批次：前后重叠拼接
                head = tensor[:, :-1]  # [1, seq_len-1]
                tail = tensor[:, 1:]  # [1, seq_len-1]
                sliced = torch.cat([head, tail], dim=1)  # [1, 2*(seq_len-1)]
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
        Dict[str, Tensor]: 合并后的批次数据，每个字段形状为 (batch_size, max_seq_len)
    """
    # 处理空输入
    if not batches:
        return {}

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
    return merged


def get_relation_dict():
    return relation_dict


def get_token_dict():
    return token_dict


data_prepare('D:\PycharmProjects\XPR-KGC\data\WN18RR/')
