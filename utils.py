import csv
import glob
import os
import random
import shutil
from typing import List, Tuple

import numpy as np
import torch

from graph_utils import get_graph_manager
from logger_config import logger


def csv_to_column_dict(file_path, delimiter=','):
    """
    读取 CSV 文件并转换为以第一列为键的字典结构
    :param file_path: CSV 文件路径
    :param delimiter: 分隔符，默认为竖线 '|'
    :return: 字典，格式为 {第一列值: {列名: 对应值, ...}, ...}
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=delimiter)
            headers = next(reader)
            result = {}
            for row in reader:
                if len(row) != len(headers):
                    continue  # 跳过不完整的行
                key = row[0]
                value_dict = {header: value for header, value in zip(headers[1:], row[1:])}
                result[key] = value_dict
            return result
    except:
        return {}


def _concat_name_desc(entity: str, entity_desc: str) -> str:
    if entity_desc.startswith(entity):
        entity_desc = entity_desc[len(entity):].strip()
    if entity_desc:
        return '{}: {}'.format(entity, entity_desc)
    return entity


def get_entity_str(entity: str) -> str:
    gm = get_graph_manager()
    entity_info = gm.get_node_info(entity)
    entity_name = entity_info['name']
    entity_desc = entity_info['description']
    entity_str = _concat_name_desc(entity_name, entity_desc)
    return entity_str


def move_dict_cuda(dict, device):
    for key in dict.keys():
        dict[key] = dict[key].to(device, non_blocking=True)
    return dict


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def save_checkpoint(state: dict, is_best: bool, filename: str, only_save_lora=False):
    """
    保存 checkpoint，支持只保存 LoRA 参数（非完整模型）
    """

    if only_save_lora:
        # 只提取 LoRA 参数，在构造 state_dict 前就过滤掉非 LoRA 权重
        lora_state_dict = {
            k: v for k, v in state['state_dict'].items()
            if 'lora_' in k or 'default.' in k or 'adapter' in k
        }
        state['state_dict'] = lora_state_dict  # 替换为仅 LoRA 的 state_dict

    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.dirname(filename) + '/model_best.mdl')
    shutil.copyfile(filename, os.path.dirname(filename) + '/model_last.mdl')


INVERSE_PREFIX = "inverse_relation_"

def get_inverse_relation(relation: str) -> str:
    """
    获取一个关系的逆关系。
    如果关系名已包含逆关系前缀，则移除前缀返回其原始关系；
    否则，添加前缀。
    """
    if relation.startswith(INVERSE_PREFIX):
        return relation[len(INVERSE_PREFIX):]
    else:
        return INVERSE_PREFIX + relation


def reverse_path_triples(path_triples: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    """
    将一条由三元组组成的路径完全反转。
    例如: [(h, r1, e1), (e1, r2, t)] -> [(t, inv_r2, e1), (e1, inv_r1, h)]
    """
    if not path_triples:
        return []

    reversed_p = []
    for h, r, t in reversed(path_triples):
        reversed_p.append((t, get_inverse_relation(r), h))
    return reversed_p


def format_path_to_string(query_rel: str, path: List[Tuple[str, str, str]]) -> str:
    """
    将包含完整实体的路径格式化为最终的制表符分隔的输出字符串。
    格式: query_relation\thead\trel1\tent1\trel2\tent2...
    """
    if not path:
        return ""

    head_entity = path[0][0]
    path_list = [query_rel, head_entity]

    for _, rel, tail in path:
        path_list.append(rel)
        path_list.append(tail)

    return '\t'.join(path_list)
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(f"trainable params: {trainable_params} || all params: {all_param} || "
          f"trainable%: {100 * trainable_params / all_param:.2f}%")


def delete_old_ckt(path_pattern: str, keep=5):
    files = sorted(glob.glob(path_pattern), key=os.path.getmtime, reverse=True)
    for f in files[keep:]:
        logger.info('Delete old checkpoint {}'.format(f))
        os.system('rm -f {}'.format(f))


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
