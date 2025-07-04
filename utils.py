import csv
import glob
import os
import random
import shutil

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


def move_dict_cuda(dict,device):
    for key in dict.keys():
        dict[key] = dict[key].to(device,non_blocking=True)
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
def save_checkpoint(state: dict, is_best: bool, filename: str):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.dirname(filename) + '/model_best.mdl')
    shutil.copyfile(filename, os.path.dirname(filename) + '/model_last.mdl')


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

