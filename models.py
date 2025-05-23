import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Sampler
import random
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any
import math
import time

from config import args
from data_dict import data_prepare
from numpy.random import Generator, PCG64


class PathModel(nn.Module):
    """
    示例模型，处理多跳路径数据
    """

    def __init__(self, vocab_size=10000, embed_dim=128, hidden_dim=256):
        super(PathModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 1)  # 假设是二分类任务

    def forward(self, paths, lengths):
        # paths: 批量的路径索引
        # lengths: 每条路径的实际长度（用于pack_padded_sequence）

        # 嵌入层
        embedded = self.embedding(paths)

        # 打包序列
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False)

        # RNN处理
        _, hidden = self.rnn(packed)

        # 分类
        output = self.classifier(hidden.squeeze(0))
        return torch.sigmoid(output)


class KHopDataset(Dataset):
    def __init__(self, total_path: List[List[List[str]]], max_hop_path: int):
        """
        初始化数据集（兼容直接索引访问）
        """
        self.data_by_k = defaultdict(list)

        # 原始分组存储（保持采样器所需结构）
        for hop in range(max_hop_path + 2):
            self.data_by_k[hop] = total_path[hop]
        self.data_by_k = dict(self.data_by_k)

        # 新增扁平化索引结构（用于兼容__getitem__）
        self.flat_data = []
        self.flat_k = []
        for k, paths in self.data_by_k.items():
            self.flat_data.extend(paths)
            self.flat_k.extend([k] * len(paths))

        # 原始统计信息（采样器需要）
        self.k_counts = {k: len(items) for k, items in self.data_by_k.items()}
        total = sum(self.k_counts.values())
        self.k_probs = {k: v / total for k, v in self.k_counts.items()}  # 修正这里
        self.available_ks = list(self.k_counts.keys())

    def __len__(self):
        return len(self.flat_data)  # 返回总数据量

    def __getitem__(self, index):
        """
        兼容性实现：按扁平化索引返回单条数据
        """
        return {
            "path": self.flat_data[index],
            "k": self.flat_k[index]
        }

    # 保持原有方法不变（供采样器使用）
    def get_batch(self, k: int, indices: List[int]):
        return [self.data_by_k[k][i] for i in indices]

    def update_sampling_probs(self, remaining_counts: Dict[int, int]):
        total = sum(remaining_counts.values())
        if total > 0:
            self.k_probs = {k: count / total for k, count in remaining_counts.items()}
        self.available_ks = [k for k in remaining_counts if remaining_counts[k] > 0]

class KHopBatchSampler(Sampler):
    def __init__(self, dataset: 'KHopDataset', batch_sizes: Dict[int, int], drop_last: bool = False):
        """
        动态跳数批次采样器

        参数:
            dataset: KHopDataset实例
            batch_sizes: 每个跳数对应的批次大小
            drop_last: 是否丢弃不足一个批次的数据
        """
        super().__init__()
        self.dataset = dataset
        self.batch_sizes = batch_sizes
        self.drop_last = drop_last

        # 初始化各跳数的可用索引为 set 而不是 list
        self.available_indices = {
            k: set(range(count))  # 改成 set！
            for k, count in dataset.k_counts.items()
        }

        # 记录各跳数剩余数据量
        self.remaining_counts = dataset.k_counts.copy()

        # 在 sampler 初始化时创建 RNG
        self.rg = Generator(PCG64(args.seed))  # 可选种子

        # 使用方式

    @property
    def available_ks(self):
        return [k for k in self.remaining_counts if self.remaining_counts[k] > 0]

    def __iter__(self):
        # 重置可用索引
        self.available_indices = {k: set(range(cnt)) for k, cnt in self.dataset.k_counts.items()}

        while len(self.available_ks) > 0:
            # 选择跳数k
            chosen_k = random.choices(
                list(self.dataset.k_probs.keys()),
                weights=list(self.dataset.k_probs.values()),
                k=1
            )[0]

            # 获取批次索引
            batch_size = self.batch_sizes[chosen_k]
            available = list(self.available_indices[chosen_k])
            indices = random.sample(available, min(batch_size, len(available)))

            # 更新状态
            self.available_indices[chosen_k] -= set(indices)
            self.dataset.update_sampling_probs(
                {k: len(v) for k, v in self.available_indices.items()}
            )

            yield indices  # 关键修改：返回索引列表而非数据

    def __len__(self):
        # 估计总批次数（不精确，因为是动态采样）
        total_batches = 0
        for k, count in self.dataset.k_counts.items():
            bs = self.batch_sizes[k]
            total_batches += count // bs
            if not self.drop_last and count % bs != 0:
                total_batches += 1
        return total_batches


def collate_fn(batch_data: List[Dict[str, Any]]):
    """
    新版collate_fn：处理来自__getitem__的数据结构
    """
    if not batch_data:
        return None  # 或其他适当的处理
    return {
        "paths": [item["path"] for item in batch_data],
        "k": batch_data[0]["k"]  # 取第一个k值代表整个批次
    }
