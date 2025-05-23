import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
import torch.optim as optim
from models import *
from data_dict import get_total_path, data_prepare
from config import args
from data_dict import merge_path, merge_batches, process_path


def train_demo_dp(max_hop_path: int, num_epochs=3):
    # 1. 准备数据
    total_path = get_total_path()
    dataset = KHopDataset(total_path, max_hop_path)
    batch_sizes = {}
    for k in dataset.available_ks:
        if k == 0:
            batch_sizes[k] = int(args.batch_size)
        else:
            batch_sizes[k] = int(args.batch_size // (0.5 + k / 2))
    print(batch_sizes)

    # 3. 创建数据加载器
    sampler = KHopBatchSampler(dataset, batch_sizes)
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=4  # 使用多个数据加载工作进程
    )

    # 4. 初始化模型并移至DP模式
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PathModel().to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
        model = DataParallel(model)

    # 5. 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    start_time = time.time()
    # 6. 训练循环
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(time.time() - start_time)
        model.train()
        print(time.time() - start_time)
        for batch_idx, batch in enumerate(dataloader):
            print(time.time() - start_time)
            if batch==None:
                break
            # 7. 准备输入数据
            paths = batch["paths"]  # 这里需要将文本路径转换为索引
            k = batch["k"]
            query, tail = [], []
            if k == 0:
                k = 1
            for path in paths:
                q, t = process_path(path, k)
                query.append(q)
                tail.append(t)
            head_tensor = merge_batches(query)
            tail_tensor = merge_batches(tail)
            print(batch_idx)
            print(time.time() - start_time)
            # 8. 前向传播
            '''
            outputs = model(paths_tensor, lengths)

            # 模拟标签 - 实际中需要从数据中获取
            labels = torch.randint(0, 2, (len(paths),)).float().to(device)

            # 9. 计算损失
            loss = criterion(outputs.squeeze(), labels)

            # 10. 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 打印训练信息
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}: k={k[0]}, Loss: {loss.item():.4f}")

            if batch_idx >= 20:  # 限制演示的批次数量
                break
'''
    print("Training completed!")


if __name__ == "__main__":
    data_prepare(args.data_dir, args.max_hop_path)
    train_demo_dp(args.max_hop_path)
