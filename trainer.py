import json

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
import torch.optim as optim
from logger_config import logger
from metric import accuracy
from models import *
from data_dict import get_train_path, data_prepare, get_valid_path
from config import args
from data_dict import merge_path, merge_batches, process_path, create_matrix_optimized
from utils import move_dict_cuda, AverageMeter, ProgressMeter, save_checkpoint, delete_old_ckt
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import os

def train_dp(max_hop_path: int, num_epochs=3):
    # 1. 准备数据
    train_path = get_train_path()
    train_dataset = KHopDataset(train_path, max_hop_path)
    valid_path = get_valid_path()
    valid_dataset = KHopDataset(valid_path, max_hop_path)
    batch_sizes = {}
    for k in train_dataset.available_ks:
        if k == 0:
            batch_sizes[k] = int(args.batch_size)
        else:
            batch_sizes[k] = int(args.batch_size // (0.5 + k / 2))

    # 3. 创建数据加载器
    train_sampler = KHopBatchSampler(train_dataset, batch_sizes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=4  # 使用多个数据加载工作进程
    )
    valid_sampler = KHopBatchSampler(valid_dataset, batch_sizes)
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_sampler=valid_sampler,
        collate_fn=collate_fn,
        num_workers=4
    )
    # 4. 初始化模型并移至DP模式
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PathModel(args.model_path).to(device)
    model_obj = model
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
        model = DataParallel(model)
        model_obj = model.module
    # 5. 定义优化器和损失函数
    optimizer = AdamW([p for p in model_obj.parameters() if p.requires_grad],
                      lr=args.lr,
                      weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss().cuda()



    num_training_steps = args.epochs * len(train_sampler)
    print(num_training_steps)
    scheduler = _create_lr_scheduler(optimizer,num_training_steps)
    args.warmup = min(args.warmup, num_training_steps // 10)
    print(args.warmup)
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    # 6. 训练循环
    for epoch in range(num_epochs):
        losses = AverageMeter('Loss', ':.4')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top3 = AverageMeter('Acc@3', ':6.2f')
        top10 = AverageMeter('Acc@10', ':6.2f')
        progress = ProgressMeter(
            len(train_dataloader),
            [losses, top1, top3, top10],
            prefix="Epoch: [{}]".format(epoch))
        model.train()

        for i,batch in enumerate(train_dataloader):
            if batch == None:
                continue
            batch_size=len(batch['paths'])
            logits, labels = get_logits_labels(batch, model, model_obj, device)
            loss = criterion(logits, labels)
            losses.update(loss.item(), logits.size(0))
            acc1, acc3, acc10 = accuracy(logits, labels, topk=(1, 3, 10))
            top1.update(acc1.item(), batch_size)
            top3.update(acc3.item(), batch_size)
            top10.update(acc10.item(), batch_size)
            optimizer.zero_grad()
            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            scheduler.step()
            if i % args.print_freq == 0:
                progress.display(i)
        logger.info('Learning rate: {}'.format(scheduler.get_last_lr()[0]))

        do_eval(model, valid_dataloader,model_obj, device,epoch)

    print("Training completed!")
best_metric=None
def do_eval(model, valid_dataloader,model_obj,device,epoch):
    global best_metric
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top3 = AverageMeter('Acc@3', ':6.2f')
    top10 = AverageMeter('Acc@10', ':6.2f')
    for i,batch in enumerate(valid_dataloader):
        batch_size = len(batch['paths'])
        with torch.no_grad():
            logits, labels = get_logits_labels(batch, model, model_obj, device)
            acc1, acc3, acc10 = accuracy(logits, labels, topk=(1, 3, 10))
            top1.update(acc1.item(), batch_size)
            top3.update(acc3.item(), batch_size)
            top10.update(acc10.item(), batch_size)

    metric_dict = {'Acc@1': round(top1.avg, 3),
                   'Acc@3': round(top3.avg, 3),
                   'Acc@10': round(top10.avg, 3),
                   }
    logger.info('Epoch {}, valid metric: {}'.format(epoch, json.dumps(metric_dict)))
    if epoch==0:
        best_metric=metric_dict
        is_best = True
    else:
        is_best = metric_dict['Acc@1'] > best_metric['Acc@1']
        if is_best:
            best_metric = metric_dict

    filename = '{}/checkpoint_{}_{}.mdl'.format(args.save_dir,args.task, epoch)

    save_checkpoint({
        'epoch': epoch,
        'args': args.__dict__,
        'state_dict': model.state_dict(),
    }, is_best=is_best, filename=filename)
    delete_old_ckt(path_pattern='{}/checkpoint_*.mdl'.format(args.save_dir),
                   keep=args.max_to_keep)
def get_logits_labels(batch, model, model_obj, device):
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
    query_tensor = merge_batches(query)
    tail_tensor = merge_batches(tail)
    if torch.cuda.is_available():
        query_tensor = move_dict_cuda(query_tensor, device)
        tail_tensor = move_dict_cuda(tail_tensor, device)
    query_embedding = model(**query_tensor)
    tail_embedding = model(**tail_tensor)

    mask = create_matrix_optimized(paths, k).to(device)
    logits, labels = model_obj.compute_logits(query_embedding, tail_embedding, mask)
    return logits, labels
def _create_lr_scheduler(optimizer, num_training_steps):
    if args.lr_scheduler == 'linear':
        return get_linear_schedule_with_warmup(optimizer=optimizer,
                                               num_warmup_steps=args.warmup,
                                               num_training_steps=num_training_steps)
    elif args.lr_scheduler == 'cosine':
        return get_cosine_schedule_with_warmup(optimizer=optimizer,
                                               num_warmup_steps=args.warmup,
                                               num_training_steps=num_training_steps)
    else:
        assert False, 'Unknown lr scheduler: {}'.format(args.scheduler)


if __name__ == "__main__":


    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    data_prepare(args.data_dir, args.max_hop_path)
    train_dp(args.max_hop_path, args.epochs)
