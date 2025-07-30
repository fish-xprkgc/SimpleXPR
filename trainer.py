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
from data_dict import get_train_path, data_prepare, get_valid_path, path_prepare, path_next_dict, token_dict
from config import args
from data_dict import merge_path, merge_batches, process_path, create_matrix_optimized
from utils import move_dict_cuda, AverageMeter, ProgressMeter, save_checkpoint, delete_old_ckt, \
    print_trainable_parameters
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import os

batch_sizes = {}


def train_dp(max_hop_path: int, num_epochs=3):
    # 1. 准备数据
    global batch_sizes
    train_path = get_train_path()
    train_dataset = KHopDataset(train_path, max_hop_path)
    valid_path = get_valid_path()
    valid_dataset = KHopDataset(valid_path, max_hop_path)
    batch_sizes = {}
    for k in train_dataset.available_ks:
        if k == 0 or k == 1:
            batch_sizes[k] = int(args.batch_size)
        else:
            if args.only_tail:
                mini_len = k
                if k > 1:
                    mini_len = 2
                batch_sizes[k] = int(args.batch_size // (0.5 + mini_len / 2))
            else:
                batch_sizes[k] = int(args.batch_size // (0.5 + k / 2))
                if args.train_tail:
                    batch_sizes[k] = int(args.batch_size // (0.5 + (k - 1) / 2))
    print(batch_sizes)
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
    model = PathModel(args.model_path, args.lora).to(device)
    model_obj = model
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
        model = DataParallel(model)
        model_obj = model.module
    print_trainable_parameters(model)
    # 5. 定义优化器和损失函数
    optimizer = AdamW([p for p in model_obj.parameters() if p.requires_grad],
                      lr=args.lr,
                      weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss().cuda()

    num_training_steps = args.epochs * len(train_sampler)
    print(num_training_steps)
    scheduler = _create_lr_scheduler(optimizer, num_training_steps)
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

        for i, batch in enumerate(train_dataloader):
            if batch == None:
                continue
            batch_size = len(batch['paths'])
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.only_tail:
                        logits, labels = get_logits_labels_tail(batch, model, model_obj, device)
                    else:
                        logits, labels = get_logits_labels(batch, model, model_obj, device)
                    loss = criterion(logits, labels)
            else:
                if args.only_tail:
                    logits, labels = get_logits_labels_tail(batch, model, model_obj, device)
                else:
                    logits, labels = get_logits_labels(batch, model, model_obj, device)


                loss = criterion(logits, labels)
            losses.update(loss.item(), logits.size(0))
            if batch_size != batch_sizes[batch['k']]:
                pass
            else:
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

        do_eval(model, valid_dataloader, model_obj, device, epoch)

    print("Training completed!")


best_metric = None


def do_eval(model, valid_dataloader, model_obj, device, epoch):
    """
    修改后的验证函数，增加了按 k 值分组统计准确率和样本数的功能。
    """
    global best_metric
    model.eval()

    # 1. 保留原有的全局统计对象
    top1_overall = AverageMeter('Acc@1_overall', ':6.2f')
    top3_overall = AverageMeter('Acc@3_overall', ':6.2f')
    top10_overall = AverageMeter('Acc@10_overall', ':6.2f')

    # 2. 【新增】创建一个字典，用于存储每个 k 值独立的 AverageMeter
    # 格式为: {k: {'top1': AverageMeter, 'top3': AverageMeter, ...}}
    meters_per_k = {}

    for i, batch in enumerate(valid_dataloader):
        # 处理可能出现的空批次
        if not batch['paths']:
            continue

        batch_size = len(batch['paths'])
        k = batch['k']  # 获取当前批次的 k 值

        # 3. 【新增】动态地为新的 k 值初始化一套 AverageMeter
        # 如果是第一次遇到这个 k 值，就为它创建专属的统计对象
        if k not in meters_per_k:
            meters_per_k[k] = {
                'top1': AverageMeter(f'Acc@1_k{k}', ':6.2f'),
                'top3': AverageMeter(f'Acc@3_k{k}', ':6.2f'),
                'top10': AverageMeter(f'Acc@10_k{k}', ':6.2f'),
            }

        with torch.no_grad():
            if args.only_tail:
                logits, labels = get_logits_labels_tail(batch, model, model_obj, device)
            else:
                logits, labels = get_logits_labels(batch, model, model_obj, device)

            # 如果 get_logits_labels 返回 None（例如，因为批次为空），则跳过
            if logits is None:
                continue
            else:
                acc1, acc3, acc10 = accuracy(logits, labels, topk=(1, 3, 10))

                # 更新全局统计数据
                top1_overall.update(acc1.item(), batch_size)
                top3_overall.update(acc3.item(), batch_size)
                top10_overall.update(acc10.item(), batch_size)

                # 4. 【新增】同时更新对应 k 值的专属统计数据
                meters_per_k[k]['top1'].update(acc1.item(), batch_size)
                meters_per_k[k]['top3'].update(acc3.item(), batch_size)
                meters_per_k[k]['top10'].update(acc10.item(), batch_size)

    # --- 循环结束后，进行日志记录 ---

    # 记录全局的平均指标
    metric_dict_overall = {
        'Acc@1': round(top1_overall.avg, 3),
        'Acc@3': round(top3_overall.avg, 3),
        'Acc@10': round(top10_overall.avg, 3),
    }
    logger.info('Epoch {}, Overall valid metric: {}'.format(epoch, json.dumps(metric_dict_overall)))

    # 5. 【新增】遍历 meters_per_k 字典，记录每个 k 值的指标和样本数
    logger.info("--- Metrics Breakdown per k ---")

    # 对 k 值进行排序，保证日志输出的顺序一致性
    sorted_k_keys = sorted(meters_per_k.keys())

    for k in sorted_k_keys:
        k_meters = meters_per_k[k]
        # AverageMeter 的 .count 属性记录了总样本数
        count = k_meters['top1'].count

        # 只有在该 k 值有数据的情况下才记录
        if count > 0:
            metric_dict_k = {
                'Acc@1': round(k_meters['top1'].avg, 3),
                'Acc@3': round(k_meters['top3'].avg, 3),
                'Acc@10': round(k_meters['top10'].avg, 3),
                'count': int(count)  # 包含该 k 值的样本总数
            }
            logger.info('  - For k={}: {}'.format(k, json.dumps(metric_dict_k)))
        else:
            # 如果某个 k 值的所有批次都被跳过了，也进行说明
            logger.info(f"  - For k={k}: No full-sized batches were processed to calculate metrics.")
    metric_dict=metric_dict_overall

    if epoch == 0:
        best_metric = metric_dict
        is_best = True
    else:
        is_best = metric_dict['Acc@1'] > best_metric['Acc@1']
        if is_best:
            best_metric = metric_dict

    filename = '{}/checkpoint_{}_{}.mdl'.format(args.save_dir, args.task, epoch)

    save_checkpoint({
        'epoch': epoch,
        'args': args.__dict__,
        'state_dict': model.state_dict(),
    }, is_best=is_best, filename=filename, only_save_lora=args.lora)
    delete_old_ckt(path_pattern='{}/checkpoint_*.mdl'.format(args.save_dir),
                   keep=args.max_to_keep)


def get_logits_labels_tail(batch, model, model_obj, device):
    """
    【only_tail 专用版 - 逻辑简化】
    本函数遵循 process_path 的设计，传递完整样本路径，并为其计算正确的 query_hop 分割点。
    """
    if not batch['paths']:
        return None, None

    # 1. 初始化
    batch_paths = batch['paths']
    task_group_k = batch['k']
    is_hr_task_batch = (task_group_k == 0)
    task_type = 'task_type_hr' if is_hr_task_batch else 'task_type_path'

    processed_paths = []  # 存储 (true_path, current_query_hop)
    gt_tails_texts = []
    all_tail_texts = {'eos'} if task_type == 'task_type_path' else set()

    # Part 1: 解析循环
    for path in batch_paths:
        is_eop = (path and path[-1] == 'eop')
        true_path = path[:-1] if is_eop else path

        current_query_hop = 0
        gt_text = ""

        if is_eop:
            gt_text = 'eos'
            current_query_hop = len(true_path) // 2
            all_tail_texts.add(','.join(true_path[-2:]))

        elif is_hr_task_batch:
            gt_text = true_path[-1]
            current_query_hop = 1  # HR任务的查询跳数固定为1

        else:  # 非 EOP 的路径任务
            gt_text = ','.join(true_path[-2:])
            current_query_hop = (len(true_path) - 2) // 2

        processed_paths.append({'path': true_path, 'hop': current_query_hop})
        gt_tails_texts.append(gt_text)
        all_tail_texts.add(gt_text)

    # Part 2: 后续处理
    unique_tails_texts = sorted(list(all_tail_texts))
    tail_to_idx = {text: i for i, text in enumerate(unique_tails_texts)}
    M_unique = len(unique_tails_texts)

    # 构建负采样 "钥匙"
    query_texts_for_masking = []
    for item in processed_paths:
        path, hop = item['path'], item['hop']
        target_len = 2 * hop
        query_path_text = task_type + ','.join(path[0:2]) + path[target_len - 1]
        query_texts_for_masking.append(query_path_text)

    # Tokenization
    query_list, labels_list = [], []
    for i, item in enumerate(processed_paths):
        q_tokens, _ = process_path(item['path'], query_hop=item['hop'], task_type=task_type)
        query_list.append(q_tokens)
        labels_list.append(tail_to_idx[gt_tails_texts[i]])

    tail_list = []
    for text in unique_tails_texts:
        if text == 'eos':
            _, t_tokens = process_path([], query_hop=0)
        else:
            if task_type == 'task_type_hr':
                _, t_tokens = process_path([text], query_hop=0, task_type=task_type)
            else:
                rel, ent = text.split(',')
                t_token_list = [token_dict['connect_flag'], token_dict[rel], token_dict[ent]]
                t_tokens = merge_path(t_token_list)
        tail_list.append(t_tokens)

    # 模型推理与损失计算
    query_tensor = merge_batches(query_list)
    tail_tensor = merge_batches(tail_list)
    labels = torch.tensor(labels_list, dtype=torch.long, device=device)

    if torch.cuda.is_available():
        query_tensor = move_dict_cuda(query_tensor, device)
        tail_tensor = move_dict_cuda(tail_tensor, device)

    if args.train_tail and not is_hr_task_batch:
        with torch.no_grad():
            tail_embedding = model(**tail_tensor, query=False)
        query_embedding = model(**query_tensor, query=True)
    else:
        query_embedding = model(**query_tensor, query=True)
        tail_embedding = model(**tail_tensor, query=False)

    logits = query_embedding.mm(tail_embedding.t())
    logits *= 20

    final_mask = torch.ones_like(logits, dtype=torch.bool)
    for i in range(len(processed_paths)):
        known_positives = path_next_dict.get(query_texts_for_masking[i], set()).copy()
        known_positives.add(gt_tails_texts[i])
        for j in range(M_unique):
            tail_text = unique_tails_texts[j]
            if tail_text in known_positives:
                final_mask[i, j] = False
        gt_idx = labels[i].item()
        final_mask[i, gt_idx] = True

    logits.masked_fill_(~final_mask, -1e4)
    return logits, labels
def get_logits_labels(batch, model, model_obj, device):
    """
    处理混合批次（包含常规样本和 eop 样本），并构建全局统一的 logits 和 mask。
    """
    # 如果批次为空，直接返回
    if not batch['paths']:
        return None, None

    # --- 步骤 1: 识别、分组并收集所有尾实体文本 ---
    k = batch['k']
    if k == 0: k = 1  # k=0 是特殊情况，实际路径长度为 1

    queries_data = []  # 存储所有查询的原始 path
    gt_tails_texts = []  # 存储每个查询对应的【基准正确尾实体】的文本
    exclusive_neg_map = {}  # 存储 eop 查询与其专属负样本的映射 {query_idx: neg_text}
    task_type = 'task_type_hr' if k == 1 and batch['k'] == 0 else 'task_type_path'
    if task_type == 'task_type_hr':
        all_tail_texts = set()
    else:
        all_tail_texts = {'eos'}  # 使用集合收集所有出现过的尾实体文本，'eos' 是 eop 的文本表示

    for i, path in enumerate(batch['paths']):
        queries_data.append(path)

        if len(path) == 2 * k:  # 这是 EOP 样本
            gt_tails_texts.append('eos')

            # 为这个 eop 样本生成专属负样本，格式为 'relation,entity'
            neg_parts = path[2 * k - 2: 2 * k]
            neg_text = ','.join(neg_parts)
            exclusive_neg_map[i] = neg_text
            all_tail_texts.add(neg_text)

        else:  # 这是常规样本
            if task_type == 'task_type_hr':
                gt_text = path[-1]
            else:
                gt_parts = path[2 * k: 2 * k + 2]
                gt_text = ','.join(gt_parts)
            gt_tails_texts.append(gt_text)
            all_tail_texts.add(gt_text)

    # --- 步骤 2: 构建全局唯一的尾实体池 ---
    unique_tails_texts = sorted(list(all_tail_texts))
    tail_to_idx = {text: i for i, text in enumerate(unique_tails_texts)}
    M_unique = len(unique_tails_texts)

    # --- 步骤 3: 准备 Query, Tail 和 Label 用于编码 ---
    query_list = []
    tail_list = []
    labels_list = []

    # 准备 Query 和 Label
    for i, path in enumerate(queries_data):
        q_tokens, _ = process_path(path, query_hop=k, task_type=task_type)
        query_list.append(q_tokens)
        labels_list.append(tail_to_idx[gt_tails_texts[i]])

    # 准备唯一的 Tail
    for text in unique_tails_texts:
        if text == 'eos':
            _, t_tokens = process_path([], query_hop=0)
        else:
            if task_type == 'task_type_hr':
                _, t_tokens = process_path([text], query_hop=0, task_type=task_type)
            else:
                rel, ent = text.split(',')
                t_token_list = [token_dict['connect_flag'], token_dict[rel], token_dict['connect_flag'],
                                token_dict[ent]]
                t_tokens = merge_path(t_token_list)
        tail_list.append(t_tokens)

    # --- 步骤 4: 统一编码 ---
    query_tensor = merge_batches(query_list)
    tail_tensor = merge_batches(tail_list)
    labels = torch.tensor(labels_list, dtype=torch.long, device=device)

    if torch.cuda.is_available():
        query_tensor = move_dict_cuda(query_tensor, device)
        tail_tensor = move_dict_cuda(tail_tensor, device)
    if args.train_tail and batch['k'] != 0:
        with torch.no_grad():
            tail_embedding = model(**tail_tensor, query=False)
        query_embedding = model(**query_tensor, query=True)
    else:
        query_embedding = model(**query_tensor, query=True)
        tail_embedding = model(**tail_tensor, query=False)

    # --- 步骤 5: 计算全局 Logits ---
    logits = query_embedding.mm(tail_embedding.t())  # 形状 -> [N, M_unique]
    logits *= 20
    # --- 步骤 6: 构建终极统一 Mask ---
    final_mask = torch.ones_like(logits, dtype=torch.bool)

    # 【应用户要求补充】生成 query_path 的文本表示
    query_texts = []
    target_len = 2 * k
    for path in queries_data:
        if args.only_tail:
            # 对于 only_tail 模式，query path 由 task_type, 头实体, 关系1, 和路径最后一个实体构成
            query_path = task_type + ','.join(path[0:2]) + path[target_len - 1]
        else:
            # 常规模式下，query path 由 task_type 和整个查询路径构成
            query_path = task_type + ','.join(path[0:target_len])
        query_texts.append(query_path)

    for i in range(len(queries_data)):  # 遍历每个查询
        known_positives = path_next_dict.get(query_texts[i], set()).copy()
        known_positives.add(gt_tails_texts[i])

        for j in range(M_unique):  # 遍历全局尾实体池
            tail_text = unique_tails_texts[j]
            if tail_text in known_positives:
                final_mask[i, j] = False

        # 关键：无论如何，都不能屏蔽基准正确答案
        gt_idx = labels[i].item()
        final_mask[i, gt_idx] = True

    # --- 步骤 7: 应用 Mask 并返回 ---
    logits.masked_fill_(~final_mask, -1e4)

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
    data_prepare(args.data_dir)
    path_prepare(args.data_dir, args.max_hop_path)
    logger.info('当前随机种子为' + str(args.seed))
    train_dp(args.max_hop_path, args.epochs)
