import copy
import json
import logging

import numpy as np
import torch
from torch.nn import DataParallel
from config import args
from models import PathModel
from data_dict import construct_next_dict, combined_rel_entity, get_combined_dict, data_prepare, get_graph_manager, \
    get_token_dict, merge_batches, process_path, merge_flag_parts
from utils import move_dict_cuda
from logger_config import logger


def construct_result_dict(next_dict):
    result_dict = {}
    with open(args.data_dir + 'test.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            rel = line[1]
            if rel not in result_dict:
                result_dict[rel] = []
            temp_dict = {}
            temp_dict['hr'] = rel + '[SEP]' + line[0]
            temp_dict['query'] = [rel, line[0]]
            temp_dict['tail'] = line[2]
            temp_dict['path'] = [[rel, line[0]]]
            temp_dict['path_flag'] = [True]
            temp_dict['path_logit'] = [1.0]
            temp_dict['mask'] = list(next_dict[temp_dict['hr']])  # 创建副本
            temp_dict['mask'].remove(temp_dict['tail'])
            if args.eval_mode == 2:
                temp_dict['finish_path'] = []
                temp_dict['finish_path_logit'] = []
            result_dict[rel].append(temp_dict)
            rel = 'inverse_relation_' + line[1]
            if rel not in result_dict:
                result_dict[rel] = []
            temp_dict = {}
            temp_dict['hr'] = rel + '[SEP]' + line[2]
            temp_dict['query'] = [rel, line[2]]
            temp_dict['tail'] = line[0]
            temp_dict['path'] = [[rel, line[2]]]
            temp_dict['path_logit'] = [1.0]
            temp_dict['path_flag'] = [True]
            if args.eval_mode == 2:
                temp_dict['finish_path'] = []
                temp_dict['finish_path_logit'] = []
            temp_dict['mask'] = list(next_dict[temp_dict['hr']])  # 创建副本
            temp_dict['mask'].remove(temp_dict['tail'])
            result_dict[rel].append(temp_dict)
    return result_dict


def evaluate_task(graph_structure):
    temp_next_dict = construct_next_dict(args.data_dir)
    all_data_dict = merge_flag_parts(temp_next_dict, None)
    if graph_structure == 'train':
        next_dict = merge_flag_parts(temp_next_dict, ['train'])
    elif graph_structure == 'valid':
        next_dict = merge_flag_parts(temp_next_dict, ['train', 'valid'])
    else:
        next_dict = merge_flag_parts(temp_next_dict, ['train', 'valid', 'test'])
    combined_rel_entity(next_dict)
    combined_dict = get_combined_dict()
    all_hr_str = np.array(list(combined_dict.keys()))
    hr_str_index_map = {value: idx for idx, value in enumerate(all_hr_str)}
    all_hr_tokens = list(combined_dict.values())

    result_dict = construct_result_dict(all_data_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PathModel.load_from_checkpoint(args.eval_model_path, args.model_path).to(device)
    model_obj = model
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
        model = DataParallel(model)
        model_obj = model.module
    dataset = all_hr_tokens
    all_embeddings = []

    for i in range(0, len(dataset), args.batch_size):
        batch_list = dataset[i:i + args.batch_size]
        batch_data = merge_batches(batch_list).to(device)
        # 构造 batch input_ids 和 attention_mask
        with torch.no_grad():
            embeddings = model(**batch_data)
        all_embeddings.append(embeddings)
    del batch_data, embeddings
    torch.cuda.empty_cache()
    all_hr_embeddings = torch.cat(all_embeddings, dim=0)  # shape: (100000, 768)

    print(all_hr_embeddings.shape)

    ema_decay = args.ema_decay
    query_dict = copy.deepcopy(result_dict)
    for hop in range(args.max_hop_path + 2):
        logger.info(str(hop) + ' hop data start')
        batch_limit = int(args.batch_size // (hop + 1))
        current_path_index = 0
        current_path = []
        current_node_index = 0
        start_node_index = 0
        for relation in result_dict:
            paths = result_dict[relation]
            for node in paths:
                path_index = {}
                for i in range(len(node['path_flag'])):
                    if node['path_flag'][i]:
                        current_path.append(node['path'][i])
                        path_index[i] = current_path_index
                        current_path_index += 1
                node['path_dict'] = path_index
                current_node_index += 1
                if current_path_index > batch_limit:
                    # beam search realize
                    current_path = [process_path(mini_path, hop + 1)[0] for mini_path in current_path]
                    path_tensor = merge_batches(current_path)
                    path_tensor = move_dict_cuda(path_tensor, device)
                    with torch.no_grad():
                        embeddings = model(**path_tensor)
                    now_node_index = 0
                    for relation in result_dict:
                        paths = result_dict[relation]
                        for node in paths:
                            now_node_index += 1
                            if now_node_index <= start_node_index:
                                continue
                            if now_node_index > current_node_index:
                                break
                            node_path_shuffle = []
                            for i in range(len(node['path_flag'])):
                                if node['path_flag'][i]:

                                    current_embedding = embeddings[node['path_dict'][i]]
                                    node_path = node['path'][i]
                                    tail_entity = node_path[-1]
                                    found_next = next_dict[tail_entity].copy()
                                    path_exist = [item for idx, item in enumerate(node['path'][i]) if
                                                  idx % 2 == 1]
                                    if hop != 0:
                                        found_next.add('[END]')
                                    else:

                                        found_next.discard(relation + '[SEP]' + node['tail'])

                                    found_next = list(found_next)
                                    indices = [hr_str_index_map[value] for value in found_next]
                                    tail = all_hr_embeddings[indices]
                                    similarity = torch.matmul(current_embedding, tail.t())
                                    similarity = similarity.squeeze(0).tolist()
                                    if not isinstance(similarity, list):
                                        similarity = [similarity]
                                    for item_index in range(len(found_next)):
                                        node_path_shuffle.append((ema_decay * similarity[item_index] + (1 - ema_decay) *
                                                                  node['path_logit'][i], found_next[item_index],
                                                                  node['path'][i], path_exist))
                            if args.eval_mode == 1:
                                current_path_list = [(node['path_logit'][i], '[current]', node['path'][i], []) for i in
                                                     range(len(node['path'])) if node['path_flag'][i] == False]
                                node_path_shuffle.extend(current_path_list)
                                node_path_shuffle.sort(reverse=True)
                                node['path'] = []
                                node['path_logit'] = []
                                node['path_flag'] = []
                                for node_info in node_path_shuffle:
                                    if len(node['path']) >= args.k_path:
                                        break
                                    new_path = copy.deepcopy(node_info[2])
                                    if node_info[1] == '[current]':
                                        node['path'].append(new_path)
                                        node['path_logit'].append(node_info[0])
                                        node['path_flag'].append(False)
                                    else:
                                        if node_info[1] == '[END]':
                                            if new_path[-1] in node['mask']:
                                                continue
                                            else:
                                                node['path'].append(new_path)
                                                node['path_logit'].append(node_info[0])
                                                node['path_flag'].append(False)
                                        else:
                                            new_path.extend(node_info[1].split('[SEP]'))
                                            if new_path[-1] in node_info[3]:
                                                continue
                                            node['path'].append(new_path)
                                            node['path_logit'].append(node_info[0])
                                            node['path_flag'].append(True)
                            elif args.eval_mode == 2:
                                node_path_shuffle.sort(reverse=True)
                                new_continue_path = []
                                new_over_path = []
                                for node_info in node_path_shuffle:
                                    if node_info[1] != '[END]':
                                        new_continue_path.append(node_info)
                                    else:
                                        new_over_path.append(node_info)
                                current_over_path = [(node['finish_path_logit'][i], '[END]', node['finish_path'][i], [])
                                                     for i
                                                     in
                                                     range(len(node['finish_path']))]
                                new_over_path.extend(current_over_path)
                                new_over_path.sort(reverse=True)
                                node['path'] = []
                                node['path_logit'] = []
                                node['path_flag'] = []
                                for node_info in new_continue_path:
                                    if len(node['path']) >= args.k_path:
                                        break
                                    new_path = copy.deepcopy(node_info[2])
                                    new_path.extend(node_info[1].split('[SEP]'))
                                    last_entity = new_path[-1]
                                    if last_entity in node_info[3]:
                                        continue
                                    node['path'].append(new_path)
                                    node['path_logit'].append(node_info[0])
                                    node['path_flag'].append(True)
                                current_tail_li = []
                                node['finish_path_logit'] = []
                                node['finish_path'] = []
                                for node_info in new_over_path:
                                    if len(node['finish_path']) >= args.k_path:
                                        break
                                    new_path = copy.deepcopy(node_info[2])
                                    last_entity = new_path[-1]
                                    if last_entity in node[
                                        'mask'] or last_entity in current_tail_li:
                                        continue
                                    current_tail_li.append(last_entity)
                                    node['finish_path'].append(new_path)
                                    node['finish_path_logit'].append(node_info[0])

                            else:
                                print('unknown eval mode')
                    del embeddings, path_tensor
                    start_node_index = current_node_index
                    current_path_index = 0
                    current_path = []
        if current_path != []:

            current_path = [process_path(mini_path, hop + 1)[0] for mini_path in current_path]
            path_tensor = merge_batches(current_path)
            path_tensor = move_dict_cuda(path_tensor, device)
            with torch.no_grad():
                embeddings = model(**path_tensor)
            now_node_index = 0
            for relation in result_dict:
                paths = result_dict[relation]
                for node in paths:
                    now_node_index += 1
                    if now_node_index <= start_node_index:
                        continue
                    if now_node_index > current_node_index:
                        break
                    node_path_shuffle = []
                    for i in range(len(node['path_flag'])):
                        if node['path_flag'][i]:

                            current_embedding = embeddings[node['path_dict'][i]]
                            node_path = node['path'][i]
                            tail_entity = node_path[-1]
                            found_next = next_dict[tail_entity].copy()
                            path_exist = [item for idx, item in enumerate(node['path'][i]) if
                                          idx % 2 == 1]
                            if hop != 0:
                                found_next.add('[END]')
                            else:
                                found_next.discard(relation + '[SEP]' + node['tail'])
                            found_next = list(found_next)
                            indices = [hr_str_index_map[value] for value in found_next]
                            tail = all_hr_embeddings[indices]
                            similarity = torch.matmul(current_embedding, tail.t())
                            similarity = similarity.squeeze(0).tolist()
                            if not isinstance(similarity, list):
                                similarity = [similarity]
                            for item_index in range(len(found_next)):
                                node_path_shuffle.append((ema_decay * similarity[item_index] + (1 - ema_decay) *
                                                          node['path_logit'][i], found_next[item_index],
                                                          node['path'][i], path_exist))
                    if args.eval_mode == 1:
                        current_path_list = [(node['path_logit'][i], '[current]', node['path'][i], []) for i in
                                             range(len(node['path'])) if node['path_flag'][i] == False]
                        node_path_shuffle.extend(current_path_list)
                        node_path_shuffle.sort(reverse=True)
                        node['path'] = []
                        node['path_logit'] = []
                        node['path_flag'] = []
                        for node_info in node_path_shuffle:
                            if len(node['path']) >= args.k_path:
                                break
                            new_path = copy.deepcopy(node_info[2])
                            if node_info[1] == '[current]':
                                node['path'].append(new_path)
                                node['path_logit'].append(node_info[0])
                                node['path_flag'].append(False)
                            else:
                                if node_info[1] == '[END]':
                                    if new_path[-1] in node['mask']:
                                        continue
                                    else:
                                        node['path'].append(new_path)
                                        node['path_logit'].append(node_info[0])
                                        node['path_flag'].append(False)
                                else:
                                    new_path.extend(node_info[1].split('[SEP]'))
                                    if new_path[-1] in node_info[3]:
                                        continue
                                    node['path'].append(new_path)
                                    node['path_logit'].append(node_info[0])
                                    node['path_flag'].append(True)
                    elif args.eval_mode == 2:
                        node_path_shuffle.sort(reverse=True)
                        new_continue_path = []
                        new_over_path = []
                        for node_info in node_path_shuffle:
                            if node_info[1] != '[END]':
                                new_continue_path.append(node_info)
                            else:
                                new_over_path.append(node_info)
                        current_over_path = [(node['finish_path_logit'][i], '[END]', node['finish_path'][i], [])
                                             for i
                                             in
                                             range(len(node['finish_path']))]
                        new_over_path.extend(current_over_path)
                        new_over_path.sort(reverse=True)
                        node['path'] = []
                        node['path_logit'] = []
                        node['path_flag'] = []
                        for node_info in new_continue_path:
                            if len(node['path']) >= args.k_path:
                                break
                            new_path = copy.deepcopy(node_info[2])
                            new_path.extend(node_info[1].split('[SEP]'))
                            last_entity = new_path[-1]
                            if last_entity in node_info[3]:
                                continue
                            node['path'].append(new_path)
                            node['path_logit'].append(node_info[0])
                            node['path_flag'].append(True)
                        current_tail_li = []
                        node['finish_path_logit'] = []
                        node['finish_path'] = []
                        for node_info in new_over_path:
                            if len(node['finish_path']) >= args.k_path:
                                break
                            new_path = copy.deepcopy(node_info[2])
                            last_entity = new_path[-1]
                            if last_entity in node[
                                'mask'] or last_entity in current_tail_li:
                                continue
                            current_tail_li.append(last_entity)
                            node['finish_path'].append(new_path)
                            node['finish_path_logit'].append(node_info[0])

                    else:
                        print('unknown eval mode')
    new_data_copy = copy.deepcopy(result_dict)
    for relation in new_data_copy:
        items = new_data_copy[relation]
        for item in items:
            del item['mask'], item['path_dict']
    with open('log_new/'+args.task +'_'+ graph_structure + '_path_result.json', 'w') as f:
        json.dump(new_data_copy, f, ensure_ascii=False, indent=4)
    logger.info('hop data finish')
    gm = get_graph_manager()
    node_ids = gm.get_all_entities()

    for relation in query_dict:
        relation_tail = [process_path([relation, i], query_hop=0)[1] for i in node_ids]
        all_embeddings = []

        for i in range(0, len(relation_tail), args.batch_size):
            batch_list = relation_tail[i:i + args.batch_size]
            batch_data = merge_batches(batch_list).to(device)
            # 构造 batch input_ids 和 attention_mask
            with torch.no_grad():
                embeddings = model(**batch_data)
            all_embeddings.append(embeddings)
        del batch_data, embeddings
        torch.cuda.empty_cache()
        all_tail_embeddings = torch.cat(all_embeddings, dim=0)
        paths = query_dict[relation]
        hr_str = []
        for node in paths:
            hr_str.append(process_path(node['query'], 1,task_type='task_type_hr')[0])
        hr_tensor = merge_batches(hr_str)
        hr_tensor = move_dict_cuda(hr_tensor, device)
        with torch.no_grad():
            embeddings = model(**hr_tensor)
        similarity = torch.matmul(embeddings, all_tail_embeddings.t())
        current_index = 0
        for node in paths:
            current_result = similarity[current_index]
            node['mask'].append(node['query'][1])
            for mask_str in node['mask']:
                mask_id = node_ids.index(mask_str)
                current_result[mask_id] = -1e4

            topk_values, topk_indices = torch.topk(current_result, k=10)

            node['hr_tail'] = []
            node['hr_logit'] = []

            # 直接使用索引获取结果，避免排序整个列表
            for value, idx in zip(topk_values.tolist(), topk_indices.tolist()):
                node['hr_tail'].append(node_ids[idx])
                node['hr_logit'].append(value)
            del node['mask']
            current_index = current_index + 1



    with open('log_new/' + graph_structure + '_hr_result.json', 'w') as f:
        json.dump(query_dict, f, ensure_ascii=False, indent=4)
    logger.info('hr data finish')
if __name__ == "__main__":
    logger.info('start')
    logger.info('当前随机种子为'+str(args.seed))
    data_prepare(args.data_dir)
    for i in ['train']:
        evaluate_task(i)
