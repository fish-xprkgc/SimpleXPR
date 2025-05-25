import argparse
import csv
import json
import os.path
import random

import igraph as ig

from graph_utils import get_graph_manager
from utils import csv_to_column_dict, _concat_name_desc

parser = argparse.ArgumentParser(description='preprocess')
parser.add_argument('--task', default='FB15k237', type=str, metavar='N',
                    help='dataset name and dir')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of workers')
parser.add_argument('--data-dir', default="./data/FB15k237/", type=str, metavar='N',
                    help='path to data dir')
parser.add_argument('--correct-num', default=8, type=int, metavar='N',
                    help='correct nums')
parser.add_argument('--max-hop-path', default=5, type=int, metavar='N',
                    help='max hop path')
args = parser.parse_args()


def find_most_frequent(arr):
    # 处理空数组
    if not arr:
        return []

    freq = {}
    for item in arr:
        freq[item] = freq.get(item, 0) + 1

    # 找到最高频率的正关系
    max_count = max(freq.values())
    most_frequent = [k for k, v in freq.items() if v == max_count]

    # 检查频率
    if max_count <= 2:
        return []

    return most_frequent[0]


def append_to_csv(file_path, new_row, delimiter=',', header=None):
    """
    修复版：安全处理文件路径中的目录
    """
    # 获取父目录路径
    dir_path = os.path.dirname(file_path)

    # 仅当目录路径非空时创建目录
    if dir_path:  # 避免空路径导致报错
        os.makedirs(dir_path, exist_ok=True)

    # 检查文件是否存在
    file_exists = os.path.isfile(file_path)

    with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=delimiter)

        if not file_exists and header:
            writer.writerow(header)

        writer.writerow(new_row)


def generate_prompt(relation):
    return f"""
请将{args.task}数据集中关系符号『{relation}』转换为适当的短语，要求：
1. 主关系：描述两个实体之间的关系（例如："is a type of"）。
2. 逆关系：描述两个实体关系的反向表达（例如："is classified as"）。
3. 输出格式：两个短语，用制表符分隔，无解释。
4. 输出语言必须为英文。
5. 确保短语适用于多种上下文，保持灵活性与准确性。
"""


def correct_prompt(ori_rel, current_rel, triple):
    h, t = triple[0], triple[1]
    return f"""
请根据头实体『{h}』和尾实体『{t}』的语义关系，按照以下规则处理关系短语：
1. 主关系必须自然适配句式："【{h}】____【{t}】"，并且能够适用于其他相似的实体。
2. 逆关系必须自然适配句式："【{t}】____【{h}】"，并且能够适用于其他相似的实体。
3. 仅当当前短语存在以下问题时才修正：
   - 逻辑关系颠倒
   - 语义错误
   - 语法结构不符合英语习惯
4. 输出的短语中的正关系必须和原本关系的语义相似，逆关系则是关系的反向表达，这是原关系『{ori_rel}』
5. 输出关系禁止包含任何和头实体，尾实体相关的信息
6. 输出格式：仅两个英文短语，使用制表符分隔，无其他文本。
当前短语对：
{current_rel[0]}\t{current_rel[1]}
"""


from openai import OpenAI

client = OpenAI(api_key="sk-0473a6b12c0e4727802f7eaf56f08f94", base_url="https://api.deepseek.com")


def get_response(prompt):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=False,
        temperature=1
    )
    return response.choices[0].message.content


import threading
import queue
import random


def process_relations_concurrent(rel_dict, relation_path, triple_rel_dict, num_threads=10):
    """
    多线程处理关系并生成数据的入口函数

    参数：
    rel_dict: 关系字典 {key: relation}
    relation_path: 关系文件路径
    args: 命令行参数对象
    triple_rel_dict: 三元组字典 {relation: triples}
    num_threads: 并发线程数 (默认4)
    """
    # 创建线程安全队列和锁
    relation_queue = queue.Queue()
    file_lock = threading.Lock()

    # 初始化队列（过滤已存在的关系）
    def init_queue():
        existing = csv_to_column_dict(relation_path)
        return [k for k in rel_dict if rel_dict[k] not in existing]

    # 工作线程处理函数
    def worker():
        while True:
            try:
                key = relation_queue.get_nowait()
            except queue.Empty:
                break

            try:
                process_relation(key)
                relation_queue.task_done()
            except Exception as e:
                print(f"处理 {key} 失败: {e}")
                # 重新放入队列进行重试
                relation_queue.put(key)
                relation_queue.task_done()

    # 单个关系处理逻辑
    def process_relation(key):
        relation = rel_dict[key]

        # 双重检查（防止其他线程已处理）
        with file_lock:
            existing = csv_to_column_dict(relation_path)
            if relation in existing:
                return

        # 生成初始响应
        prompt = generate_prompt(relation)
        res = get_response(prompt)
        # 多轮校正
        corrections = []
        most_common = res
        for _ in range(args.correct_num):
            try:
                # 使用线程安全随机选择
                with threading.Lock():
                    triple = random.choice(triple_rel_dict[relation])
                head, tail = triple['head'], triple['tail']
                new_prompt = correct_prompt(relation, res.split('\t'), [head, tail])
                current_res = get_response(new_prompt)
                corrections.append(current_res)
                result = find_most_frequent(corrections + [res])
                if not result:
                    continue
                else:
                    most_common = result
                    break
            except Exception as e:
                print(f"校正过程出错: {e}")
                continue

        # 确定最终关系

        # 线程安全写入
        with file_lock:
            append_to_csv(
                relation_path,
                [relation, res, most_common, corrections],
                delimiter=',',
                header=['ori_relation', 'new_relation', 'most_relation', 'relations']
            )

    # 主处理循环
    while True:
        try:
            # 初始化队列
            for key in init_queue():
                relation_queue.put(key)

            # 启动工作线程
            threads = []
            for _ in range(num_threads):
                t = threading.Thread(target=worker)
                t.start()
                threads.append(t)

            # 等待队列处理完成
            relation_queue.join()

            # 检查是否全部处理完成
            remaining = [k for k in rel_dict
                         if rel_dict[k] not in csv_to_column_dict(relation_path)]
            if not remaining:
                break

        except Exception as e:
            print(f"处理过程中发生错误: {e}")
            # 清空队列避免重复处理
            while not relation_queue.empty():
                relation_queue.get()
                relation_queue.task_done()
            continue

        # 等待所有线程退出
        for t in threads:
            t.join()

    print("所有关系处理完成")


def process_wn18rr_entity(data_dir):
    ids = open(data_dir + 'entities.dict', 'r', encoding='utf-8').readlines()
    gm = get_graph_manager()
    entity_dict = {}
    for line in ids:
        entity_dict[line.strip().split('\t')[1]] = {'entity_name': '', 'entity_desc': ''}
    name_file = open(data_dir + 'wordnet-mlj12-definitions.txt', 'r', encoding='utf-8').readlines()
    for line in name_file:
        line_data = line.strip().split('\t')
        entity_dict[line_data[0]]['entity_name'] = ' '.join(line_data[1].replace('__', '').split('_')[:-2]).strip()
        entity_dict[line_data[0]]['entity_desc'] = line_data[2].strip()
    for key in entity_dict.keys():
        gm.add_node(key, entity_dict[key]['entity_name'], entity_dict[key]['entity_desc'])
    return gm


def process_data(graph, data_file):
    rel_dict = {}
    triple_rel_dict = {}
    train_data = open(data_file, 'r', encoding='utf-8').readlines()
    relation_path = os.path.dirname(data_file) + '/relations.csv'
    graph_path = os.path.dirname(data_file) + '/igraph.pkl'
    if not os.path.exists(relation_path):
        for line in train_data:

            triple = line.strip().split('\t')
            if triple[1] not in rel_dict:
                rel_dict[triple[1]] = triple[1]
                head_node = graph.get_node_info(triple[0])
                tail_node = graph.get_node_info(triple[2])
                head_str = _concat_name_desc(head_node['name'], head_node['description'])
                tail_str = _concat_name_desc(tail_node['name'], tail_node['description'])
                triple_dic = {'head': head_str, 'tail': tail_str}
                triple_rel_dict[triple[1]] = [triple_dic]
            else:
                head_node = graph.get_node_info(triple[0])
                tail_node = graph.get_node_info(triple[2])
                head_str = _concat_name_desc(head_node['name'], head_node['description'])
                tail_str = _concat_name_desc(tail_node['name'], tail_node['description'])
                triple_dic = {'head': head_str, 'tail': tail_str}
                triple_rel_dict[triple[1]].append(triple_dic)

        process_relations_concurrent(rel_dict, relation_path, triple_rel_dict)

    if not os.path.exists(graph_path):
        i = 0
        for line in train_data:
            i += 1
            if i % 1000 == 0:
                print('added ' + str(i) + ' triples to graph')
            triple = line.strip().split('\t')
            graph.add_edge(triple[0], triple[2], triple[1])
            graph.add_edge(triple[2], triple[0], 'inverse_relation_' + triple[1])
        graph.save_pickle(graph_path)
        return graph


def process_fb15_entity(data_dir):
    ids = open(data_dir + 'FB15k_mid2name.txt', 'r', encoding='utf-8').readlines()
    gm = get_graph_manager()
    entity_dict = {}
    for line in ids:
        entity_dict[line.strip().split('\t')[0].strip()] = {
            'entity_name': line.strip().split('\t')[1].replace('_', ' '), 'entity_desc': ''}
    name_file = open(data_dir + 'FB15k_mid2description.txt', 'r', encoding='utf-8').readlines()
    for line in name_file:
        line_data = line.strip().split('\t')
        entity_dict[line_data[0]]['entity_desc'] = line_data[1].strip()[1:-4]
    for key in entity_dict.keys():
        gm.add_node(key, entity_dict[key]['entity_name'], entity_dict[key]['entity_desc'])
    return gm


def process_path(gm, data_dir, file_name,path_name):
    paths = []
    with open(data_dir + file_name, 'r') as f:
        lines = f.readlines()
        num = 0
        for line in lines:
            triple = line.strip().split('\t')
            gm.get_node_info(triple[0])
            gm.get_node_info(triple[2])

            x, path = gm.get_shortest_path(triple[0], triple[2], triple[1])
            if x == -1 or x > args.max_hop_path:
                current_path = '\t'.join([triple[1], triple[0], triple[1], triple[2]])
            else:
                current_path = [triple[1], triple[0]]
                for triple in path:
                    current_path.append(triple[1])
                    current_path.append(triple[2])
                current_path = '\t'.join(current_path)

            paths.append(current_path)
            triple = line.strip().split('\t')
            x, path = gm.get_shortest_path(triple[2], triple[0],'inverse_relation_'+triple[1])
            if x == -1 or x > args.max_hop_path:
                current_path = '\t'.join(['inverse_relation_'+triple[1], triple[2], 'inverse_relation_'+triple[1], triple[0]])
            else:
                current_path = ['inverse_relation_'+triple[1], triple[2]]
                for triple in path:
                    current_path.append(triple[1])
                    current_path.append(triple[2])
                current_path = '\t'.join(current_path)

            paths.append(current_path)
            num += 1
            if num % 1000 == 0:
                print('processed ' + str(num) + ' paths')

    with open(data_dir + path_name, 'w') as f:
        for path in paths:
            f.write(path + '\n')


def main():
    data_dir = args.data_dir
    if args.task.lower() == 'wn18rr':
        graph = process_wn18rr_entity(data_dir)
        process_data(graph, data_dir + 'train.txt')

    elif args.task.lower() == 'fb15k237':
        graph = process_fb15_entity(data_dir)
        process_data(graph, data_dir + 'train.txt')

    gm = get_graph_manager(data_dir + 'igraph.pkl')
    process_path(gm, data_dir,'train.txt','train_path.txt')
    process_path(gm, data_dir,'valid.txt','valid_path.txt')

if __name__ == '__main__':
    main()
