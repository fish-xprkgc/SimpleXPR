import csv
from graph_utils import get_graph_manager


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
