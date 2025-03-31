import csv

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
