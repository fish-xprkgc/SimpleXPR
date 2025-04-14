from graph_utils import get_graph_manager
from utils import csv_to_column_dict

relation_dict = {}


def data_prepare(data_dir):
    global relation_dict
    get_graph_manager(data_dir + 'igraph.pkl')
    rel = csv_to_column_dict(data_dir + 'relations.csv')
    inverse = 'inverse_relation_'
    for i in rel:
        most_rel = rel[i]['most_relation'].split('\t')
        relation_dict[i] = most_rel[0]
        relation_dict[inverse + i] = most_rel[1]


def get_relation_dict():
    return relation_dict
