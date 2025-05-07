import argparse

from graph_utils import get_graph_manager

parser = argparse.ArgumentParser(description='config arguments')
parser.add_argument('--task', default='WN18RR', type=str, metavar='N',
                    help='dataset name and dir')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of workers')
parser.add_argument('--data-dir', default="./data/WN18RR/", type=str, metavar='N',
                    help='path to data dir')
parser.add_argument('--use-llm-relation', action='store_true',
                    help='use large language relation description')
parser.add_argument('--model-path', default="./model", type=str, metavar='N',
                    help='path to data dir')
parser.add_argument('--log-dir', default="./log", type=str, metavar='N',)
parser.add_argument('--save-dir', default="./checkpoint", type=str, metavar='N',)
args = parser.parse_args()


def get_args():
    return args
