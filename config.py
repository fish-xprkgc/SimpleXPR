import argparse

from graph_utils import get_graph_manager

parser = argparse.ArgumentParser(description='config arguments')
parser.add_argument('--task', default='FB15k237', type=str, metavar='N',
                    help='dataset name and dir')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of workers')
parser.add_argument('--data-dir', default="./data/FB15k237/", type=str, metavar='N',
                    help='path to data dir')
parser.add_argument('--correct-num', default=8, type=int, metavar='N',
                    help='correct nums')
args = parser.parse_args()


def get_args():
    return args
