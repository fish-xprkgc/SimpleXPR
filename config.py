import argparse
import random
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
parser.add_argument('--model-path', default="/mnt/data/sushiyuan/SimKGC/yhy/model/bert-base-uncased", type=str, metavar='N',
                    help='path to data dir')
parser.add_argument('--max-hop-path', default=5, type=int, metavar='N',
                    help='max hop paths')
parser.add_argument('--log-dir', default="./log", type=str, metavar='N',)
parser.add_argument('--save-dir', default="./checkpoint", type=str, metavar='N',)
parser.add_argument('--batch-size', default=1024, type=int, metavar='N',)
parser.add_argument('--seed', default=2025, type=int, metavar='N',)
parser.add_argument('--pooling', default="mean", type=str, metavar='N',)
parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr-scheduler', default='linear', type=str,
                    help='Lr scheduler to use')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--warmup', default=400, type=int, metavar='N',
                    help='warmup steps')
parser.add_argument('--max-to-keep', default=3, type=int, metavar='N',
                    help='max number of checkpoints to keep')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--use-amp', action='store_true',
                    help='Use amp if available')
parser.add_argument('--grad-clip', default=10.0, type=float, metavar='N',
                    help='gradient clipping')

args = parser.parse_args()
random.seed(args.seed)

def get_args():

    return args
