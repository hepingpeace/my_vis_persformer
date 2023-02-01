import os
from utils.utils import *
from config import my_persformer_openlane
from experiments.ddp import ddp_init
from experiments.runner import *

def main():
    parser = define_args() # args in utils.py
    args = parser.parse_args()
    # specify dataset and model config
    # persformer_once.config(args)
    my_persformer_openlane.config(args)
    # initialize distributed data parallel set 
    # 初始化分布式数据并行集
    ddp_init(args)
    # define runner to begin training or evaluation 
    # 定义runner开始训练或评估
    runner = Runner(args)
    args.evaluate = True #查看可视化的结果
    if not args.evaluate:
        runner.train()
    else:
        runner.eval()

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
