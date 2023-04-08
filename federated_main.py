import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
import time

import os
import copy
import numpy as np
from tqdm import tqdm

from utils import get_dataset, average_weights, exp_details,count_parameters
from draw import visualize

# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

import trainers.coop
import trainers.cocoop
import trainers.zsclip




def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.PROMPTFL = CN()
    cfg.TRAINER.PROMPTFL.N_CTX = 16  # number of context vectors
    cfg.TRAINER.PROMPTFL.CSC = False  # class-specific context
    cfg.TRAINER.PROMPTFL.CTX_INIT = ""  # initialization words
    cfg.TRAINER.PROMPTFL.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.PROMPTFL.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'


    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.DATASET.USERS = 2  # number of clients
    cfg.DATASET.IID = False  # is iid
    cfg.DATASET.USEALL = False # use all data for training instead of few shot
    cfg.DATASET.REPEATRATE = 0.0 # repeat rate on each client
    cfg.OPTIM.ROUND = 10 # global round
    cfg.OPTIM.MAX_EPOCH = 5 # local epoch

    cfg.MODEL.BACKBONE.PRETRAINED = True


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg



def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        # print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    # print_args(args, cfg)
    # print("Collecting env info ...")
    # print("** System info **\n{}\n".format(collect_env_info()))

    global_trainer = build_trainer(cfg)
    print("type",type(global_trainer))
    # count_parameters(global_trainer.model,"prompt_learner")
    # count_parameters(global_trainer.model, "image_encoder")
    # count_parameters(global_trainer.model, "text_encoder")
    global_trainer.fed_before_train(is_global=True)

    # copy weights
    global_weights = global_trainer.model.state_dict()
    local_weights, local_losses = [], []

    local_trainer = build_trainer(cfg)
    local_trainer.fed_before_train()

    # Training
    start_epoch = 0
    max_epoch = cfg.OPTIM.ROUND
    # global_trainer.before_train()
    global_test_acc_list = []
    global_test_error_list = []
    global_test_f1_list = []
    global_epoch_list = []
    global_time_list = []
    start = time.time()
    for epoch in range(start_epoch, max_epoch):
        # m = max(int(args.frac * args.num_users), 1)
        # idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        idxs_users = list(range(0,cfg.DATASET.USERS))
        print("idxs_users",idxs_users)
        print("------------local train start epoch:",epoch,"-------------")
        for idx in idxs_users:
            local_trainer.model.load_state_dict(global_weights)
            local_trainer.train(idx=idx,global_epoch=epoch,is_fed=True)
            local_weight = local_trainer.model.state_dict()
            local_weights.append(local_weight)
        print("------------local train finish epoch:",epoch,"-------------")

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_trainer.model.load_state_dict(global_weights)

        # Calculate avg training accuracy over all users at every epoch
        print("------------global test start-------------")
        result = global_trainer.test(is_global=True, current_epoch=epoch)
        global_test_acc_list.append(result[0])
        global_test_error_list.append(result[1])
        global_test_f1_list.append(result[2])
        global_epoch_list.append(epoch)
        global_time_list.append(time.time()-start)
        print("------------global test finish-------------")
        # print("------------local test start-------------")
        # for c in range(args.num_users):
        #     local_trainer.model.load_state_dict(global_weights)
        #     local_trainer.test()
        # print("------------local test finish-------------")
        print("Epoch on server :", epoch)
    local_trainer.fed_after_train()
    global_trainer.fed_after_train()
    visualize(global_test_acc_list, global_test_error_list, global_test_f1_list, global_epoch_list, global_time_list, cfg.OUTPUT_DIR)










if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    # parser.add_argument('--num_users', type=int, default=2, help="number of users: K")
    # parser.add_argument('--frac', type=float, default=1.0, help='the fraction of clients: C')
    args = parser.parse_args()
    main(args)








