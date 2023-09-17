from dataset import load_dataset
from config import *


import torch
import deepspeed
from transformers.deepspeed import HfDeepSpeedConfig
import numpy as np
from tqdm import tqdm
import os
import random
import time
import logging
import argparse

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--model', type=str, default='nextgpt')
    parser.add_argument('--mode', type=str, default='train', help='train or test or validation')
    parser.add_argument('--dataset', type=str, default='cc3m', help='the dataset name, it could be cc3m, webvid, audiocap, instruction')
    parser.add_argument('--data_path', type=str, default='cc3m/cc3m.json')  # training data
    parser.add_argument('--mm_root_path', type=str, default='cc3m/images')  # training data
    parser.add_argument('--embed_path', type=str, default='./embed/')  # the embedding path of video/audio/image caption via text encoder in different diffusion model
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--save_path', type=str, default='../ckpt/delta_ckpt/nextgpt/7b_v0/')
    parser.add_argument('--log_path', type=str, default='../ckpt/delta_ckpt/nextgpt/7b_v0/log_rest/')
    parser.add_argument('--assets_path', type=str, default='./assets/')

    # model configurations
    parser.add_argument('--max_length', type=int, default=512)  # the maximum input sequence length for LLMs
    parser.add_argument('--stage', type=int, default=1)  # the training stage
    parser.add_argument('--modality', type=list, default=['image', 'video', 'audio', 'text'])
    return parser.parse_args()


def initialize_distributed(args):
    args['master_ip'] = os.getenv('MASTER_ADDR', 'localhost')
    args['master_port'] = os.getenv('MASTER_PORT', '6000')
    args['world_size'] = int(os.getenv('WORLD_SIZE', '1'))
    args['local_rank'] = int(os.getenv('RANK', '0')) % torch.cuda.device_count()
    device = args['local_rank'] % torch.cuda.device_count()
    torch.cuda.set_device(device)
    deepspeed.init_distributed(dist_backend='nccl')


def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def config_env(args):
    args['root_dir'] = '../'
    # args['mode'] = 'train'
    config = load_config(args)
    args.update(config)
    initialize_distributed(args)
    set_random_seed(args['seed'])


def build_directory(path):
    if os.path.exists(path):
        pass
    else:  # recursively construct directory
        os.makedirs(path, exist_ok=True)


def main(**args):
    config_env(args)
    print(args)
    args['ds_config_path'] = f'dsconfig/stage_{args["stage"]}.json'
    dschf = HfDeepSpeedConfig(args['ds_config_path'])
    args['dschf'] = dschf

    build_directory(args['save_path'])
    build_directory(args['log_path'])

    if args['log_path']:
        logging.basicConfig(
            format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
            level=logging.DEBUG,
            filename=f'{args["log_path"]}/train_{time.asctime()}.log',
            filemode='w'
        )
    train_data, train_iter, sampler = load_dataset(args, args['dataset'])

    length = args['epochs'] * len(train_data) // args['world_size'] // dschf.config[
        'train_micro_batch_size_per_gpu']
    total_steps = args['epochs'] * len(train_data) // dschf.config['train_batch_size']
    args['total_steps'] = total_steps
    agent = load_model(args)
    torch.distributed.barrier()

    # begin to train
    pbar = tqdm(total=length)  # maximum total number
    current_step = 0
    for epoch_i in tqdm(range(args['epochs'])):
        # for train_iter in train_iter_list:
        for batch in train_iter:
            agent.train_model(
                batch,
                current_step=current_step,
                pbar=pbar
            )
            current_step += 1
            # if current_step % 2000 == 0:
            #     torch.distributed.barrier()
            #     agent.save_model(args['save_path'], current_step)
    # save at the end of the training
    torch.distributed.barrier()
    agent.save_model(args['save_path'], current_step)


if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    main(**args)