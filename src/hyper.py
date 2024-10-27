import os
import sys
import time
import argparse
import itertools

from pathlib import Path
from multiprocessing import Pool
from functools import partial
import main as runner


def log_result(result, results_f):
    # modified only by the main process, not the pool workers.
    with open(results_f, 'a') as f:
        f.write(','.join(map(str, result)) + '\n')


def start(config_file, log_dir):
    # redirect stdout for given process/experiment to unique file
    model_name = Path(config_file).stem
    sys.stdout = open(os.path.join(log_dir, model_name + '_' + str(
        os.getpid()) + '.out'), 'w', buffering=1)
    sys.stderr = open(os.devnull, 'w')

    return model_name, runner.main(config_file, test=False)


def run(args):
    dataset = args.dataset
    log_dir = os.path.join(args.log_dir, dataset)
    config_dir = os.path.join(args.config_dir, dataset)
    os.makedirs(log_dir, exist_ok=True)

    results_f = os.path.join(
        log_dir, ('ram' if args.ram else 'ours') + '_results.out')
    open(results_f, 'w').close()

    config_files = [os.path.join(config_dir, f)
                    for f in os.listdir(config_dir) if f.endswith('.yaml') and
                    ('ram' if args.ram else 'ours') in f]

    pool = Pool(args.pool_size)
    for conf in config_files:
        pool.apply_async(start, args=(
            conf, log_dir, ), callback=partial(log_result, results_f=results_f))
    pool.close()
    pool.join()


def generate(args):
    dataset = args.dataset
    config_dir = os.path.join(args.config_dir, dataset)
    os.makedirs(config_dir, exist_ok=True)

    data = dict()

    #!!! uber default
    # misc params
    data['best'] = True
    data['resume'] = False
    data['seed'] = 1234
    data['cuda'] = True
    # testing params
    data['m'] = 1
    data['n_saved_samples'] = 0

    if dataset == 'mnist':

        #!!! default
        # data params
        data['data_name'] = dataset
        data['data_dir'] = '/s/chopin/l/grad/stock/nvme/data/cs/'
        data['ckpt_dir'] = '../ckpt/hyper/' + dataset
        data['plot_dir'] = None
        data['standardize_x'] = False
        data['n_outputs'] = 10
        data['c'] = 1

        if args.ram:
            # network params
            g = [8]
            k = [1]
            s = [1]
            h_g = [128, 256]
            h_l = [64, 128]
            n_hiddens = [128, 256]
            loc_std = [0.1, 0.05, 0.01]

            # memory params
            data['use_memory'] = False
            n_heads = [0]
            dim_feedforward = [0]
            dropout = [0]

            # training params
            epochs = [300]
            n_glimpses = [6]
            lr = [0.001, 0.0003]
            lr_patience = [20]
            batch_size = [128]
            data['shuffle'] = True
            train_patience = [40]
            data['normalize_loss'] = False

        else:
            # network params
            g = [8]
            k = [1]
            s = [1]
            h_g = [128, 256]
            h_l = [64, 128]
            n_hiddens = [256]
            loc_std = [0.1, 0.05]

            # memory params
            data['use_memory'] = True
            n_heads = [1, 2, 4]
            dim_feedforward = [128, 256]
            dropout = [0.1]

            # training params
            epochs = [350]
            n_glimpses = [6]
            lr = [0.0003]
            lr_patience = [20]
            batch_size = [128]
            data['shuffle'] = True
            train_patience = [40]
            data['normalize_loss'] = False

    elif dataset == 'cluttered':
        #!!! default
        # data params
        data['data_name'] = dataset
        data['data_dir'] = '/s/chopin/l/grad/stock/nvme/data/cs/cluttered-mnist'
        data['ckpt_dir'] = '../ckpt/hyper/' + dataset
        data['plot_dir'] = None
        data['standardize_x'] = False
        data['n_outputs'] = 10
        data['c'] = 1

        if args.ram:
            # network params
            g = [12]
            k = [3]
            s = [2]
            h_g = [256, 512]
            h_l = [64, 128]
            n_hiddens = [256, 512]
            loc_std = [0.1, 0.05]

            # memory params
            data['use_memory'] = False
            n_heads = [0]
            dim_feedforward = [0]
            dropout = [0]

            # training params
            epochs = [400]
            n_glimpses = [6]
            lr = [0.0003]
            lr_patience = [20]
            batch_size = [128]
            data['shuffle'] = True
            train_patience = [40]
            data['normalize_loss'] = False

        else:
            # network params
            g = [12]
            k = [3]
            s = [2]
            h_g = [256, 512]
            h_l = [64]
            n_hiddens = [256, 512]
            loc_std = [0.1]

            # memory params
            data['use_memory'] = True
            n_heads = [1, 2, 4]
            dim_feedforward = [256]
            dropout = [0.1]

            # training params
            epochs = [400]
            n_glimpses = [6]
            lr = [0.0003]
            lr_patience = [20]
            batch_size = [128]
            data['shuffle'] = True
            train_patience = [40]
            data['normalize_loss'] = False

    elif dataset == 'tc':
        #!!! default
        # data params
        data['data_name'] = dataset
        data['data_dir'] = '/s/chopin/l/grad/stock/nvme/data/ai2es/mlhub/nasa_tc'
        data['ckpt_dir'] = '../ckpt/hyper/' + dataset
        data['plot_dir'] = None
        data['standardize_x'] = False
        data['n_outputs'] = 3
        data['c'] = 1

        if args.ram:
            # network params
            g = [12, 18]
            k = [3]
            s = [2]
            h_g = [512]
            h_l = [64]
            n_hiddens = [512]
            loc_std = [0.1]

            # memory params
            data['use_memory'] = False
            n_heads = [0]
            dim_feedforward = [0]
            dropout = [0]

            # training params
            epochs = [400]
            n_glimpses = [6]
            lr = [0.0003]
            lr_patience = [20]
            batch_size = [64]
            data['shuffle'] = True
            train_patience = [60]
            data['normalize_loss'] = False

        else:
            # network params
            g = [12]
            k = [3]
            s = [2]
            h_g = [256]
            h_l = [64]
            n_hiddens = [256, 512]
            loc_std = [0.1]

            # memory params
            data['use_memory'] = True
            n_heads = [2, 4, 8]
            dim_feedforward = [128, 256]
            dropout = [0.1]

            # training params
            epochs = [400]
            n_glimpses = [8]
            lr = [0.0003]
            lr_patience = [20]
            batch_size = [128]
            data['shuffle'] = True
            train_patience = [40]
            data['normalize_loss'] = False

    elif dataset == 'intel':
        #!!! default
        # data params
        data['data_name'] = dataset
        data['data_dir'] = '/s/chopin/l/grad/stock/nvme/data/cs/intel-img/'
        data['ckpt_dir'] = '../ckpt/hyper/' + dataset
        data['plot_dir'] = None
        data['standardize_x'] = False
        data['n_outputs'] = 6
        data['c'] = 3

        if args.ram:
            # network params
            g = [16]
            k = [3]
            s = [2]
            h_g = [512, 1024]
            h_l = [64]
            n_hiddens = [512, 1024]
            loc_std = [0.1]

            # memory params
            data['use_memory'] = False
            n_heads = [0]
            dim_feedforward = [0]
            dropout = [0]

            # training params
            epochs = [400]
            n_glimpses = [8]
            lr = [0.0003]
            lr_patience = [20]
            batch_size = [128]
            data['shuffle'] = True
            train_patience = [60]
            data['normalize_loss'] = False

        else:
            # network params
            g = [16]
            k = [3]
            s = [2]
            h_g = [512]
            h_l = [64]
            n_hiddens = [512]
            loc_std = [0.1]

            # memory params
            data['use_memory'] = True
            n_heads = [4, 8, 16]
            dim_feedforward = [256, 512]
            dropout = [0.1]

            # training params
            epochs = [400]
            n_glimpses = [8]
            lr = [0.0003]
            lr_patience = [20]
            batch_size = [128]
            data['shuffle'] = True
            train_patience = [40]
            data['normalize_loss'] = False

    else:
        raise ValueError('invalid dataset')

    for i, d in enumerate(itertools.product(*[g, k, s, h_g, h_l, n_hiddens, loc_std, n_heads,
                                              dim_feedforward, dropout, epochs, n_glimpses,
                                              lr, lr_patience, batch_size, train_patience])):
        data['model_name'] = ('ram' if args.ram else 'ours') + \
            '_' + str(i).zfill(3)
        data['g'] = d[0]
        data['k'] = d[1]
        data['s'] = d[2]
        data['h_g'] = d[3]
        data['h_l'] = d[4]
        data['n_hiddens'] = d[5]
        data['loc_std'] = d[6]
        data['n_heads'] = d[7]
        data['dim_feedforward'] = d[8]
        data['dropout'] = d[9]
        data['epochs'] = d[10]
        data['n_glimpses'] = d[11]
        data['lr'] = d[12]
        data['lr_patience'] = d[13]
        data['batch_size'] = d[14]
        data['train_patience'] = d[15]

        # write configs to yaml file under config_dir
        with open(os.path.join(config_dir, data['model_name'] + '.yaml'), 'w') as f:
            for k, v in data.items():
                f.write(k + ': ' + str(v) + '\n')


def main(args):
    if args.generate:
        generate(args)
    else:
        run(args)


if __name__ == '__main__':
    """
    Usage: 
    python hyper.py --dataset mnist --generate --ram
    nohup python hyper.py --dataset mnist --ram >/dev/null 2>&1 &
    """

    parser = argparse.ArgumentParser(description='hyperparameter runner')
    parser.add_argument('-d', '--dataset', metavar='path', type=str,
                        required=True, choices=['mnist', 'cluttered', 'tc', 'intel'],
                        help='dataset to use')
    parser.add_argument('--log-dir', metavar='path', type=str,
                        default='../logs/hyper', help='log directory')
    parser.add_argument('--config-dir', metavar='path', type=str,
                        default='../configs/hyper', help='configs directory')
    parser.add_argument('--generate', action='store_true',
                        help='generate files')
    parser.add_argument('-p', '--pool-size', nargs='?',
                        const=4, type=int, default=4, help='number of processes')
    parser.add_argument('--ram', action='store_true',
                        help='params for RAM')
    args = parser.parse_args()

    main(args)
