import sys
import yaml
import argparse
import numpy as np

import torch

import dataloader
from trainer import Trainer


def main(config_path: str, test: bool):

    np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(suppress=True)

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    device = 'cpu'
    torch.manual_seed(config['seed'])
    if config['cuda'] and torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
        torch.backends.cudnn.deterministic = True
        device = 'cuda'

    dataset = dataloader.get_dataset(config, test)
    print(
        f"Finished loading data: {config['data_name']} "
        f"for {'training' if not test else 'testing'} "
        f"with {dataset[0].shape} samples."
    )
    trainer = Trainer(config, device=device)
    return trainer.train(dataset) if not test else trainer.eval(dataset, test)


if __name__ == "__main__":
    """
    Usage: python main.py -c ../configs/__config__.yaml --test
    """

    parser = argparse.ArgumentParser(description='experimental configuration')
    parser.add_argument('-c', '--config', metavar='path', type=str,
                        required=True, help='the path to config file')
    # default false, training mode...
    parser.add_argument('--test', action='store_true', help='test mode')
    args = parser.parse_args()

    main(config_path=args.config, test=args.test)
