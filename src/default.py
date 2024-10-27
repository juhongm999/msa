import sys
import argparse
import numpy as np
import torch

import dataloader

from vit_pytorch import ViT

from mlbase.networks.pytorch import neuralnetworks as nn
import mlbase.utilities.mlutilities as mlu


def _make_batches(batch_size, X, T=None):
    if batch_size == -1:
        if T is None:
            yield X
        else:
            yield X, T
    else:
        for i in range(0, X.shape[0], batch_size):
            if T is None:
                yield X[i:i+batch_size]
            else:
                yield X[i:i+batch_size], T[i:i+batch_size]


def main(args):
    np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(suppress=True)

    config = {}
    config['data_name'] = args.dataset

    if args.dataset == 'mnist':
        config['data_dir'] = '/s/chopin/l/grad/stock/nvme/data/cs/'
    elif args.dataset == 'cluttered':
        config['data_dir'] = '/s/chopin/l/grad/stock/nvme/data/cs/cluttered-mnist'
    elif args.dataset == 'tc':
        config['data_dir'] = '/s/chopin/l/grad/stock/nvme/data/ai2es/mlhub/nasa_tc'
    elif args.dataset == 'intel':
        config['data_dir'] = '/s/chopin/l/grad/stock/nvme/data/cs/intel-img/'
    elif args.dataset == 'cifar':
        config['data_dir'] = '/s/chopin/l/grad/stock/nvme/data/cs/cifar-10-batches-py'

    config['seed'] = 1234  # 1235, 1236

    torch.manual_seed(config['seed'])
    device = 'cpu'
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
        torch.backends.cudnn.deterministic = True
        device = 'cuda'

    Xtrain, Ttrain, Xval, Tval, Xtest, Ttest = dataloader.get_dataset(
        config, False, all=True)

    epochs = 20

    if args.model in ['fc', 'cnn']:

        if args.model == 'fc':
            n_hiddens_list = [256]*2
            conv_layers = None
            if args.dataset == 'cluttered':
                epochs = 10

        elif args.model == 'cnn':
            n_hiddens_list = [32]*1

            if args.dataset == 'mnist':
                conv_layers = [{'n_units': 16, 'shape': 3},
                               {'n_units': 32, 'shape': 3}]
                epochs = 20

            elif args.dataset == 'cluttered':
                conv_layers = [{'n_units': 8, 'shape': 3},
                               {'n_units': 16, 'shape': 3},
                               {'n_units': 32, 'shape': 3},
                               {'n_units': 64, 'shape': 3}]
                epochs = 30

            elif args.dataset == 'tc':
                conv_layers = [{'n_units': 8, 'shape': 3},
                               {'n_units': 16, 'shape': 3},
                               {'n_units': 32, 'shape': 3},
                               {'n_units': 64, 'shape': 3}]
                epochs = 30

            elif args.dataset == 'intel':
                conv_layers = [{'n_units': 8, 'shape': 3},
                               {'n_units': 16, 'shape': 3},
                               {'n_units': 32, 'shape': 3},
                               {'n_units': 64, 'shape': 3}]
                epochs = 30

            elif args.dataset == 'cifar':
                conv_layers = [{'n_units': 8, 'shape': 3},
                               {'n_units': 16, 'shape': 3},
                               {'n_units': 32, 'shape': 3},
                               {'n_units': 64, 'shape': 3}]
                epochs = 30

        nnet = nn.NeuralNetworkClassifier(Xtrain.shape[1:], n_hiddens_list, len(
            np.unique(Ttrain)), conv_layers, activation_f='tanh', use_gpu=True, seed=config['seed'])

        nnet.train(Xtrain, Ttrain, validation_data=(Xval, Tval),
                   n_epochs=epochs, batch_size=32, learning_rate=0.0003, opt='adam',
                   standardize_x=False, shuffle=True, early_stopping=False, verbose=True)

        Ytest = nnet.use(Xtest)

        mlu.evaluate(Ytest, Ttest, verbose=True)

    elif args.model == 'vit':

        Xtrain, Ttrain = dataloader._convert_to_tensor(Xtrain, Ttrain)
        Xval, Tval = dataloader._convert_to_tensor(Xval, Tval)
        Xtest, Ttest = dataloader._convert_to_tensor(Xtest, Ttest)

        if args.dataset == 'mnist':
            patch_size = 7
            mlp_dim = 256
            epochs = 20
            heads = 4

        elif args.dataset == 'cluttered':
            patch_size = 12
            mlp_dim = 256
            epochs = 150
            heads = 4

        elif args.dataset == 'tc':
            patch_size = 16
            mlp_dim = 256
            epochs = 50
            heads = 4

        elif args.dataset == 'intel':
            patch_size = 16
            mlp_dim = 256
            epochs = 50
            heads = 4

        model = ViT(
            image_size=Xtrain.shape[-1],
            patch_size=patch_size,
            num_classes=len(np.unique(Ttrain)),
            channels=Xtrain.shape[1],
            dim=256,
            depth=1,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=0.1,
            emb_dropout=0.1
        )
        print(model)
        print(
            f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
        criterion = torch.nn.CrossEntropyLoss()

        # hyperparameters
        shuffle = True
        batch_size = 128

        n_train_batches = (
            Xtrain.shape[0] + batch_size - 1) // batch_size
        n_val_batches = (
            Xval.shape[0] + batch_size - 1) // batch_size
        for epoch in range(epochs):
            if shuffle:
                torch.manual_seed(config['seed'] + epoch)
                train_inds = torch.randperm(Xtrain.size()[0])
                Xtrain = Xtrain[train_inds]
                Ttrain = Ttrain[train_inds]

            epoch_loss = 0
            epoch_accuracy = 0

            # train for 1 epoch
            model.train()
            for i, (X, T) in enumerate(_make_batches(batch_size, Xtrain, Ttrain)):
                optimizer.zero_grad()

                # overlapping transfer if pinned memory
                X = X.to(device, non_blocking=True)
                T = T.to(device, non_blocking=True)

                Y = model(X)
                loss = criterion(Y, T)
                # update
                loss.backward()
                optimizer.step()

                acc = (Y.argmax(dim=1) == T).float().mean()
                epoch_accuracy += acc / n_train_batches
                epoch_loss += loss / n_train_batches

            epoch_val_accuracy = 0
            epoch_val_loss = 0

            # evaluate on validation set
            model.eval()
            with torch.no_grad():
                epoch_val_accuracy = 0
                epoch_val_loss = 0
                for i, (X, T) in enumerate(_make_batches(batch_size, Xval, Tval)):
                    # overlapping transfer if pinned memory
                    X = X.to(device, non_blocking=True)
                    T = T.to(device, non_blocking=True)

                    Y = model(X)
                    val_loss = criterion(Y, T)

                    acc = (Y.argmax(dim=1) == T).float().mean()
                    epoch_val_accuracy += acc / n_val_batches
                    epoch_val_loss += val_loss / n_val_batches

            print(
                f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy*100:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy*100:.4f}"
            )

        # test set
        model.eval()
        with torch.no_grad():
            Ytest = []
            for i, (X, T) in enumerate(_make_batches(batch_size, Xtest, Ttest)):
                # overlapping transfer if pinned memory
                X = X.to(device, non_blocking=True)
                T = T.to(device, non_blocking=True)
                Y = model(X)
                Ytest.append(Y.argmax(dim=1).cpu().numpy())
            Ytest = np.hstack(Ytest).reshape(-1, 1)
            mlu.evaluate(Ytest, Ttest, verbose=True)


if __name__ == '__main__':
    """
    Usage:
    python default.py --dataset mnist --model fc
    nohup python default.py --dataset mnist --model fc >/dev/null 2>&1 &
    """

    parser = argparse.ArgumentParser(description='hyperparameter runner')
    parser.add_argument('-d', '--dataset', metavar='path', type=str,
                        required=True, choices=['mnist', 'cluttered', 'tc', 'intel'],
                        help='dataset to use')
    parser.add_argument('-m', '--model', metavar='path', type=str,
                        required=True, choices=['fc', 'cnn', 'vit'],
                        help='model to train')

    args = parser.parse_args()

    main(args)
