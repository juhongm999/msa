import os
import sys
import time
import copy
import shutil

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np

from tqdm import tqdm

from model import RecurrentAttention
from utils import AverageMeter

import mlbase.utilities.mlutilities as mlu


class Trainer:
    """
    Inspired by:

    https://github.com/kevinzakka/recurrent-visual-attention

    Volodymyr Mnih, Nicolas Heess, and Alex Graves.
    Recurrent models of visual attention.
    Advances in neural information processing systems, 27, 2014.
    """

    def __init__(self, config, device):

        # build RAM model
        self.model = RecurrentAttention(
            config['g'],
            config['k'],
            config['s'],
            config['c'],
            config['h_g'],
            config['h_l'],
            config['n_hiddens'],
            config['loc_std'],
            config['n_outputs'],
            config['n_glimpses'],
            config['use_memory'],
            config['n_heads'],
            config['dim_feedforward'],
            config['dropout'],
        )
        self.device = device
        self.model.to(self.device)

        print(self.model)
        print("Total number of parameters: {}".format(
            sum(p.numel() for p in self.model.parameters()))
        )

        # initialize optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config['lr']
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 'min', patience=config['lr_patience']
        )

        self.use_memory = config['use_memory']

        # data params
        self.Xmeans = None
        self.Xstds = None
        self.Xconstant = None
        self.XstdsFixed = None
        self.standardize_x = config['standardize_x']

        # training params
        self.shuffle = config['shuffle']
        self.batch_size = config['batch_size']
        self.n_glimpses = config['n_glimpses']
        self.epochs = config['epochs']
        self.start_epoch = 0
        self.normalize_loss = config['normalize_loss']

        self.new_epoch_counter = 0
        self.train_patience = config['train_patience']

        # testing params
        self.M = config['m']
        self.n_saved_samples = config['n_saved_samples']
        self.random_loc = config.get('random_loc', False)

        # misc params
        self.loc = config.get('loc', 'random')

        self.resume = config['resume']
        self.seed = config['seed']
        self.model_name = config['model_name']
        self.best = config['best']
        self.ckpt_dir = config['ckpt_dir']
        self.plot_dir = config['plot_dir']
        self.best_valid_acc = 0.0

    def _standardizeX(self, X):
        if self.standardize_x:
            result = (X - self.Xmeans) / self.XstdsFixed
            result[:, self.Xconstant] = 0.0
            return result
        else:
            return X

    def _unstandardizeX(self, Xs):
        return (self.Xstds * Xs + self.Xmeans) if self.standardize_x else Xs

    def _setup_standardize(self, X):
        if self.Xmeans is None:
            self.Xmeans = X.mean(axis=0)
            self.Xstds = X.std(axis=0)
            self.Xconstant = self.Xstds == 0
            self.XstdsFixed = copy.copy(self.Xstds)
            self.XstdsFixed[self.Xconstant] = 1

    def _make_batches(self, X, T=None):
        if self.batch_size == -1:
            if T is None:
                yield X
            else:
                yield X, T
        else:
            for i in range(0, X.shape[0], self.batch_size):
                if T is None:
                    yield X[i:i+self.batch_size]
                else:
                    yield X[i:i+self.batch_size], T[i:i+self.batch_size]

    def train(self, dataset):
        Xtrain, Ttrain, Xval, Tval = dataset

        # load the most recent checkpoint
        if self.resume:
            self._load_checkpoint(best=False)

        self._setup_standardize(Xtrain)
        Xtrain = self._standardizeX(Xtrain)
        Xval = self._standardizeX(Xval)

        self.n_train_batches = (
            Xtrain.shape[0] + self.batch_size - 1) // self.batch_size

        for epoch in range(self.start_epoch, self.epochs):
            # shuffle training data
            if self.shuffle:
                torch.manual_seed(self.seed + epoch)
                train_inds = torch.randperm(Xtrain.size()[0])
                Xtrain = Xtrain[train_inds]
                Ttrain = Ttrain[train_inds]

            # train for 1 epoch
            train_loss, train_acc = self._train_one_epoch(
                Xtrain, Ttrain, epoch)

            # evaluate on validation set
            valid_loss, valid_acc = self.eval((Xval, Tval), epoch=epoch)

            # reduce lr if validation loss plateaus
            self.scheduler.step(-valid_acc)

            is_best = valid_acc > self.best_valid_acc
            msg1 = "train loss: {:.3f} - train acc: {:.3f} "
            msg2 = "- val loss: {:.3f} - val acc: {:.3f} - val err: {:.3f}"
            if is_best:
                self.new_epoch_counter = 0
                msg2 += " [*]"
            msg = msg1 + msg2
            print(
                msg.format(
                    train_loss, train_acc, valid_loss, valid_acc, 100 - valid_acc
                )
            )
            if not is_best:
                self.new_epoch_counter += 1
            if self.new_epoch_counter > self.train_patience:
                print("[!] No improvement in a while, stopping training.")
                return self.best_valid_acc

            # check for improvement
            self.best_valid_acc = max(valid_acc, self.best_valid_acc)
            self._save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_state": self.model.state_dict(),
                    "optim_state": self.optimizer.state_dict(),
                    "best_valid_acc": self.best_valid_acc,
                    "x_standardize": (self.Xmeans, self.Xstds, self.Xconstant, self.XstdsFixed)
                },
                is_best,
            )
        print('Finished Training.')

        return self.best_valid_acc

    def _compute_loss(self, T, a_t, baselines, log_pi,
                      monte_carlo=False):

        # classification probs
        log_probs = F.log_softmax(a_t, dim=1)

        baselines = torch.stack(baselines).transpose(1, 0)
        log_pi = torch.stack(log_pi).transpose(1, 0)

        if monte_carlo:
            log_probs = log_probs.view(self.M, -1, log_probs.shape[-1])
            log_probs = torch.mean(log_probs, dim=0)

            baselines = baselines.contiguous().view(
                self.M, -1, baselines.shape[-1])
            baselines = torch.mean(baselines, dim=0)

            log_pi = log_pi.contiguous().view(self.M, -1, log_pi.shape[-1])
            log_pi = torch.mean(log_pi, dim=0)

        Y = torch.max(log_probs, 1)[1]

        # p.6 of Mnih et al. 2014,
        # reward at last time step was 1 if classified correctly, 0 otherwise.
        # The rewards for all other timesteps were 0.
        # Note: this takes a long time to converge if you use it.
        # R = torch.zeros((T.shape[0], self.n_glimpses), device=self.device)
        # R[:, -1] = (Y.detach() == T).float()

        R = (Y.detach() == T).float()
        R = R.unsqueeze(1).repeat(1, self.n_glimpses)

        if self.normalize_loss:
            R_mean = torch.mean(R)
            R_std = torch.std(R)

        if self.normalize_loss:  # normalize baselines
            bl_mean = torch.mean(baselines)
            bl_std = torch.std(baselines)
            baselines = (baselines - bl_mean) / \
                (bl_std + torch.finfo(torch.float32).eps)
            baselines = baselines * R_std + R_mean

        # compute reinforce loss
        advantage = R - baselines.detach()
        if self.normalize_loss:  # normalize advantage
            adv_mean = torch.mean(advantage)
            adv_std = torch.std(advantage)
            advantage = (advantage - adv_mean) / \
                (adv_std + torch.finfo(torch.float32).eps)
        # summed weighted log likelihood over timesteps
        loss_reinforce = torch.sum(-log_pi * advantage, dim=1)
        # expected mean across 'episodes', i.e. the batch
        loss_reinforce = torch.mean(loss_reinforce, dim=0)

        # compute losses for differentiable modules
        if self.normalize_loss:  # normalize rewards for baseline loss
            R = (R - R_mean) / (R_std + torch.finfo(torch.float32).eps)
        loss_action = F.nll_loss(log_probs, T)
        loss_baseline = F.mse_loss(baselines, R)

        # sum up into a hybrid loss
        # p.5 of Mnih et al. 2014, location network trained with REINFORCE.
        # p.5 of Mnih et al. 2014, apply MSE loss to baseline.
        # p.4/5 of Mnih et al. 2014, apply action loss + reinforce loss to glimpse
        #                            network, core network, and action network
        loss = loss_action + loss_reinforce * 0.01 + loss_baseline
        if torch.isnan(loss):
            print('loss is nan')
            sys.exit()
        return Y, loss

    def _train_one_epoch(self, Xtrain, Ttrain, epoch):

        batch_time = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()

        self.model.train()

        with tqdm(total=Xtrain.shape[0], position=0, leave=True) as pbar:
            for i, (X, T) in enumerate(self._make_batches(Xtrain, Ttrain)):
                start_t = time.time()
                self.optimizer.zero_grad()

                # overlapping transfer if pinned memory
                X = X.to(self.device, non_blocking=True)
                T = T.to(self.device, non_blocking=True)

                h_t, l_t = self.model.reset(X.shape[0], self.device)

                # extract the glimpses
                log_pi = []
                baselines = []
                for t in range(self.n_glimpses - 1):
                    h_t, l_t, b_t, p = self.model(X, l_t, h_t)

                    baselines.append(b_t)
                    log_pi.append(p)

                # last iteration
                h_t, l_t, b_t, a_t, _, p = self.model(
                    X, l_t, h_t, last=True)
                log_pi.append(p)
                baselines.append(b_t)

                Y, loss = self._compute_loss(
                    T, a_t, baselines, log_pi)

                acc = 100 * ((Y == T).float().sum() / len(T))

                losses.update(loss.item(), X.size()[0])
                accs.update(acc.item(), X.size()[0])

                # update
                loss.backward()
                self.optimizer.step()

                end_t = time.time()
                batch_time.update(end_t - start_t)

                pbar.set_description(
                    (
                        f"Epoch: {epoch + 1} [{i + 1}/{self.n_train_batches}] "
                        f"Time {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                        f"Loss {losses.val:.4f} ({losses.avg:.4f}) "
                        f"Acc {accs.val:.3f} ({accs.avg:.3f})"
                    )
                )
                pbar.update(X.shape[0])

            return losses.avg, accs.avg

    @torch.no_grad()
    def eval(self, dataset, test=False, epoch=0):
        """Evaluate the RAM model on the validation set.
        """
        Xval, Tval = dataset

        if test:
            self._load_checkpoint(best=self.best)
            Xval = self._standardizeX(Xval)

        losses = AverageMeter()
        accs = AverageMeter()

        if self.n_saved_samples == len(Xval) or self.n_saved_samples == -1:
            torch.manual_seed(self.seed)
            plot_inds = torch.arange(len(Xval))
        else:
            plot_inds = []
            # TODO: allow for duplicates from each class
            for t in torch.unique(Tval):
                inds = (Tval == t).nonzero().flatten()
                # this will reset the model.reset seed
                torch.manual_seed(self.seed)  # + 1)
                plot_inds.append(inds[torch.randperm(len(inds))[:1]])
                if len(plot_inds) == self.n_saved_samples:
                    break
            plot_inds = torch.cat(plot_inds)

        # torch.manual_seed(self.seed)
        # plot_inds = torch.randperm(Xval.shape[0])[:self.n_saved_samples]

        saved_imgs, saved_locs, saved_attn = [], [], []
        saved_output = []
        start_i = 0

        if test:
            Ys = []
            GYs = []

        self.model.eval()
        for i, (X, T) in enumerate(self._make_batches(Xval, Tval)):
            X, T = X.to(self.device), T.to(self.device)

            # duplicate M times
            if self.M > 1:
                X = X.repeat(self.M, 1, 1, 1)

            h_t, l_t = self.model.reset(X.shape[0], self.device, loc=self.loc)

            locs = []
            log_pi = []
            baselines = []
            locs.append(l_t)
            if test:
                G = []
            for t in range(self.n_glimpses - 1):

                if test:  # get acc for each glimpse
                    h_t, l_t, b_t, a_t, attention, p = self.model(
                        X, l_t, h_t, last=True)
                    G.append(torch.max(F.log_softmax(a_t, dim=1), 1)[1])
                else:
                    h_t, l_t, b_t, p = self.model(X, l_t, h_t)

                if self.random_loc:
                    l_t = torch.FloatTensor(X.shape[0], 2).uniform_(-1, 1).to(
                        self.device)

                baselines.append(b_t)
                log_pi.append(p)
                locs.append(l_t)

            # last iteration
            h_t, l_t, b_t, a_t, attention, p = self.model(
                X, l_t, h_t, last=True)
            log_pi.append(p)
            baselines.append(b_t)

            Y, loss = self._compute_loss(
                T, a_t, baselines, log_pi, monte_carlo=self.M > 1)

            if test:
                Ys.append(Y)
                G.append(torch.max(F.log_softmax(a_t, dim=1), 1)[1])
                GYs.append(torch.stack(G))

            acc = 100 * ((Y == T).float().sum() / len(T))

            losses.update(loss.item(), X.size()[0])
            accs.update(acc.item(), X.size()[0])

            # save data for plotting, based on inds in plot_inds (true index of img)
            if self.n_saved_samples > 0:
                adjusted_inds = plot_inds - start_i
                adjusted_inds = adjusted_inds[(adjusted_inds >= 0) & (
                    adjusted_inds < T.shape[0])] * self.M
                if len(adjusted_inds) > 0:
                    if adjusted_inds.dim() == 0:
                        adjusted_inds = adjusted_inds.unsqueeze(0)
                    saved_imgs.append(self._unstandardizeX(
                        X[adjusted_inds].cpu()).numpy())
                    saved_locs.append(torch.stack(locs).transpose(
                        1, 0)[adjusted_inds].cpu().numpy())
                    if self.use_memory:
                        saved_attn.append(
                            attention[adjusted_inds].cpu().numpy())
                    saved_output.append(torch.stack(
                        [T[adjusted_inds], Y[adjusted_inds]], dim=1).cpu().numpy())

            start_i += T.shape[0]

        if test:
            Ys = torch.cat(Ys)
            GYs = torch.hstack(GYs).cpu().numpy()
            print(repr((Tval.numpy() == GYs).sum(axis=1) / Tval.shape[0]))
            mlu.evaluate(Ys.cpu().numpy(), Tval.numpy(), verbose=True)
            print(
                f"[*] test loss: {losses.avg:.3f} - "
                f"test acc: {accs.avg:.3f} - test err: {100 - accs.avg:.3f}"
            )
        if self.n_saved_samples > 0:
            output_npz_file = 'test.npz' if test else f'val_{epoch + 1}.npz'

            saved_data = dict()
            saved_data['imgs'] = np.vstack(saved_imgs)
            saved_data['locs'] = np.vstack(saved_locs)
            if self.use_memory:
                saved_data['attn'] = np.vstack(saved_attn)
            saved_data['out'] = np.vstack(saved_output)

            os.makedirs(os.path.join(self.plot_dir,
                        self.model_name), exist_ok=True)
            np.savez(os.path.join(self.plot_dir, self.model_name,
                     output_npz_file), **saved_data)

        return losses.avg, accs.avg

    def _save_checkpoint(self, state, is_best):
        """Saves a checkpoint of the model.
        If this model has reached the best validation accuracy thus
        far, a seperate file with the suffix `best` is created.
        """
        filename = self.model_name + "_ckpt.pth.tar"
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        torch.save(state, ckpt_path)
        if is_best:
            filename = self.model_name + "_model_best.pth.tar"
            shutil.copyfile(ckpt_path, os.path.join(self.ckpt_dir, filename))

    def _load_checkpoint(self, best=False):
        """Load the best copy of a model.
        This is useful for 2 cases:
        - Resuming training with the most recent model checkpoint.
        - Loading the best validation model to evaluate on the test data.
        Args:
            best: if set to True, loads the best model.
                Use this if you want to evaluate your model
                on the test data. Else, set to False in which
                case the most recent version of the checkpoint
                is used.
        """
        print("[*] Loading model from {}".format(self.ckpt_dir))

        filename = self.model_name + "_ckpt.pth.tar"
        if best:
            filename = self.model_name + "_model_best.pth.tar"
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        ckpt = torch.load(ckpt_path)

        # load variables from checkpoint
        self.start_epoch = ckpt["epoch"]
        self.best_valid_acc = ckpt["best_valid_acc"]
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])
        self.Xmeans, self.Xstds, self.Xconstant, self.XstdsFixed = ckpt["x_standardize"]

        if best:
            print(
                "[*] Loaded {} checkpoint @ epoch {} "
                "with best valid acc of {:.3f}".format(
                    filename, ckpt["epoch"], ckpt["best_valid_acc"]
                )
            )
        else:
            print("[*] Loaded {} checkpoint @ epoch {}".format(filename, ckpt["epoch"]))
