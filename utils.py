import shutil
import json

import torch
import numpy as np

from matplotlib import pyplot as plt


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name="", fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def save_config(opt, file_path):
    with open(file_path, "w") as f:
        json.dump(opt.__dict__, f, indent=2)


def load_config(opt, file_path):
    with open(file_path, "r") as f:
        opt.__dict__ = json.load(f)


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar", prefix=""):
    tries = 15
    error = None

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            torch.save(state, prefix + filename)
            if is_best:
                shutil.copyfile(prefix + filename, prefix + "model_best.pth.tar")
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        print("model save {} failed, remaining {} trials".format(filename, tries))
        if not tries:
            raise error


def save_loss_for_split(state, filename="loss_for_split.pth.tar", prefix=""):
    tries = 15
    error = None

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            torch.save(state, prefix + filename)
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        print("model save {} failed, remaining {} trials".format(filename, tries))
        if not tries:
            raise error


def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def plot_gmm(epoch, gmm, X, clean_index, save_path=''):
    plt.clf()
    ax = plt.gca()

    # Compute PDF of whole mixture
    x = np.linspace(0, 1, 1000)
    logprob = gmm.score_samples(x.reshape(-1, 1))
    pdf = np.exp(logprob)

    # Compute PDF for each component
    responsibilities = gmm.predict_proba(x.reshape(-1, 1))
    pdf_individual = responsibilities * pdf[:, np.newaxis]

    # Plot data histogram
    ax.hist(X[clean_index == 1], bins=100, density=True, histtype='stepfilled', color='green', alpha=0.3,
            label='Clean Samples')
    ax.hist(X[clean_index == 0], bins=100, density=True, histtype='stepfilled', color='red', alpha=0.3, label='Noisy Samples')

    font1 = {'family': 'DejaVu Sans',
             'weight': 'normal',
             'size': 13,
             }

    if epoch == 10:
        # Plot PDF of whole model
        ax.plot(x, pdf, '-k', label='Mixture PDF')

        # Plot PDF of each component
        ax.plot(x, pdf_individual[:, 0], '--', label='Component A', color='green')
        ax.plot(x, pdf_individual[:, 1], '--', label='Component B', color='red')

    # ax.set_xlabel('Per-sample loss, epoch {}'.format(epoch), font1)
    ax.set_xlabel('Per-sample loss', font1)
    ax.set_ylabel('Density', font1)
    x_ticks = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xticks(x_ticks)
    plt.tick_params(labelsize=11)

    ax.legend(loc='upper right', prop=font1)

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
