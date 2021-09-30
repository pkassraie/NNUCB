from data_loader import data_loader
from UCB_algs import NNUCBDiag
import numpy as np
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from textwrap import wrap

if __name__ == '__main__':
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)

    size = 5000
    dataset = 'mnist'
    shuffle = 1
    seed = 0
    sigma = 1
    layers = 2
    beta = 1e-6
    hidden = 128
    net = 'NN'

    figure, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
    colors = ['mediumspringgreen','salmon']
    width = 0.09

    use_seed = None if seed == 0 else seed
    bandit = data_loader(dataset, is_shuffled=shuffle, seed=use_seed, underrep=True,train = False) #loading data from mnist testset
    learner = NNUCBDiag(net = net, input_dim = bandit.dim, layers=layers, hidden = hidden, beta= beta, sigma = sigma)
    learner.load_model()

    regrets = []
    total_regret = 0
    var_c00 = []
    var_c01 = []
    ucb_00 = []
    ucb_01 = []
    var_c10 = []
    var_c11 = []
    ucb_11 = []
    ucb_10 = []

    for t in range(min(size, bandit.size)):
        context, rwd = bandit.step()
        arm_select = learner.select(context)
        vars = learner.vars
        UCBs = learner.UCBs
        r = rwd[arm_select]
        reg = np.max(rwd) - r
        if rwd[0]==1: #mean the true digit is a 0, the underrep class
            var_c00.append(vars[0])
            var_c01.append(vars[1])
            ucb_00.append(UCBs[0])
            ucb_01.append(UCBs[1])
        else:
            var_c11.append(vars[1])
            var_c10.append(vars[0])
            ucb_11.append(UCBs[1])
            ucb_10.append(UCBs[0])
        total_regret+=reg
        if t % 100 == 0:
            print('t {}: R(t) {:.3f}'.format(t, total_regret))

    max_ucb = max([max([max(ucb_11), max(ucb_00)]), max([max(ucb_10), max(ucb_01)])])# + np.std(np.array(ucb_00))
    min_ucb = min([min([min(ucb_11), min(ucb_00)]), min([min(ucb_10), min(ucb_01)])])
    max_var = max([max([max(var_c11), max(var_c00)]), max([max(var_c10), max(var_c01)])]) #+ np.std(np.array(var_c00))
    min_var = min([min([min(var_c11), min(var_c00)]), min([min(var_c10), min(var_c01)])])
    bin_num = 30

    varbin = np.linspace(min_var, max_var, bin_num+1)
    histogram,_ = np.histogram(var_c00, bins = varbin)

    axes[0][0].bar(varbin[0:bin_num], histogram/np.sum(histogram), align = 'edge', width = width, label='Correct pick', alpha = 0.8, color = colors[0])
    histogram, _ = np.histogram(var_c01, bins=varbin)

    axes[0][0].bar(varbin[0:bin_num], histogram/np.sum(histogram),  align = 'edge', width = width, label='Incorrect pick', alpha=0.8, color = colors[1])
    axes[0][0].legend(prop = {'size':13})
    axes[0][0].set(xlabel=r'$\sigma^{post}$')

    histogram,_ = np.histogram(var_c11, bins = varbin)
    axes[0][1].bar(varbin[0:bin_num], histogram/np.sum(histogram),  align = 'edge', width = width, label='Correct pick', alpha=0.8, color = colors[0])
    histogram, _ = np.histogram(var_c10, bins=varbin)
    axes[0][1].bar(varbin[0:bin_num], histogram/np.sum(histogram), align = 'edge', width = width, label='Incorrect pick', alpha=0.8, color=colors[1])
    axes[0][1].set(xlabel=r'$\sigma^{post}$')
    axes[0][1].legend( prop = {'size':13})

    ucbbin = np.linspace(min_ucb, max_ucb, bin_num+1)
    histogram, _ = np.histogram(ucb_00, bins=ucbbin)

    axes[1][0].bar(ucbbin[0:bin_num],histogram/np.sum(histogram), align = 'edge', width = width,  label='Correct pick', alpha = 0.8, color = colors[0])
    histogram, _ = np.histogram(ucb_01, bins=ucbbin)
    axes[1][0].bar(ucbbin[0:bin_num],histogram/np.sum(histogram), align = 'edge', width = width,   label='Incorrect pick',
                    alpha=0.8, color = colors[1])
    axes[1][0].set(xlabel=r'$UCB$')
    axes[1][0].legend( prop = {'size':13})


    histogram, _ = np.histogram(ucb_11, bins=ucbbin)
    axes[1][1].bar(ucbbin[0:bin_num],histogram/np.sum(histogram), align = 'edge', width = width,  label ='Correct pick', alpha = 0.8,color = colors[0])
    histogram, _ = np.histogram(ucb_10, bins=ucbbin)
    axes[1][1].bar(ucbbin[0:bin_num],histogram/np.sum(histogram),align = 'edge', width = width,
                    label='Incorrect pick', alpha=0.8,color = colors[1])
    axes[1][1].legend( prop = {'size':13})
    axes[1][1].set(xlabel=r'$UCB$')

    axes[0][0].text(-0.25, 0.43, 'Post. Var.', rotation='vertical', transform=axes[0][0].transAxes)
    axes[1][0].text(-0.25, 0.43, 'UCB', rotation='vertical', transform=axes[1][0].transAxes)
    axes[0][0].text(0.2, 1.05, 'Under-represented Digits', transform=axes[0][0].transAxes)
    axes[0][1].text(0.2, 1.05,'Over-represented Digits', transform=axes[0][1].transAxes)
    figure.tight_layout()
    figure.savefig('plots/Underrep_plot.pdf') #var hist
