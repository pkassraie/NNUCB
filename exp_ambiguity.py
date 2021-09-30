from data_loader import data_loader_ambig as data_loader
from UCB_algs import NNUCBDiag
import numpy as np
import torch
import matplotlib.pyplot as plt

if __name__ == '__main__':
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)

    size = 15000
    dataset = 'mnist'
    shuffle = True
    seed = 0
    net = 'CNN'
    print_every = 100
    layers = 2
    hidden = 128
    beta = 1e-6
    sigma = 1
    UCB = NNUCBDiag
    colors = ['mediumspringgreen','salmon']
    linecolors = ['mediumseagreen','indianred']

    use_seed = None if seed == 0 else seed
    bandit = data_loader(dataset, is_shuffled=shuffle, seed=use_seed, train = False) #loading data from mnist testset
    learner = NNUCBDiag(net = net, input_dim = bandit.dim, layers=layers, hidden = hidden,beta= beta, sigma = sigma)
    learner.load_model()

    regrets = []
    ucbdif_clear = []
    ucbdif_ambig = []
    total_regret = 0

    for t in range(min(size, bandit.size)):
        context, rwd, ambig = bandit.step()
        arm_select = learner.select(context)
        UCBs = learner.UCBs
        r = rwd[arm_select]
        reg = np.max(rwd) - r
        total_regret+=reg
        regrets.append(total_regret)
        UCBs.sort()
        if ambig == 1:
            ucbdif_ambig.append(UCBs[-1] - UCBs[-2])
        else:
            ucbdif_clear.append(UCBs[-1] - UCBs[-2])
        if t % 100 == 0:
            print('t {}: R(t) {:.3f}'.format(t, total_regret))

    max_dif = max([max(ucbdif_clear), max(ucbdif_ambig)])
    min_dif = min([min(ucbdif_clear), min(ucbdif_ambig)])

    bin_num = 30
    width = 0.15
    figure = plt.subplots(figsize=( 8,6))
    ucb_bin = np.linspace(min_dif, max_dif, bin_num * 2)
    histogram,_ = np.histogram(ucbdif_clear, ucb_bin)
    histogram =  histogram/np.sum(histogram)
    ymax = 0.35
    plt.bar(ucb_bin[0:bin_num], histogram[0:bin_num], width = width, color = colors[0], alpha = 0.8, label ='Non-ambiguous Samples')
    plt.vlines(np.median(ucbdif_clear), ymin=0, ymax=ymax, linestyle='--', color=linecolors[0],
               label='Median for Non-ambiguous Samples')
    histogram, _ = np.histogram(ucbdif_ambig, ucb_bin)
    histogram = histogram / np.sum(histogram)
    ymax =  0.35
    plt.bar(ucb_bin[0:bin_num], histogram[0:bin_num], width = width, color = colors[1], alpha = 0.8, label ='Ambiguous Samples')

    plt.vlines(np.median(ucbdif_ambig), ymin = 0, ymax=ymax,linestyle = '--', color = linecolors[1], label ='Median for Ambiguous Samples')
    plt.xlabel(r'$U_{x^*_1}-U_{x^*_2}$', fontsize = 13)
    plt.legend(prop = {'size':13})
    plt.tight_layout()
    plt.savefig('plots/Ambig_UCB_Plot.pdf')
