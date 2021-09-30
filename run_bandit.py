import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

from data_loader import data_loader
import matplotlib.pyplot as plt
import pandas as pd
from UCB_algs import NNUCBDiag, NTKUCB
from utils import fit_sublinear
import time
import numpy as np

T = 5000
dataset = 'mnist'
shuffle = 1
seed = 0
n_arm = 10
net = 'NN'
hidden = 128 # number of hidden layers
layers = 2
betas = [ 1e-04]
sigmas = [1.0]
UCB = NTKUCB
T0 =200 #Before T0 training is done after every round, after T0 it is done in batches
batch_size = 100
print_every = 10


print(f'Working on {UCB.__name__} algorithm.')

figure, axes = plt.subplots(nrows = 1, ncols = 2 ,figsize = (16,6))
data = pd.DataFrame(np.arange(T), columns=['Steps'])
total_settings = len(betas)*len(sigmas)
setting = 1
for beta in betas:
    for sigma in sigmas:
        print(f'Setting {setting}/{total_settings}. Trying beta = {beta}, sigma = {sigma}')
        start_time = time.time()
        use_seed = None if seed == 0 else seed
        bandit = data_loader(dataset, is_shuffled=shuffle, seed=use_seed)
        learner = UCB(net = net, input_dim = bandit.dim, n_arm = n_arm, layers=layers,hidden = hidden, beta= beta, sigma = sigma)
        regrets = []
        I = []
        reg_sum = 0
        new_context = []
        new_reward = []
        for t in range(min(T, bandit.size)):
            context, rwd = bandit.step()
            arm_select = learner.select(context)
            r = rwd[arm_select]
            reg = np.max(rwd) - r
            reg_sum += reg
            if t < T0:
                loss = learner.train(context[arm_select], r)
            else:  # After some time just train in batches
                # save the new datapoints
                if len(new_reward) > 0:
                    new_reward.append(r)
                    new_context.append(context[arm_select])
                else:
                    new_reward = [r]
                    new_context = [context[arm_select]]
                # when there's enough, update the GP
                if t % batch_size == 0:
                    loss = learner.train(new_context, new_reward)
                    new_context = []  # remove from unused points
                    new_reward = []

            regrets.append(reg_sum)
            if t % print_every == 0: #only calculate I once every 100 steps.
                I_T = learner.get_infogain()
                print('At step {}: Regret {:.3f} - Information Gain {:.3f}'.format(t+1, reg_sum, I_T))
                I.append(I_T)
            else:
                I.append(-1)
        end_time = time.time()
        print(f'{learner.name} with {T} steps takes {(end_time - start_time)/60} mins.')
        ucb_info = 'beta{:.3f}_sigma{:.6f}'.format(beta, sigma)
        data[ucb_info] = regrets
        data['I_'+ucb_info] = I

        _, coefs = fit_sublinear(np.arange(T) + 1, np.array(regrets)+1, with_coefs=True)
        axes[1].plot(regrets, label='{}: T^({:.3f})'.format(ucb_info, coefs[0]))
        I = np.array(I)
        I = I[I>0]
        _, coefs = fit_sublinear(np.arange(I.shape[0]) + 1, I, with_coefs=True)
        axes[0].plot(I, label='{}: T^({:.3f})'.format(ucb_info, coefs[0]))

        learner.save_model()
        setting += 1


name = '{}_{}L_{}m_{}T'.format( UCB.__name__,layers, hidden, T)
data.to_csv(f'hypersearches/{name}.csv')

plt.rcParams.update({'axes.labelsize': 'large'})
axes[0].set(xlabel = 'T', ylabel = r'$I(y_T;f_T)$')
axes[1].set(xlabel = 'T', ylabel = r'$R_T$')
axes[0].legend(bbox_to_anchor=(0.1,0.85), loc='upper left',prop = {'size':13})
axes[1].legend(bbox_to_anchor=(0.1, 0.85), loc='upper left',prop = {'size':13})
axes[0].set_title('Information Gain')
axes[1].set_title('Regret')
figure.tight_layout()
plt.savefig(f'plots/{name}.pdf')