# NNUCB
This is a light version of the code for the paper [Neural Contextual Bandits without Regret].(https://arxiv.org/abs/2107.03144#:~:text=Contextual%20bandits%20are%20a%20rich,approximate%20the%20unknown%20reward%20function.)

## Scripts
* `data_loader.py`: Is the module that imitates a contextual bandit machine, i.e. prepares the context that is to be shown to the learner and gives reward.
In the case of optimization on graph, there is no context. So we'd need to modify it accordingly.
* `UCB_algs.py`: Contains the our algorithm classes: (C)NN-UCB and (C)NTK-UCB as in the paper and [NeuralUCB](https://arxiv.org/abs/1911.04462).
* `run_bandit.py`: Is the module that simulates the interaction between the bandit and the learner. The current code also generates a simple regret and information gain plot.
* `models.py`: To store the network models. 
* `exp_ambig.py`: Contains the code to the ambiguity experiment presented in Appendix A.3. Note that the network should be trained using the training data before running this descript. 
* `exp_underrep.py`: Contains the code to the imbalanced classes experiment given in Appendix A.3.  Note that the network should be trained using the training data before running this descript. The model should be trained using a `data_loader` with the argument `underrep = True`.

## Remarks
* I have not tried the experiment script with CNN-UCB or with (C)NTK-UCB.
## Package Requirement
I generated an automatic file with (I hope) all the relevant packages.
