# NNUCB
This is a light version of the code for our paper "Neural Contextual Bandits without Regret"

## Scripts
* `data_loader.py`: Is the module that imitates a contextual bandit machine, i.e. prepares the context that is to be shown to the learner and gives reward.
In the case of optimization on graph, there is no context. So we'd need to modify it accordingly.
* `UCB_algs.py`: contains the our algorithm classes. 
* `run_bandit.py`: Is the module that simulates the interaction between the bandit and the learner. The current code also generates a simple regret and information gain plot.
* `models.py`: To store the network models. 
* `exp_ambig.py`: Contains the code to the ambiguity experiment presented in Appendix A.3. Note that the network should be trained using the training data before running this descript. 
* `exp_underrep.py`: Contains the code to the imbalanced classes experiment given in Appendix A.3.  Note that the network should be trained using the training data before running this descript. The model should be trained with `undderrep = True` argument for the data_loader class.

## Package Requirement

I generated an automatic file with (I hope) all the relevant packages. If things didn't work on your devices, we can either try to make the code compatible with the most recent version of the packages, or I can give you the virtual environment that I use.

