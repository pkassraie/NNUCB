# NNUCB
This is a light version of the code for the paper [Neural Contextual Bandits without Regret](https://arxiv.org/abs/2107.03144).

## Scripts
* `data_loader.py`: Contains the class that imitates a contextual bandit machine, i.e. prepares the context that is to be shown to the learner and gives reward.
* `UCB_algs.py`: Contains the algorithm classes: (C)NN-UCB and (C)NTK-UCB as in the paper and [NeuralUCB](https://arxiv.org/abs/1911.04462).
* `run_bandit.py`: Is the module that simulates the interaction between the bandit and the learner. The current code also generates a simple regret and information gain plot.
* `models.py`: To store the network models. 
* `exp_ambig.py`: Contains the code to the ambiguity experiment presented in Appendix A.3. Note that the network should be trained using the training data before running this descript. 
* `exp_underrep.py`: Contains the code to the imbalanced classes experiment given in Appendix A.3.  Note that the network should be trained using the training data before running this descript. The model should be trained using a `data_loader` with the argument `underrep = True`.
 
## Package Requirement
An automatic requirements.txt file with (hopefully) all the relevant packages is included. 
I have not checked whether the code is compatible with the most recent version of the used libraries, JAX in particular worries me. 
We may also share the used virtual environment should there be any difficulties. 
