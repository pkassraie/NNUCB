import numpy as np
import jax.numpy as jnp
import torch
import copy
import torch.optim as optim
from utils import normalize_init
from models import *
from utils import get_kernel
import neural_tangents as nt


class NeuralUCBDiag:  # Similar to implementation in "Contexual Neural Bandits with UCB Exploration"
    def __init__(self, net, input_dim,n_arm = 10, layers=1, sigma=1, beta=1, hidden=100, xavier=False):
        if net == 'NN':
            self.func = NN(input_dim=input_dim, depth=layers, width=hidden).cuda()
        else:
            self.func = CNN(input_dim=input_dim, depth=layers, width=hidden).cuda()

        if xavier == True:
            self.func = normalize_init(self.func)

        self.context_list = []
        self.reward = []
        self.sigma = sigma  # observation noise
        self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)
        self.U = sigma * torch.ones((self.total_param,)).cuda()
        self.beta = beta  # exploration coefficient
        self.net = net
        self.width = hidden
        if net == 'NN':
            self.name = 'Neural-UCB'
        else:
            self.name = 'CNeural-UCB'
        self.path = 'trained_models/{}_{}dim_{}L_{}m_{:.3e}beta_{:.1e}sigma'.format(self.net,input_dim, layers, hidden, self.beta, self.sigma)


    def select(self, context):
        if self.net == 'CNN':
            context = np.expand_dims(context, axis=1)
        tensor = torch.from_numpy(context).float().cuda()
        post_mean = self.func(tensor)
        grad_list = []
        self.UCBs = []
        self.vars = []
        for fx in post_mean:
            self.func.zero_grad()
            fx.backward(retain_graph=True)
            g = torch.cat([p.grad.flatten().detach() for p in self.func.parameters()])
            grad_list.append(g)
            post_var = torch.sqrt(torch.sum(self.beta * self.sigma * g * g / self.U / self.width))
            UCB = fx.item() + post_var.item()
            self.UCBs.append(UCB)
            self.vars.append(post_var.item())

        arm = np.argmax(self.UCBs)
        self.U += grad_list[arm] * grad_list[arm] / self.width  # U is diagonal
        return arm

    def train(self, context, reward):
        if isinstance(context, list):  # for batch training
            self.context_list = self.context_list + list(map(self.prep_context, context))
        else:
            self.context_list.append(self.prep_context(context))
        if isinstance(reward, list):
            self.reward = self.reward + reward
        else:
            self.reward.append(reward)
        optimizer = optim.SGD(self.func.parameters(), lr=1e-2, weight_decay=self.sigma)
        length = len(self.reward)
        index = np.arange(length)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0
        while True:
            epoch_loss = 0
            for idx in index:
                c = self.context_list[idx]
                r = self.reward[idx]
                optimizer.zero_grad()
                delta = self.func(c.cuda()) - r
                loss = delta * delta
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 1000:  # train each epoch for J \leq 1000
                    return tot_loss / 1000
            if epoch_loss / length <= 1e-3:  # stop training if the average loss is less than 0.001
                return epoch_loss / length

    def get_infogain(self):
        K_T = self.U - self.sigma * torch.ones((self.total_param,)).cuda()
        return 0.5 * np.log(torch.prod(1 + K_T / self.sigma).cpu().numpy())

    def save_model(self):
        torch.save(self.func, self.path)


    def load_model(self):
        try:
            self.func = torch.load(self.path)
            self.func.eval()
        except:
            print('Pretrained model not found.')

    def prep_context(self, context):
        if self.net == 'NN':
            return torch.from_numpy(context.reshape(1, -1)).float()
        else:
            return torch.from_numpy(np.expand_dims(context.reshape(1, -1), axis=1)).float()


class NNUCBDiag(NeuralUCBDiag):  # Our main method
    def __init__(self, net, input_dim, layers=1, sigma=1, beta=1, hidden=128, xavier=False):
        super().__init__(net, input_dim, layers, sigma, beta, hidden, xavier)
        self.f0 = copy.deepcopy(self.func)
        if xavier == True:
            self.f0 = normalize_init(self.f0)
        self.path2 = 'trained_models/{}_init_{}dim_{}L_{}m_{:.3e}beta_{:.1e}sigma'.format(self.net, input_dim, layers, hidden, self.beta, self.sigma)
        if net == 'NN':
            self.name = 'NN-UCB'
        else:
            self.name = 'CNN-UCB'

    def select(self, context):
        if self.net == 'CNN':
            context = np.expand_dims(context, axis=1)
        tensor = torch.from_numpy(context).float().cuda()
        post_mean = self.func(tensor)
        post_mean0 = self.f0(tensor)
        grad_list = []
        self.UCBs = []
        self.vars = []
        for fx, fx0 in zip(post_mean, post_mean0):
            self.f0.zero_grad()
            fx0.backward(retain_graph=True)
            g = torch.cat([p.grad.flatten().detach() for p in self.f0.parameters()])
            grad_list.append(g)
            post_var = torch.sqrt(torch.sum(self.beta * self.sigma * g * g / self.U / self.width))
            UCB = fx.item() + post_var.item()
            self.UCBs.append(UCB)
            self.vars.append(post_var.item())
        arm = np.argmax(self.UCBs)
        self.U += grad_list[arm] * grad_list[arm] / self.width  # U is diagonal
        return arm

    def load_model(self):
        super().load_model()
        try:
            self.f0 = torch.load(self.path2)
            self.f0.eval()
        except:
            print('Model at initialization not found.')

    def save_model(self):
        super().save_model()
        torch.save(self.f0, self.path2)

class NTKUCB():
    def __init__(self,input_dim,beta = 0.01,  sigma=1e-4, n_arm = 10,net = 'NN', hidden=100, layers = 1):
        self.diag_reg = sigma
        self.x_train = None # these have to be jax matrixes
        self.y_train = None
        self.kernel = get_kernel(width = hidden, depth = layers, type = net)
        self.predict_fn = None
        self.n_arm = n_arm
        self.beta = beta
        self.hidden = hidden
        self.step = 0
        self.net = net
        self.layers = layers
        if net == 'NN':
            self.name = f'NTKUCB_{layers}L'
        else:
            self.name = f'CNTKUCB_{layers}L'

    def select(self, context):
        if self.predict_fn == None:
            #randomly pick an arm
            arm = np.random.randint(0,10)
            self.UCBs = jnp.zeros((10))
            self.vars = jnp.ones((10)) #HMMMMM
        else:
            self.UCBs = []
            self.vars = []
            for a in range(self.n_arm):
                mu_pred, sigma_pred = self.predict_fn(x_test=self.prep_context(context[a]), get='ntk', compute_cov=True)
                ucb = mu_pred + self.beta * sigma_pred
                self.UCBs.append(ucb)
                self.vars.append(sigma_pred)
            arm = jnp.argmax(jnp.asarray(self.UCBs))
        return arm

    def train(self, context=None, reward=None):
        if np.ndim(context) == 0:
            self.predict_fn = nt.predict.gradient_descent_mse_ensemble(self.kernel, self.x_train, self.y_train,
                                                                       diag_reg=self.diag_reg)
        else:
            self.add_data(context, reward)
            self.predict_fn = nt.predict.gradient_descent_mse_ensemble(self.kernel, self.x_train, self.y_train,
                                                                       diag_reg=self.diag_reg)

    def add_data(self, context, reward):
        context = np.array(context)
        reward = np.array(reward)
        if context.ndim < 2:
            self.add_context(context)
            self.add_reward(reward)
            self.step += 1
        else :
            for context_t, reward_t in zip(context, reward):
                self.add_context(context_t)
                self.add_reward(reward_t)
                self.step += 1

    def prep_context(self,x):
        if self.net == 'NN':
            if x.ndim <2:
                x = np.expand_dims(x, axis=0)
        else:
            if x.ndim <2:
                x = np.reshape(x, (1, 1, 1, -1))
        x = x.astype(np.float32)
        return jnp.asarray(x)

    def prep_reward(self,x):
        if x.ndim <2:
            x = np.expand_dims(x, axis=0)
        x = x.astype(np.float32)
        return jnp.asarray(x)

    def add_context(self,x):
        if self.x_train == None:
            self.x_train = self.prep_context(x)
        else:
            self.x_train = jnp.vstack((self.x_train,self.prep_context(x)))

    def add_reward(self,y):
        if self.y_train == None:
            self.y_train = self.prep_reward(y)
        else:
            self.y_train = jnp.vstack((self.y_train,self.prep_reward(y)))

    def get_infogain(self, normalize = False):
        K_T = self.kernel(self.x_train, self.x_train, 'ntk')
        size = K_T.shape[0]
        K_T = jnp.squeeze(K_T)
        if normalize ==True:
            max_k = jnp.max(jnp.diag(K_T))
            if max_k > 1:
                K_T = K_T/max_k
        I = 0.5 * jnp.log(jnp.linalg.det(jnp.identity(size) + K_T / self.diag_reg))
        return np.squeeze(I)

    def save_model(self):
        pass
