import numpy as np
from collections import deque
import torch
from learner import Learner
import matplotlib.pyplot as plt
import gym
import random
from envs.path_plan_env import PathPlanEnv
import time


def set_weights(model,fps_weights):
  start_idx=0
  
  for param in model.parameters():
    # print(param.data)
    param_size = param.size()
    param_num = param_size.numel()
    fpa_values = torch.tensor(fps_weights[start_idx:start_idx+param_num])
    fpa_values = torch.reshape(fpa_values, param_size)
    fpa_values = fpa_values.float()
    param.data = torch.nn.parameter.Parameter(fpa_values)
    start_idx+= param_num
    # print(param.data)

def initialize_weights(m):
  if isinstance(m, torch.nn.Conv2d):
      torch.nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
      if m.bias is not None:
          torch.nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, torch.nn.BatchNorm2d):
      torch.nn.init.constant_(m.weight.data, 1)
      torch.nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, torch.nn.Linear):
      torch.nn.init.kaiming_uniform_(m.weight.data)
      torch.nn.init.constant_(m.bias.data, 0)


def levy(d):
    lamda = 1.5
    # sigma = (math.gamma(1 + lamda) * math.sin(math.pi * lamda / 2) / (
    # math.gamma((1 + lamda) / 2) * lamda * (2 ** ((lamda - 1) / 2)))) ** (1 / lamda)
    sigma = 0.6965745025576968
    u = np.random.randn(1, d) * sigma
    v = np.random.randn(1, d)
    step = u / abs(v) ** (1 / lamda)
    return 0.01 * step


def fpa(l, n, p, N_iter, d, num_obs):

    model = l.create_policy_network()

    # replay = deque(maxlen=100)
    sol = np.ones((n, d))
    fitness = np.ones((n, 1))

    #initialize n flower weights
    for i in range(0, n):
        model.apply(initialize_weights)
        f_val, iter = evaluate_model(l,model, num_obs)
        fitness[i, 0] = f_val
        params = []
        for param in model.parameters():
            params.append(param.view(-1))
        params = torch.cat(params)
        sol[i,] = params.detach().cpu().numpy()

    #best fitness and flower
    fmax, I = fitness.max(0), fitness.argmax(0)
    best = sol[I,]
    l_iter = 100

    # for i in range(0,n):
    #     sol[i,]=best*np.random.randn(1);

    S = sol.copy()
    print("fmax",fmax)
    for t in range(0, N_iter):
        for i in range(0, n):
            # p=0.8-0.7*t/N_iter
            if np.random.random() < p:
                L = levy(d)
                S[i,] = sol[i,] + L * (sol[i,] - best)
            else:
                epsilon = np.random.random_sample()
                jk = np.random.permutation(n)
                S[i,] = S[i,] + epsilon * (sol[jk[0],] - sol[jk[1],])
            # S[i,] = simple(S[i,], lb, ub, d)
            Fnew, iter = fun(S[i,], model, l, num_obs)
            if Fnew >= fitness[i]:
                sol[i,] = S[i,]
                fitness[i] = Fnew
            if Fnew >= fmax:
                best = S[i,]
                fmax = Fnew
                l_iter = iter
        if t % 1 == 0:
            print("t",t, "fmax: ",fmax)

    print("best:")
    print(best)
    print("fmax: {}, iter: {}".format(fmax, l_iter))
    set_weights(model,best)
    return model,fmax, l_iter


def fun(u,model,l, num_obs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_weights(model,u)
    model.to(device)
    fitness, iter = evaluate_model(l,model, num_obs)
    return fitness, iter

def evaluate_model(l,policy_net, num_obs):
    device = l.device
    rewards = np.zeros(l.T)

    start = [120, 120]
    goal =  [450, 440]

    map_path = "maps/Map_{}_obs.png".format(num_obs)
    env: PathPlanEnv = gym.make("envs/PathPlanEnv-v0", file=map_path, start=np.array(start), goal=np.array(goal))

    obs = env.reset()

    t_map, curr_state, curr_pos = l.reset_vars(env)

    for j in range(l.T):
            sample = random.random()
            # if sample < l.epsilon:
            # 	# print("Taking random action")
            # 	action = random.choice(l.act)
            # else:
            with torch.no_grad():
                q_vals = policy_net(curr_state) #q_vals of current state

                # print("taking best action")
                action = (torch.argmax(q_vals)).int().item()


            # print("action is: {}".format(action))
            # r, map, next_pos = transition(map, curr_pos, goal, action)
            obs, reward, done, _ = env.step(action)
            # print(reward)
            rewards*=l.gamma
            rewards[j] = reward
            # print("moving to: {}, reward: {}".format(env.current_position, reward))

            #updating map for torch format
            # t_map = map.reshape(4,600,600) #t_map = torch_map, map in format for pytorch
            t_map = torch.from_numpy(obs["map"] / 255)
            t_map = t_map.float()
            t_map = t_map.unsqueeze(0)
            t_map = t_map.unsqueeze(0)

            # if (j%5 == 0):
            # 	imgplot = plt.imshow(t_map.squeeze(0).squeeze(0).detach().numpy(), cmap="gray")
            # 	plt.show()

            curr_pos = env.current_position
            curr_state = t_map

            # self.epsilon = self.epsilon - self.epsilon_decay_const
            l.totIter += 1
            l.episodeIter += 1
            if (done):
                print("reached goal. iterations: {}".format(l.episodeIter))
                rewards[j] = 100 + 25*(l.T - l.episodeIter)
                imgplot = plt.imshow(t_map.squeeze(0).squeeze(0).detach().numpy(), cmap="gray")
                # plt.show()
                break
    return rewards.sum(), l.episodeIter
    # return reward


if __name__ == '__main__':
    start = time.time()

    n = 8 #number of flowers
    p = 0.8 #probability of global vs local
    N_iters = 10
    d = 702596 #num. dims = no. of parameters to initialize

    num_obs = 8
    path_threshold = 0
    max_episodes = 50

    l = Learner(False, max_episodes)
    model, fmax, l_iter = fpa(l, n, p, N_iters, d,num_obs)
    l.epsilon = l.epsilon - l.epsilon_decay_const*fmax
    if (l.epsilon < l.epsilon_o):
        l.epsilon = l.epsilon_o
    l.run_learner(model, num_obs , path_threshold)
    print("total time: ", time.time()-start)


