import torch
import numpy as np
from utils.utils import *
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

def train_vdb(var_model, memory, demonstrations, beta, args):
    loss = nn.MSELoss()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(var_model.parameters())
    memory = np.array(memory) 
    states = np.vstack(memory[:, 0]) 
    actions = list(memory[:, 1]) 
    states = torch.Tensor(states)
    actions = torch.Tensor(actions)
    
    for _ in range(args.vdb_update_num):
        mle_loss = 0
        st=torch.cat([states, actions],dim=1)
        demonstrations = torch.Tensor(demonstrations)
        target= var_model.forward(demonstrations)
        var_model.zero_grad()
        log_priors = torch.zeros(1)
        log_variational_posteriors =torch.zeros(1)
        outputs = var_model.forward(st, sample=True)
        log_prior= var_model.log_prior()
        log_variational_posterior=var_model.log_variational_posterior()
        negative_log_likelihood = F.mse_loss(outputs, target[0:len(outputs)], size_average=False)
        loss =(log_variational_posterior - log_prior)/3 + negative_log_likelihood
        loss.backward()
        optimizer.step()

def gaussian_fit_loss(data, mean, rho):
    """ Compute log likelihood of data, assuming 
    a N(mean, log(1 + e^rho)) model.
    """
    sigma = torch.log(1 + torch.exp(rho))
    return torch.mean((mean - data) ** 2 / (2 * sigma ** 2)) + torch.mean(torch.log(sigma))


    

def train_actor_critic(actor, critic, memory, actor_optim, critic_optim, args):
    memory = np.array(memory) 
    states = np.vstack(memory[:, 0]) 
    actions = list(memory[:, 1]) 
    rewards = list(memory[:, 2]) 
    masks = list(memory[:, 3]) 

    old_values = critic(torch.Tensor(states))
    returns, advants = get_gae(rewards, masks, old_values, args)
    
    mu, std = actor(torch.Tensor(states))
    old_policy = log_prob_density(torch.Tensor(actions), mu, std)

    criterion = torch.nn.MSELoss()
    n = len(states)
    arr = np.arange(n)

    for _ in range(args.ppo_update_num):
        np.random.shuffle(arr)

        for i in range(n // args.batch_size): 
            batch_index = arr[args.batch_size * i : args.batch_size * (i + 1)]
            batch_index = torch.LongTensor(batch_index)
            
            inputs = torch.Tensor(states)[batch_index]
            actions_samples = torch.Tensor(actions)[batch_index]
            returns_samples = returns.unsqueeze(1)[batch_index]
            advants_samples = advants.unsqueeze(1)[batch_index]
            oldvalue_samples = old_values[batch_index].detach()
            
            values = critic(inputs)
            clipped_values = oldvalue_samples + \
                             torch.clamp(values - oldvalue_samples,
                                         -args.clip_param, 
                                         args.clip_param)
            critic_loss1 = criterion(clipped_values, returns_samples)
            critic_loss2 = criterion(values, returns_samples)
            critic_loss = torch.max(critic_loss1, critic_loss2).mean()

            loss, ratio, entropy = surrogate_loss(actor, advants_samples, inputs,
                                         old_policy.detach(), actions_samples,
                                         batch_index)
            clipped_ratio = torch.clamp(ratio,
                                        1.0 - args.clip_param,
                                        1.0 + args.clip_param)
            clipped_loss = clipped_ratio * advants_samples
            actor_loss = -torch.min(loss, clipped_loss).mean()

            loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

            critic_optim.zero_grad()
            loss.backward(retain_graph=True) 
            critic_optim.step()

            actor_optim.zero_grad()
            loss.backward()
            actor_optim.step()

def get_gae(rewards, masks, values, args):
    rewards = torch.Tensor(rewards)
    masks = torch.Tensor(masks)
    returns = torch.zeros_like(rewards)
    advants = torch.zeros_like(rewards)
    
    running_returns = 0
    previous_value = 0
    running_advants = 0

    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + (args.gamma * running_returns * masks[t])
        returns[t] = running_returns

        running_delta = rewards[t] + (args.gamma * previous_value * masks[t]) - \
                                        values.data[t]
        previous_value = values.data[t]
        
        running_advants = running_delta + (args.gamma * args.lamda * \
                                            running_advants * masks[t])
        advants[t] = running_advants

    advants = (advants - advants.mean()) / advants.std()
    return returns, advants

def surrogate_loss(actor, advants, states, old_policy, actions, batch_index):
    mu, std = actor(states)
    new_policy = log_prob_density(actions, mu, std)
    old_policy = old_policy[batch_index]

    ratio = torch.exp(new_policy - old_policy)
    surrogate_loss = ratio * advants
    entropy = get_entropy(mu, std)

    return surrogate_loss, ratio, entropy