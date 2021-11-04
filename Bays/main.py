import os
import gym
import pickle
import argparse
import numpy as np
from collections import deque
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter 

from utils.utils import *
from utils.zfilter import ZFilter
from model import Actor, Critic, BNN
from train_model import train_actor_critic, train_vdb

parser = argparse.ArgumentParser(description='PyTorch VAIL')
parser.add_argument('--env_name', type=str, default="MountainCar-v0", 
                    help='name of the environment to run')
parser.add_argument('--load_model', type=str, default=None, 
                    help='path to load the saved model')
parser.add_argument('--render', action="store_true", default=False, 
                    help='if you dont want to render, set this to False')
parser.add_argument('--gamma', type=float, default=0.99, 
                    help='discounted factor (default: 0.99)')
parser.add_argument('--lamda', type=float, default=0.98, 
                    help='GAE hyper-parameter (default: 0.98)')
parser.add_argument('--hidden_size', type=int, default=100, 
                    help='hidden unit size of actor, critic and vdb networks (default: 100)')
parser.add_argument('--z_size', type=int, default=4,
                    help='latent vector z unit size of vdb networks (default: 4)')
parser.add_argument('--learning_rate', type=float, default=3e-4, 
                    help='learning rate of models (default: 3e-4)')
parser.add_argument('--l2_rate', type=float, default=1e-3, 
                    help='l2 regularizer coefficient (default: 1e-3)')
parser.add_argument('--clip_param', type=float, default=0.2, 
                    help='clipping parameter for PPO (default: 0.2)')
parser.add_argument('--alpha_beta', type=float, default=1e-4,
                    help='step size to be used in beta term (default: 1e-4)')
parser.add_argument('--i_c', type=float, default=0.5, 
                    help='constraint for KL-Divergence upper bound (default: 0.5)')
parser.add_argument('--vdb_update_num', type=int, default=3, 
                    help='update number of variational discriminator bottleneck (default: 3)')
parser.add_argument('--ppo_update_num', type=int, default=10, 
                    help='update number of actor-critic (default: 10)')
parser.add_argument('--total_sample_size', type=int, default=2048, 
                    help='total sample size to collect before PPO update (default: 2048)')
parser.add_argument('--batch_size', type=int, default=64, 
                    help='batch size to update (default: 64)')
parser.add_argument('--suspend_accu_exp', type=float, default=0.8,
                    help='accuracy for suspending discriminator about expert data (default: 0.8)')
parser.add_argument('--suspend_accu_gen', type=float, default=0.8,
                    help='accuracy for suspending discriminator about generated data (default: 0.8)')
parser.add_argument('--max_iter_num', type=int, default=100,
                    help='maximal number of main iterations (default: 4000)')
parser.add_argument('--seed', type=int, default=500,
                    help='random seed (default: 500)')
parser.add_argument('--logdir', type=str, default='logs',
                    help='tensorboardx logs directory')
args = parser.parse_args()


def main():
    env = gym.make(args.env_name)
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n
    state_act=num_actions+num_inputs
    running_state = ZFilter((num_inputs,), clip=5)

    print('state size:', num_inputs) 
    print('action size:', num_actions)

    actor = Actor(num_inputs, num_actions, args)
    critic = Critic(num_inputs, args)
    bnn    = BNN(state_act)

    actor_optim = optim.Adam(actor.parameters(), lr=args.learning_rate)
    critic_optim = optim.Adam(critic.parameters(), lr=args.learning_rate, 
                              weight_decay=args.l2_rate) 
    
    # load demonstrations
    raw_demo = np.load(file="./expert_demo/expert_trajectories.npy",allow_pickle=True)
    demonstrations = np.zeros((20, len(raw_demo[0]), state_act)) 
    for x in range(20):
        for y in range(len(raw_demo[0])):
            demonstrations[x][y][0] = raw_demo[x][y][0]
            demonstrations[x][y][1] = raw_demo[x][y][1]
            if raw_demo[x][y][2]==1:
                demonstrations[x][y][2] = 0.2
                demonstrations[x][y][3] = 0.96
                demonstrations[x][y][4] = 0.2
            if raw_demo[x][y][2]==2:
                demonstrations[x][y][2] = 0.2
                demonstrations[x][y][3] = 0.0
                demonstrations[x][y][4] = 0.96
            else:
                demonstrations[x][y][2] = 0.96
                demonstrations[x][y][3] = 0.0
                demonstrations[x][y][4] = 0.2
    writer = SummaryWriter(args.logdir)

    if args.load_model is not None:
        saved_ckpt_path = os.path.join(os.getcwd(), 'save_model', str(args.load_model))
        ckpt = torch.load(saved_ckpt_path)

        actor.load_state_dict(ckpt['actor'])
        critic.load_state_dict(ckpt['critic'])
        vdb.load_state_dict(ckpt['bnn'])

        running_state.rs.n = ckpt['z_filter_n']
        running_state.rs.mean = ckpt['z_filter_m']
        running_state.rs.sum_square = ckpt['z_filter_s']

        print("Loaded OK ex. Zfilter N {}".format(running_state.rs.n))

    
    episodes = 0
    train_discrim_flag = True

    for iter in range(args.max_iter_num):
        actor.eval(), critic.eval()
        memory = deque()
        steps = 0
        scores = []
        while steps < args.total_sample_size: 
            state = env.reset()
            score = 0

            state = running_state(state)
            
            for _ in range(500): 
                if args.render:
                    env.render()
                steps += 1
                mu, std = actor(torch.Tensor(state).unsqueeze(0))
                action = get_action(mu, std)[0]
                act=np.argmax(action)
                next_state, reward, done, _ = env.step(act)
                stat_act=np.append(state,action)
                stat_act=torch.from_numpy(stat_act).float()
                irl_reward = bnn.forward(stat_act)
                if done:
                    mask = 0
                else:
                    mask = 1
                memory.append([state, action, irl_reward, mask])

                next_state = running_state(next_state)
                state = next_state

                score += reward

                if done:
                    break
            
            episodes += 1
            scores.append(score)
        
        score_avg = np.mean(scores)
        print('{}:: {} episode score is {:.2f}'.format(iter, episodes, score_avg))
        writer.add_scalar('log/score', float(score_avg), iter)

        train_vdb(bnn, memory,demonstrations, 0, args)
        
        train_actor_critic(actor, critic, memory, actor_optim, critic_optim, args)

        # save the models
        if iter+1==args.max_iter_num and done:
            score_avg = int(score_avg)

            model_path = os.path.join(os.getcwd(),'./Bays/save_model')
            if not os.path.isdir(model_path):
                os.makedirs(model_path)

            ckpt_path = os.path.join(model_path, 'ckpt_'+ str(score_avg)+'.pth.tar')

            save_checkpoint({
                'actor': actor.state_dict(),
                'critic': critic.state_dict(),
                'bnn': bnn.state_dict(),
                'z_filter_n':running_state.rs.n,
                'z_filter_m': running_state.rs.mean,
                'z_filter_s': running_state.rs.sum_square,
                'args': args,
                'score': score_avg
            }, filename=ckpt_path)

if __name__=="__main__":
    main()