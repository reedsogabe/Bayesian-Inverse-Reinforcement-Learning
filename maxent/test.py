import gym
import pylab
import numpy as np
import time

q_table = np.load(file="results/maxent_q_table.npy") # (400, 3)
one_feature = 20 # number of state per one feature

def idx_to_state(env, state):
    """ Converting the position and velocity of the moutaincar environment which is originally continuous to an integer value"""
    env_low = env.observation_space.low                              #minimum value of position and velocity
    env_high = env.observation_space.high                            #maximum value of position and velocity"""
    env_distance = (env_high - env_low) / one_feature
    position_idx = int((state[0] - env_low[0]) / env_distance[0])
    velocity_idx = int((state[1] - env_low[1]) / env_distance[1])
    state_idx = position_idx + velocity_idx * one_feature            #converts the state into one whole number"""
    print('this is stateindexboi', state_idx)   #prints state index for my entertainment
    time.sleep(0.2)                             #delay between each state index
    return state_idx                            #returns state index  to be used later on

def main():
    env = gym.make('MountainCar-v0')            #creates the mountain car environment
    episodes, scores = [], []                   #creates list for episodes, and scores obtained

    for episode in range(100):                  #creates a for loop (which is for 100 tries )
        state = env.reset()                     #resets the Mountain Car (gym) environment to the intial state
        score = 0                               #sets the initial value of the score as 0

        while True:                             #while loop is looped 100 times
            env.render()                        #renders the open gym environemnt
            state_idx = idx_to_state(env, state)
            action = np.argmax(q_table[state_idx])
            next_state, reward, done, _ = env.step(action)

            score += reward
            state = next_state

            if done:
                scores.append(score)
                episodes.append(episode)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./learning_curves/maxent_test.png")
                break

        if episode % 1 == 0:
            print('{} episode score is {:.2f}'.format(episode, score))
    env.close()

if __name__ == '__main__':
    main()
