import gym
from reinforce_model import PolicyNetwork
from reinforce_update import update_policy
import numpy as np
import sys
import matplotlib.pyplot as plt
from IPython.display import clear_output
import seaborn as sns; sns.set()
import time



def main(GAMMA=.9):
    start = time.time()
    
    env = gym.make('CartPole-v0')
    policy_net = PolicyNetwork(env.observation_space.shape[0], env.action_space.n, 128)
    
    max_episode_num = 10000
    max_steps = 10000
    numsteps = []
    avg_numsteps = []
    all_rewards = []

    for episode in range(max_episode_num):
        state = env.reset()
        log_probs = []
        rewards = []

        for steps in range(max_steps):
            #env.render()
            action, log_prob = policy_net.get_action(state)
            new_state, reward, done, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                update_policy(policy_net, rewards, log_probs, GAMMA=GAMMA)
                numsteps.append(steps)
                avg_numsteps.append(np.mean(numsteps[-10:]))
                all_rewards.append(np.sum(rewards))
                if episode % 1 == 0:
                    sys.stdout.write("episode: {}, total reward: {}, average_reward: {}, length: {}\n".format(episode, np.round(np.sum(rewards), decimals = 3),  np.round(np.mean(all_rewards[-10:]), decimals = 3), steps))
                    clear_output(wait=True)
                break
            
            state = new_state
        convergence_interval = 100
        if episode > convergence_interval:
            if np.std(numsteps[-convergence_interval:]) < .1:
                break
    
    plt.plot(numsteps, label="Number of Steps")
    plt.plot(avg_numsteps, label="SMA of Number of Steps", linestyle='--')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.legend()
    plt.savefig(f'Learning Curve (gamma={GAMMA}).png')
    plt.show();

    
    done = False
    total_games = 100
    rewards = np.zeros(total_games)
    for i in range(total_games):
        total_reward = 0
        observation = env.reset()
        while not done:       
            action, log_prob = policy_net.get_action(observation)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                done = False
                rewards[i] = total_reward
                break
    
    time_elapsed = time.time() - start
    time_elapsed = np.round(time_elapsed/60.0, 2)
    print(f'The average reward of the physical solution is {np.average(rewards)}.')
    print(f'The standard deviation of the reward of the physical solution is {np.std(rewards)}.')
    print(f'The total number of episiodes is {episode + 1}.')
    print(f'The time elapsed is {time_elapsed} minutes.')