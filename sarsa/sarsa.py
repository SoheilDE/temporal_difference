import numpy as np
import time
from tqdm import tqdm
import gym

env = gym.make("FrozenLake-v1", is_slippery=True)


class SARSA:
    def __init__(self):
        self.num_observ = env.observation_space.n
        self.num_action = env.action_space.n

    def opt_policy(self, params_policy):
        q = np.zeros([self.num_observ, self.num_action])
        num_epi = params_policy['num_episode']
        epsilon = params_policy['eps']
        gamma = params_policy['gamma']
        alpha = params_policy['alpha']
        start = time.time()
        for num in range(num_epi):
            state = env.reset()
            if type(state) == int:
                state = state
            else:
                state = state[0]
            done = False
            while not done:
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(q[state, :])
                new_state, reward, done, info, _ = env.step(action)
                if np.random.rand() < epsilon:
                    new_action = env.action_space.sample()
                else:
                    new_action = np.argmax(q[new_state, :])
                q[state, action] = q[state, action] + alpha * (
                            reward + gamma * q[new_state, new_action] - q[state, action])
                state = new_state
                action = new_action
            optimal_policy = []
            for i in range(self.num_observ):
                action_opt = np.argmax(q[i, :])
                optimal_policy.append(action_opt)
            if num % 50000 == 0 and num is not 0:
                print('----------------------- \n Episode Number: {} \n'.format(num))
                print('\n Q-Value:\n{} \n'.format(q))
                print('\n Optimal Policy: \n', optimal_policy)
        stop = time.time()
        t_train = (stop - start) / 60

        return optimal_policy, t_train

    def reward(self, params_reward):
        num_episode = params_reward['num_episode']
        epsilon = params_reward['epsilon']
        gamma = params_reward['gamma']
        alpha = params_reward['alpha']
        num_episode_test = params_reward['num_episode_test']
        num_test = params_reward['num_test']
        mean_rewards = []
        for k in tqdm(range(num_test)):
            q = np.zeros([self.num_observ, self.num_action])
            mean_reward = []
            for i in range(num_episode):
                state = env.reset()
                if type(state) == int:
                    state = state
                else:
                    state = state[0]
                done = False
                results_list = []
                while not done:
                    if np.random.rand() < epsilon:
                        action = env.action_space.sample()
                    else:
                        action = np.argmax(q[state, :])
                    new_state, reward, done, info, _ = env.step(action)
                    if np.random.rand() < epsilon:
                        new_action = env.action_space.sample()
                    else:
                        new_action = np.argmax(q[new_state, :])
                    q[state, action] = q[state, action] + alpha * (
                                reward + gamma * q[new_state, new_action] - q[state, action])
                    results_list.append((state, action))
                    state = new_state
                    action = new_action

                optimal_policy = []
                for i in range(self.num_observ):
                    action_opt = np.argmax(q[i, :])
                    optimal_policy.append(action_opt)
                sum_reward = 0
                for i in range(num_episode_test):
                    done = False
                    state_test = env.reset()
                    if type(state_test) == int:
                        state_test = state_test
                    else:
                        state_test = state_test[0]
                    while not done:
                        state_test, reward_test, done, info, _ = env.step(optimal_policy[state_test])
                        if reward_test > 0:
                            sum_reward += 1
                mean_reward.append(sum_reward / 100)
            mean_rewards.append(mean_reward)
        return mean_rewards
