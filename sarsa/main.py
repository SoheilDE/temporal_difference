from sarsa import SARSA

import numpy as np
import plotly.express as px

if __name__ == '__main__':
    params_policy = {
        'num_episode': 500000,
        'eps': 0.3,
        'gamma': 0.9,
        'alpha': 0.1
    }

    params_reward = {
        'num_episode': 1000,
        'epsilon': 0.3,
        'gamma': 0.9,
        'alpha': 0.1,
        'num_episode_test': 40,
        'num_test': 100
    }

    sarsa = SARSA()
    optimal_policy, t_train = sarsa.opt_policy(params_policy)
    print("Number of episodes: {} \n Training time: {} minutes".format(params_policy['num_episode'], t_train))
    print('\n ----------------------- \n Final Policy: \n', optimal_policy)

    mean_rewards = sarsa.reward(params_reward)
    fig = px.line(y=np.mean(mean_rewards, axis=0))
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    fig.show()
