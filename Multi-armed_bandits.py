import numpy as np
import matplotlib.pyplot as plt
import random


class GaussianBandit:
    def __init__(self):
        self._arm_means = np.random.uniform(0., 1., 10)  # Sample some means
        self.n_arms = len(self._arm_means)
        self.rewards = []
        self.total_played = 0

    def reset(self):
        self.rewards = []
        self.total_played = 0

    def play_arm(self, a):
        # Use sampled mean and covariance of 1.
        reward = np.random.normal(self._arm_means[a], 1.)
        self.total_played += 1
        self.rewards.append(reward)
        return reward


def greedy(bandit, timesteps):
    rewards = np.zeros(bandit.n_arms)
    n_plays = np.zeros(bandit.n_arms)
    Q = np.zeros(bandit.n_arms)
    possible_arms = range(bandit.n_arms)

    # TODO: init variables (rewards, n_plays, Q) by playing each arm once
    for i in range(bandit.n_arms):
        reward = bandit.play_arm(i)
        rewards[i] += reward
        n_plays[i] += 1
        Q[i] = rewards[i]/n_plays[i]

    # Main loop
    while bandit.total_played < timesteps:
        # This example shows how to play a random arm:
        '''
        a = random.choice(possible_arms)
        reward_for_a = bandit.play_arm(a)
        '''
        # TODO: instead do greedy action selection
        a = np.argmax(Q)
        reward = bandit.play_arm(a)

        # TODO: update the variables (rewards, n_plays, Q) for the selected arm
        rewards[a] += reward
        n_plays[a] += 1
        Q[a] = rewards[a]/n_plays[a]


def epsilon_greedy(bandit, timesteps, eps):
    # Inititialize
    rewards = np.zeros(bandit.n_arms)
    n_plays = np.zeros(bandit.n_arms)
    Q = np.zeros(bandit.n_arms)
    possible_arms = range(bandit.n_arms)

    # TODO: init variables (rewards, n_plays, Q) by playing each arm once
    for i in range(bandit.n_arms):
        reward = bandit.play_arm(i)
        rewards[i] += reward
        n_plays[i] += 1
        Q[i] = rewards[i]/n_plays[i]

    # TODO: epsilon greedy action selection (you can copy your code for greedy as a starting point)

    while bandit.total_played < timesteps:
        x = np.random.uniform()  # Uniform random distribution between [0,1]

        if x >= eps:  # Explore
            a = random.choice(possible_arms)
        else:         # Exploit
            a = np.argmax(Q)

        reward = bandit.play_arm(a)
        rewards[a] += reward
        n_plays[a] += 1
        Q[a] = rewards[a]/n_plays[a]


def main():
    n_episodes = 10000  # TODO: set to 10000 to decrease noise in plot
    n_timesteps = 10000

    eps = 0.9
    rewards_greedy = np.zeros(n_timesteps)
    rewards_egreedy = np.zeros(n_timesteps)

    for i in range(n_episodes):
        if i % 100 == 0:
            print("current episode: " + str(i))

        b = GaussianBandit()  # initializes a random bandit
        greedy(b, n_timesteps)
        rewards_greedy += b.rewards

        b.reset()  # reset the bandit before running epsilon_greedy
        epsilon_greedy(b, n_timesteps, eps)
        rewards_egreedy += b.rewards

    rewards_greedy /= n_episodes
    rewards_egreedy /= n_episodes
    plt.plot(rewards_greedy, label="greedy")
    print("Total reward of greedy strategy averaged over " +
          str(n_episodes) + " episodes: " + str(np.sum(rewards_greedy)))
    plt.plot(rewards_egreedy, label="e-greedy")
    print("Total reward of epsilon greedy strategy averaged over " +
          str(n_episodes) + " episodes: " + str(np.sum(rewards_egreedy)))
    plt.legend()
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.savefig('bandit_strategies.png')
    plt.show()
    print("Enter:")
    input()


if __name__ == "__main__":
    main()
