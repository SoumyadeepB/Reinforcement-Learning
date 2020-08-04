# Reinforcement Learning

A Collection of Reinforcement Learning Algorithms implemented in Python:

- **Multi-Armed Bandit**
  - The multi-armed bandit problem is a classical problem that demonstrates the Exploration vs Exploitation dilemma. 
     - Situation: *k* slot machines in a casino - each configured with unknown reward probabilities.
     - Question: Which of the *k* levers must be pulled to achieve highest long-term rewards?

- **Frozen Lake (Brute Force all State-Action pairs)**
    - FrozenLake is a simple grid world with 4 actions (0-left 1-down 2-right 3-up). However, the ground is slippery (the
agent is on a frozen lake), so that it ends up on the correct next field only with probability 1/3 (e.g. instead of going
down it could also end up left or right). When the action would bump the agent into a border it would stay in the
same state. At the goal the agent will receive +1 reward, elsewhere it receives 0 reward. An episode terminates when the agent ends up at the goal or in a hole. 
    - Brute-Force Approach: Iterate over all possible policies and compute v_pi. Find optimal value function v* and thus compute the optimal policy.

- **Frozen Lake (Dynamic Programming)**
  - Approach: Dynamic programming to implement a recursive decomposition of the Bellman Equation
     - Achieve optimal substructure
     - Exploit the overlapping nature of the subproblems

- **Frozen Lake (Policy Iteration)**

- **Monte-Carlo method on the Blackjack game (First-visit and Exploring Starts)**
  - Approach: Monte-Carlo Learning
     - Exploring Starts: Estimate the Q-Value function by randomly starting at any state, then choose the best (greedy) action.
     - First-visit MC: Increment total return by only considering the first time-step 't' that state 's' is visited in an episode.

- **Sarsa**
- **Q-Learning**

(To be updated...)

## Requirements
* Python 3.x
* OpenAI Gym 
   * `pip install gym`
