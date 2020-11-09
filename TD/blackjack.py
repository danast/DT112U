import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import heapq
from pprint import pprint

env_i = gym.make('Blackjack-v0')

#env = gym.wrappers.FlattenObservation(env_i)

class BlackJackWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(BlackJackWrapper, self).__init__(env)
        self.observation_space = self.flatten_space(env.observation_space)
        d = [env.observation_space[i].n for i in range(len(env.observation_space))]
        self.mf = [int(np.prod(d[:n])) for n in range(len(d))]

    def observation(self, observation):
        return self.flatten(self.env.observation_space, observation)

    def flatten_space(self, obs_space):
        return gym.spaces.Discrete(np.prod([s.n for s in obs_space]))
    
    def flatten(self, obs_space, x):
        xv = [int(v) for v in x]
        return np.dot(self.mf, xv)

env = BlackJackWrapper(env_i)

def makeQ(env):
    s_count = env.observation_space.n
    a_count = env.action_space.n
    isize = s_count*a_count
    Q = np.zeros(isize)
    Q.shape = (s_count, a_count)
    return Q



rg = np.random.default_rng()

def getAction(Q, s_index, epsilon):
    if rg.random() > epsilon:
        # Choose greedy action
        return np.argmax(Q[s_index])
    else:
        # Choose random action
        return rg.integers(0,Q.shape[1])


#pprint(makeQ(env))


def singleSarsa(env, terminal_state, epsilon, alpha, gamma, maxEpisodes, epsilonDecrease=False):
    episodeCounts = np.zeros(maxEpisodes)
    totalRewards = np.zeros(maxEpisodes) 
    episode = 0
    delta = 0.1
    Q = makeQ(env)
    Q[terminal_state] = np.zeros(env.action_space.n)
    while episode < maxEpisodes:
        done = False
        count = 0
        totalReward = 0
        observation = env.reset()
        action = getAction(Q,observation,epsilon)
        while not done:
            count = count+1
            new_observation, reward, done, _ = env.step(action)
            totalReward += reward
            new_action = getAction(Q,new_observation,epsilon)
            Qas = Q[observation][action]
            Q[observation][action] = Qas + alpha*(reward + gamma*Q[new_observation][new_action] - Qas)
            observation = new_observation
            action = new_action
            if done:
                break
        episodeCounts[episode] = count
        totalRewards[episode] = totalReward
        episode = episode+1

        if epsilonDecrease and episode % 50 == 0:
            epsilon = epsilon/2

    return (episodeCounts, Q, totalRewards)    

def qLearning(env, terminal_state, epsilon, alpha, gamma, maxEpisodes, epsilonDecrease=False):
    episodeCounts = np.zeros(maxEpisodes)
    totalRewards = np.zeros(maxEpisodes) 
    episode = 0
    delta = 0.1
    Q = makeQ(env)
    Q[terminal_state] = np.zeros(env.action_space.n)
    while episode < maxEpisodes:
        done = False
        count = 0
        totalReward = 0
        observation = env.reset()
        action = getAction(Q,observation,epsilon)
        while not done:
            count = count+1
            new_observation, reward, done, _ = env.step(action)
            totalReward += reward
            new_action = getAction(Q,new_observation,epsilon)
            Qas = Q[observation][action]
            mx = np.max(Q[new_observation])
            Q[observation][action] = Qas + alpha*(reward + gamma*mx - Qas)
            observation = new_observation
            action = new_action
            if done:
                break
        episodeCounts[episode] = count
        totalRewards[episode] = totalReward
        episode = episode+1

        if epsilonDecrease and episode % 50 == 0:
            epsilon = epsilon/2

    return (episodeCounts, Q, totalRewards)    


epsilon = 0.1
alpha = 0.25
gamma = 1
maxEpisodes = 150000
terminal_state = 0
delta = -1

rounds = 1
trSSarsa = np.zeros((rounds, maxEpisodes))
trQl1 = np.zeros((rounds, maxEpisodes))
for r in range(rounds):
    ecSSarsa, sSarsaQ, trSSarsa[r] = singleSarsa(env, terminal_state, epsilon, alpha, gamma, maxEpisodes, True)
    ecQl1, qLQ, trQl1[r] = qLearning(env, terminal_state, epsilon, alpha, gamma, maxEpisodes, True)

def plotTotalRewards(tr, color, label):
    series = pd.Series(tr)
    rmva = series.rolling(50).mean().values
    plt.plot(rmva, color=color, label=label)

#plotTotalRewards(np.average(trSSarsa,axis=0), 'red', 'Sarsa')
#plotTotalRewards(np.average(trQl1,axis=0), 'blue', 'Q-Learning')

#plt.ylim(-100,0)
#plt.ylabel("Total Reward")
#plt.xlabel("Episode")
#plt.title(f'Average of total reward for {rounds} rounds')
#plt.legend()

#

Y = [x for x in range(4,22)]
X = [y for y in range(2,13)]

P = np.array([(1 if p[0]<p[1] else 0) for p in sSarsaQ])
P.shape = (2, 11, 32)
Pace1 = P[0,1:11,4:21].T
Pace2 = P[1,1:11,4:21].T
(Xg, Yg) = np.meshgrid(X,Y)
plt.subplot(2,2,1)
plt.pcolormesh(Xg, Yg, Pace1)
plt.ylabel("Agent card total")
plt.xlabel("Dealer card showing")
plt.title("Sarsa - No useable ace")
plt.subplot(2,2,2)
plt.pcolormesh(Xg, Yg, Pace2)
plt.ylabel("Agent card total")
plt.xlabel("Dealer card showing")
plt.title("Sarsa - Useable ace")


P = np.array([(1 if p[0]<p[1] else 0) for p in qLQ])
P.shape = (2, 11, 32)
Pace1 = P[0,1:11,4:21].T
Pace2 = P[1,1:11,4:21].T
(Xg, Yg) = np.meshgrid(X,Y)
plt.subplot(2,2,3)
plt.pcolormesh(Xg, Yg, Pace1)
plt.ylabel("Agent card total")
plt.xlabel("Dealer card showing")
plt.title("Q-Learning - No useable ace")
plt.subplot(2,2,4)
plt.pcolormesh(Xg, Yg, Pace2)
plt.ylabel("Agent card total")
plt.xlabel("Dealer card showing")
plt.title("Q-Learning - Useable ace")

# P.shape = (2, 11, 32)
# Pace1 = P[0,:,:] 
# Pace2 = P[1,:,:] 

# plt.subplot(2,2,3)
# plt.pcolormesh(Pace1)
# plt.subplot(2,2,4)
# plt.pcolormesh(Pace2)


plt.show()

def printObs(s):
    useable_ace = s // (32*11)
    rs = s % (32*11)
    dealer_card = rs // 32
    your_card = rs % 32
    print (f"Your card: {your_card}, Dealer card: {dealer_card}, Useable ace: {useable_ace}")


def followQ(Q):
    reward = 0
    epsilon = 0
    done = False
    observation = env.reset()
    action = getAction(Q,observation,epsilon)
    printObs(observation)
    while not done:
        print("Hit" if action==1 else "Stand")
        new_observation, reward, done, _ = env.step(action)
        printObs(new_observation)
        if done:
            print(f"Result: {reward}")
            break
        else:
            action = getAction(Q,new_observation,epsilon)

#for e in range(10):
#    followQ(qLQ)



