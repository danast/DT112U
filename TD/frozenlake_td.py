import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint

env = gym.make('FrozenLake8x8-v0', is_slippery=False)

##env = gym.make('FrozenLake8x8-v0')

def makeQ(env):
    s_count = env.nS
    a_count = env.action_space.n
    isize = s_count*a_count
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    return Q
    Q = (np.random.rand(isize)-0.5)/10000
    Q.shape = (s_count, a_count)

    for s in range(env.nS):
        p = env.P[s]
        act_done = [p[k][0][3] for k in p]
        if all(act_done):
            Q[s] = np.zeros(env.nA)

    return Q

rg = np.random.default_rng()


def getAction(Q, s_index, epsilon):
    return np.argmax(Q[s_index] + np.random.rand(env.action_space.n)*epsilon)
    if rg.random() > epsilon:
        # Choose greedy action
        return np.argmax(Q[s_index])
    else:
        # Choose random action
        return rg.integers(0,Q.shape[1])

def getAChar(a):
    achars = ('^','>','v','<')
    return achars[a]

def printQ(Q):
    out = ""
    for i in range(Q.shape[0]):
        position = np.unravel_index(i, (env.desc.shape))
        actChar = getAChar(np.argmax(Q[i]))
        out += f' {actChar} '
        if position[1] == env.desc.shape[1]-1:
            out += "\n"
    print(out)


def singleSarsa(env, terminal_state, epsilon, alpha, gamma, maxEpisodes, epsilonDecrease=False):
    episodeCounts = np.zeros(maxEpisodes)
    totalRewards = np.zeros(maxEpisodes) 
    episode = 0
    delta = 0.1
    Q = makeQ(env)
    Q[terminal_state] = np.zeros(4)
    while episode < maxEpisodes:
        done = False
        count = 0
        totalReward = 0
        observation = env.reset()
        action = getAction(Q,observation,epsilon)
        while not done:
            env.render()
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
    Q = makeQ(env)
    #Q[terminal_state] = np.zeros(4)
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
        #if totalReward > 0:
        #    print(Q)

        if epsilonDecrease and episode % 500 == 0:
            epsilon = epsilon/2

    return (episodeCounts, Q, totalRewards)    



def doubleSarsa(env, terminal_state, epsilon, alpha, gamma, maxEpisodes, epsilonDecrease=False):
    episodeCounts = np.zeros(maxEpisodes)
    totalRewards = np.zeros(maxEpisodes) 
    episode = 0
    Q1 = makeQ(env)
    Q2 = makeQ(env)
    Q1[terminal_state] = np.zeros(4)
    Q2[terminal_state] = np.zeros(4)
    #Q2 = np.copy(Q1)
    while episode < maxEpisodes:

        done = False
        count = 0
        totalReward = 0

        observation = env.reset()
        action = getAction(Q1+Q2,observation,epsilon)

        while not done:
            count = count+1
            new_observation, reward, done, _ = env.step(action)
            totalReward += reward
            new_action = getAction(Q1+Q2,new_observation,epsilon)
            if rg.random() < 0.5:
                Qas = Q1[observation][action]
                Q1[observation][action] = Qas + alpha*(reward + gamma*Q2[new_observation][new_action] - Qas)
            else:
                Qas = Q2[observation][action]
                Q2[observation][action] = Qas + alpha*(reward + gamma*Q1[new_observation][new_action] - Qas)
            observation = new_observation
            action = new_action
            if done:
                break
        episodeCounts[episode] = count
        totalRewards[episode] = totalReward
        episode = episode+1

        if epsilonDecrease and episode % 50 == 0:
            epsilon = epsilon/2

    return (episodeCounts, Q1+Q2, totalRewards)

epsilon = 0.1
alpha = 0.628
gamma = 0.9
maxEpisodes = 5000
terminal_state = env.nS-1
delta = -1

#ecSSarsa, sSarsaQ, trSSarsa = singleSarsa(env, terminal_state, epsilon, alpha, gamma, maxEpisodes, False)
#ecSSarsa2, sSarsaQ2, trSSarsa2 = singleSarsa(env, terminal_state, epsilon, alpha+0.05, gamma, maxEpisodes, False)

ecQl1, qLQ, trQl1 = qLearning(env, terminal_state, epsilon, alpha, gamma, maxEpisodes, True)

ecDSarsa, dSarsaQ, trDSarsa = doubleSarsa(env, terminal_state, epsilon, alpha, gamma, maxEpisodes, True)

pprint(qLQ)

#print("Final policy Sarsa:")   
#printQ(sSarsaQ)
#print("Final policy Double Sarsa:")   
#printQ(dSarsaQ)
print("Final policy QLearning:")   
printQ(qLQ)
print(ecQl1)
print(trQl1)
print(trQl1[-100:-1])




printQ(dSarsaQ)
print(ecDSarsa)
print(trDSarsa[-100:-1])
#pprint(env.P)

# counts = pd.Series(ecSSarsa)
# cmva = counts.rolling(1).mean().values

# counts2 = pd.Series(ecDSarsa)
# cmva2 = counts2.rolling(1).mean().values

# fig, ax = plt.subplots()
#plt.subplot(211)
#plt.plot(cmva, color='blue')
#plt.plot(cmva2, color='red')
#plt.subplot(212)
# totR1 = pd.Series(trSSarsa)
# rmva1 = totR1.rolling(10).mean().values
# totR2 = pd.Series(trQl1)
# rmva2 = totR2.rolling(10).mean().values
# #totR2 = pd.Series(trDSarsa)
# #rmva2 = totR2.rolling(10).mean().values
# plt.plot(rmva1, color='blue')
# #plt.plot(rmva2, color='green')
# plt.plot(rmva2, color='red')
# plt.ylim(-100,0)
# #plt.show()

#pprint(trDSarsa)