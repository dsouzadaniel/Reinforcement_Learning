# Cartpole Example from the OpenAI Gym Environment

# Libraries
import gym
import numpy as np
# Create Environments
env = gym.make('CartPole-v0')


bestParams = None
bestreward = 0
for i_episode in range(100000):
    observation = env.reset()
    parameters = (np.random.rand(4)*2) - 1
    total_reward = 0
    for t in range(100):
        env.render()
        # print(observation)
        action = 0 if np.matmul(observation,parameters)<0 else 1
        observation, reward, done, info = env.step(action)
        total_reward+=reward
        if done:
            print("Episode :",i_episode,"\tTotal_Reward  = ",total_reward)
            # print("Episode finished after {} timesteps".format(t + 1))
            break
    if total_reward > bestreward:
        bestParams = parameters
        bestreward = total_reward
    if total_reward > 200:
        print(" Stabilized")
        print(" Good Weights are : ", parameters)
        break
print(" The Best Params were :",bestParams," with Reward :",bestreward)

#
# print(env.action_space)
#
# print(env.observation_space)