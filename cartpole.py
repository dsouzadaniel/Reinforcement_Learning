# Cartpole Example from the OpenAI Gym Environment

# Libraries
import gym
import numpy as np
# Create Environments
env = gym.make('CartPole-v0')

bestreward = 0
rando_noise_scaling = 0.1
parameters = (np.random.rand(4)*2) - 1
for i_episode in range(1000):
    observation = env.reset()
    new_parameters = parameters + (((np.random.rand(4)*2) - 1)*rando_noise_scaling)
    total_reward = 0
    for t in range(200):
        env.render()
        # print(observation)
        action = 0 if np.matmul(observation,new_parameters)<0 else 1
        observation, reward, done, info = env.step(action)
        total_reward+=reward
        if done:
            print("Episode :",i_episode,"\tTotal_Reward  = ",total_reward)
            # print("Episode finished after {} timesteps".format(t + 1))
            break
    if total_reward > bestreward:
        parameters = new_parameters
        bestreward = total_reward
    if total_reward == 200:
        print(" Stabilized")
        print(" Good Weights are : ", parameters)
        break
print(" The Bests Reward :",bestreward)

#
# print(env.action_space)
#
# print(env.observation_space)

