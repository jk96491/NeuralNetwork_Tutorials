import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from Utils import convertToTensorInput

env = gym.make("CartPole-v0")
env.reset()

input_size = 4
output_size = 2


class RL_MODEL(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, state):
        actions = self.linear(state)
        return actions

random_episode = 0
reward_sum = 0
learning_rate = 0.1

num_episode = 2000
dis = 0.9
rList = []

model = RL_MODEL()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for i in range(num_episode):
    e = 1. / ((1 / 10) / + 1)
    rAll = 0
    step_count = 0
    s = env.reset()
    done = False

    while not done:
        step_count += 1
        Qs = model(convertToTensorInput(s, input_size))

        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else:
            _, action = torch.max(Qs, 1)
            action = action.data[0].item()

        new_state, reward, done, _ = env.step(action)

        Q1 = model(convertToTensorInput(new_state, input_size))
        maxQ1 = torch.max(Q1.data)

        targetQ = Variable(Qs.data, requires_grad=False)

        if done:
            targetQ[0, action] = reward
        else:
            targetQ[0, action] = reward + torch.mul(maxQ1, dis)

        output = model(convertToTensorInput(s, input_size))
        loss = torch.mean((output - targetQ) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rList.append(step_count)
        state = new_state

        print("Episode: {} steps: {}".format(i, step_count))

        if len(rList) < 10 and np.mean(rList[-10:]) > 500:
            break

    observation = env.reset()
    reward_sum = 0

    while True:
        env.render()
        x = np.reshape(observation, [1, input_size])
        Qs = model(torch.FloatTensor(x))

        _, action = torch.max(Qs, 1)
        action = action.data[0].item()

        observation, reward, done, _ = env.step(action)
        reward_sum += reward

        if done:
            print("Total score: {}".format(reward_sum))
            break

env.close()