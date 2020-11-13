import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
from Week10.Example.Model.frozenModel import RL_MODEL
import torch.optim as optim
from torch.autograd import Variable
import torch

register(
    id='FrozenLake-v3',
    entry_point = 'gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name':'4x4',
           'is_slippery':False}
)

env = gym.make('FrozenLake-v3')

observation_space = env.observation_space.n
action_space = env.action_space.n

model = RL_MODEL(observation_space, action_space)

optimizer = optim.Adam(model.parameters(), lr=0.001)

# discount 정의 => 미래의 reward를 현재의 reward 보다 조금 낮게 본다.
dis = 0.99

# 몇 번 시도를 할 것인가 (에피소드)
num_episodes = 2000

# 에피소드마다 총 리워드의 합을 저장하는 리스트
rList = []


def ConvertTensor(x, l):
    x = torch.LongTensor([[x]])
    one_hot = torch.FloatTensor(1,l)
    return Variable(one_hot.zero_().scatter_(1,x,1))


for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False

    # exploration의 확률 (decaying)
    e = 1. / ((i / 100) + 1)

    # Q learning 알고리즘
    while not done:
        # E-Greedy 알고리즘으로 action 고르기
        # 0 : LEFT, 1 : DOWN, 2: RIGHT, 3: UP
        StateTensor = ConvertTensor(state, observation_space)
        Qs = model(StateTensor)
        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else:
            _, action = torch.max(Qs, 1)
            action = action.data[0].item()

        new_state, reward, done, _ = env.step(action)

        new_stateTensor = ConvertTensor(new_state, observation_space)
        Q1 = model(new_stateTensor)
        maxQ1, _ = torch.max(Q1.data, 1)
        maxQ1 = torch.FloatTensor(maxQ1)

        targetQ = Variable(Qs.data, requires_grad=False)

        if done:
            targetQ[0, action] = reward
        else:
            targetQ[0, action] = reward + torch.mul(maxQ1, dis)

        output = model(StateTensor)
        loss = torch.mean((output - targetQ)**2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rAll += reward
        state = new_state
        env.render()

    rList.append(rAll)

print("Success rate : "+str(sum(rList) / num_episodes))
print("Final Q-Table Values")
print(model.linear.weight)

plt.bar(range(len(rList)), rList, color="blue")
plt.show()
