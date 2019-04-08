import torch, torch.nn as nn, numpy as np, random, gym, tensorboardX

env = gym.make("CartPole-v1")

pi = nn.Sequential(nn.Linear(env.observation_space.shape[0], 32), nn.ReLU(),
                   nn.Linear(32, env.action_space.n))

opt = torch.optim.Adam(pi.parameters(), lr=1e-4)

writer = tensorboardX.SummaryWriter(comment="PG-CartPole")

for episode_num in xrange(10**6):
  state = env.reset()
  done = False

  log_probs = []
  rewards = []

  while not done:
    if episode_num > 10000: env.render()

    logits = pi(torch.as_tensor(state).float())
    distribution = torch.distributions.Categorical(logits=logits)
    action = distribution.sample()
    log_prob = distribution.log_prob(action)
    action = action.item()

    next_state, reward, done, info = env.step(action)
    state = next_state

    log_probs.append(log_prob)
    rewards.append(reward)

  returns = torch.as_tensor([sum(rewards[t:]) for t in range(len(rewards))]).float().view(-1, 1)
  log_probs = torch.stack(log_probs).view(-1, 1)

  loss = torch.sum(-log_probs * returns)

  opt.zero_grad()
  loss.backward()
  opt.step()


  writer.add_scalar("CartPole-v1/reward", sum(rewards), episode_num)

