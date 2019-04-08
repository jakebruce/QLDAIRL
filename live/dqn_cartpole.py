import torch, torch.nn as nn, numpy as np, gym, random, time, tensorboardX


env = gym.make("CartPole-v1")


q_network = nn.Sequential(nn.Linear(env.observation_space.shape[0], 32), nn.ReLU(),
                          nn.Linear(32, env.action_space.n))

q_opt = torch.optim.Adam(q_network.parameters(), lr=1e-3)

memory = []

writer = tensorboardX.SummaryWriter(comment="DQN_CartPole")

for episode_num in xrange(10**6):
  state = env.reset()
  done = False
  total_reward = 0

  while not done:
    #env.render()

    if np.random.random() < 0.3:
      action = np.random.randint(2)
    else:
      q_vals = q_network(torch.as_tensor(state).float())
      action = torch.argmax(q_vals, dim=0).item()

    next_state, reward, done, info = env.step(action)

    total_reward += reward

    memory.append((state, action, reward, next_state, done))

    state = next_state

    if len(memory) > 32:
      batch = random.sample(memory, 32)

      batch_state  = torch.as_tensor([s for s,a,r,n,d in batch]).float().view(32, -1)
      batch_action = torch.as_tensor([a for s,a,r,n,d in batch]). long().view(32, -1)
      batch_reward = torch.as_tensor([r for s,a,r,n,d in batch]).float().view(32, -1)
      batch_next   = torch.as_tensor([n for s,a,r,n,d in batch]).float().view(32, -1)
      batch_done   = torch.as_tensor([d for s,a,r,n,d in batch]).float().view(32, -1)


      q_all = q_network(batch_state)
      q_act = torch.gather(q_all, index=batch_action, dim=1)

      q_all_next = q_network(batch_next)
      q_max, q_idx = torch.max(q_all_next, dim=1, keepdim=True)

      q_target = batch_reward + 0.95 * (1-batch_done) * q_max

      q_loss = torch.mean((q_act - q_target.detach())**2)

      q_opt.zero_grad()
      q_loss.backward()
      q_opt.step()

  writer.add_scalar("CartPole-v1/reward", total_reward, episode_num)

