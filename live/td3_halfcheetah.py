import torch, torch.nn as nn, numpy as np, gym, random, tensorboardX, time

#=============================
# CLASSES

class TD3Component(nn.Module):
  def __init__(self, in_size, out_size, tanh):
    super(TD3Component, self).__init__()
    modules = [nn.Linear(in_size, 300), nn.ReLU(), nn.Linear(300, 400), nn.ReLU(), nn.Linear(400, out_size)]
    if tanh: modules.append(nn.Tanh())
    self.sequential = nn.Sequential(*modules)
  def forward(self, x): return self.sequential(x)

#=============================
# SETUP

env = gym.make("HalfCheetah-v1")


pi   = TD3Component(env.observation_space.shape[0], env.action_space.shape[0], tanh=True)
pi_t = TD3Component(env.observation_space.shape[0], env.action_space.shape[0], tanh=True)

q1   = TD3Component(env.observation_space.shape[0] + env.action_space.shape[0], 1, tanh=False)
q1_t = TD3Component(env.observation_space.shape[0] + env.action_space.shape[0], 1, tanh=False)

q2   = TD3Component(env.observation_space.shape[0] + env.action_space.shape[0], 1, tanh=False)
q2_t = TD3Component(env.observation_space.shape[0] + env.action_space.shape[0], 1, tanh=False)


opt_q  = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=1e-3)
opt_pi = torch.optim.Adam(pi.parameters(), lr=1e-3)

pi_t.load_state_dict(pi.state_dict())
q1_t.load_state_dict(q1.state_dict())
q2_t.load_state_dict(q2.state_dict())

memory = []

writer = tensorboardX.SummaryWriter(comment="TD3-HalfCheetah")

#=============================
# RUN

step = 0
for episode_num in xrange(10**6):
  state = env.reset()
  done = False
  total_reward = 0

  while not done:
    step += 1
    if step > 10000: env.render()

    if step < 10000:
      action = np.random.random(env.action_space.shape)*2-1
    else:
      action = pi(torch.as_tensor(state).float()) + torch.randn(env.action_space.shape)*0.1
      action = torch.clamp(action, -1, 1)
      action = action.data.numpy()

    next_state, reward, done, info = env.step(action)
    total_reward += reward

    memory.append((state, action, reward, next_state, done))

    state = next_state

    if len(memory) > 256 and step > 10000:
      batch = random.sample(memory, 256)

      torch_state  = torch.as_tensor([s for s,a,r,n,d in batch]).float().view(256, -1)
      torch_action = torch.as_tensor([a for s,a,r,n,d in batch]).float().view(256, -1)
      torch_reward = torch.as_tensor([r for s,a,r,n,d in batch]).float().view(256, -1)
      torch_next   = torch.as_tensor([n for s,a,r,n,d in batch]).float().view(256, -1)
      torch_done   = torch.as_tensor([d for s,a,r,n,d in batch]).float().view(256, -1)

      state_action = torch.cat([torch_state, torch_action], dim=1)

      q1_estimate = q1(state_action)
      q2_estimate = q2(state_action)

      proposed_next_action = pi_t(torch_next)
      next_state_action = torch.cat([torch_next, proposed_next_action], dim=1)

      q1_next = q1_t(next_state_action)
      q2_next = q2_t(next_state_action)
      q_next  = torch.min(q1_next, q2_next)

      q_target = torch_reward + 0.99 * (1-torch_done) * q_next

      q1_loss = torch.mean((q1_estimate - q_target.detach())**2)
      q2_loss = torch.mean((q2_estimate - q_target.detach())**2)

      q_loss = q1_loss + q2_loss

      opt_q.zero_grad()
      q_loss.backward()
      opt_q.step()

      if step % 2 == 0:
        proposed_action = pi(torch_state)
        proposed_state_action = torch.cat([torch_state, proposed_action], dim=1)
        policy_loss = torch.mean(-q1(proposed_state_action))


        opt_pi.zero_grad()
        policy_loss.backward()
        opt_pi.step()

        for param_online, param_target in zip(pi.parameters(), pi_t.parameters()):
          param_target.data = param_online.data*0.005 + param_target.data*0.995
        for param_online, param_target in zip(q1.parameters(), q1_t.parameters()):
          param_target.data = param_online.data*0.005 + param_target.data*0.995
        for param_online, param_target in zip(q2.parameters(), q2_t.parameters()):
          param_target.data = param_online.data*0.005 + param_target.data*0.995

  writer.add_scalar("HalfCheetah-v1/reward", total_reward, episode_num)

