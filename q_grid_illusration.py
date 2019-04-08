import numpy as np, torch, torch.nn as nn, cv2, random, tensorboardX, gym

#==============================================================================
# GRID WORLD

class GridWorld:
  def __init__(self, size):
    self.size = size
    self.world_rewards = np.zeros((size, size), np.float32)
    self.world_rewards[self.size//2, self.size//2] = 1
    self.actions = [[-1,0], [+1,0], [0,-1], [0,+1]]

  def reset(self):
    self.location = np.random.randint(0,self.size,(2,))
    return self.location.astype(np.float32)

  def step(self, action):
    self.location = np.clip(self.location + self.actions[action], 0, self.size-1)
    idx = tuple(self.location)
    return self.location, self.world_rewards[idx], self.world_rewards[idx] > 0

  def render(self):
    idx = tuple(self.location)
    world = cv2.cvtColor(self.world_rewards.astype(np.uint8)*255, cv2.COLOR_GRAY2BGR)
    world[idx] = [255,0,0]
    world = cv2.resize(world, dsize=None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST)

    cv2.namedWindow("world", cv2.WINDOW_NORMAL)
    cv2.imshow("world", world)
    cv2.waitKey(20)

#==============================================================================
# SETUP

S = 11
VI = 1
EI = 10

env = GridWorld(S)

q_network = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64,64), nn.ReLU(), nn.Linear(64, len(env.actions)))

opt = torch.optim.Adam(q_network.parameters(), lr=1e-4)

experience = []

BATCH = 256

writer = tensorboardX.SummaryWriter(comment="DQN")

#==============================================================================
# RUN

for episode_num in xrange(10**6):
  state = env.reset()
  done = False
  length = 0

  while not done:
    if episode_num % EI == 0: env.render()

    eps = 0.3
    if np.random.random() < eps:
      action = np.random.randint(len(env.actions))
    else:
      q_values = q_network(torch.as_tensor(state).float())
      action = torch.argmax(q_values, dim=0).item()

    next_state, reward, done = env.step(action)

    experience.append((state, action, reward, next_state, float(done)))
    state = next_state

    length += 1
    if length > S: done = True

    if len(experience) > BATCH:
      batch = random.sample(experience, BATCH)
      torch_state = torch.as_tensor([s for s,a,r,n,d in batch]).float().view(BATCH, -1)
      torch_act   = torch.as_tensor([a for s,a,r,n,d in batch]). long().view(BATCH, -1)
      torch_rew   = torch.as_tensor([r for s,a,r,n,d in batch]).float().view(BATCH, -1)
      torch_next  = torch.as_tensor([n for s,a,r,n,d in batch]).float().view(BATCH, -1)
      torch_done  = torch.as_tensor([d for s,a,r,n,d in batch]).float().view(BATCH, -1)

      q_all  = q_network(torch_state)
      q_act  = torch.gather(q_all, index=torch_act, dim=1)
      q_next = q_network(torch_next)
      q_max, q_idx = torch.max(q_next, dim=1, keepdim=True)

      q_target = torch_rew + 0.9 * (1-torch_done) * q_max

      q_loss = torch.mean((q_act - q_target.detach())**2)

      opt.zero_grad()
      q_loss.backward()
      opt.step()

  if episode_num % VI == 0:
    indices = torch.as_tensor([[x,y] for x in range(S) for y in range(S)]).float()
    q_values = q_network(indices)
    q_values, q_argmax = torch.max(q_values, dim=-1)
    qmn = torch.min(q_values)
    qmx = torch.max(q_values)

    img = np.zeros((S,S), np.uint8)
    for (x,y),q in zip(indices, q_values):
      img[int(x.item()), int(y.item())] = (q-qmn)/(qmx-qmn+1e-8)*255

    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    img = cv2.resize(img, dsize=None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
    cv2.namedWindow("q", cv2.WINDOW_NORMAL)
    cv2.imshow("q", img)
    cv2.waitKey(1)

  writer.add_scalar("GridWorld/length",  length, episode_num)
  writer.add_scalar("GridWorld/epsilon", eps,    episode_num)

