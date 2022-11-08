class A2CModel(A2C):
  def __init__(self,
               env: gym.Env,
               **kwargs):
    # **kwargs will pass our arguments on to A2C
    super().__init__(env=env,
                     **kwargs)
    self.feature_net = nn.Sequential(
        nn.Conv2d(
            in_channels=self.observation_space.shape[0],
            out_channels=32,
            kernel_size=8,
            stride=4
        ),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=4,
            stride=2
        ),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1
        ),
        nn.Flatten(),
        nn.ReLU(),
    )
    # Dimensionality of the embedded feature space
    feature_shape = self._get_output_shape(self.feature_net)
    self.actor = nn.Sequential(
        nn.Linear(feature_shape[1], 512),
        nn.ReLU(),
        nn.Linear(512, self.action_space.n),
        nn.Softmax(dim=1),
    )
    self.critic = nn.Sequential(
        nn.Linear(feature_shape[1], 512),
        nn.ReLU(),
        nn.Linear(512, 1),
    )

    self.save_hyperparameters()

  def forward(self, x: torch.Tensor):
    x = self.feature_net(x)
    out = self.actor(x)
    dist = distributions.Categorical(probs=out)
    return dist, self.critic(x).flatten()

  def predict(self, x: torch.Tensor, deterministic: bool = False, **kwargs):
    x = self.feature_net(x)
    out = self.actor(x)
    if deterministic:
      out = torch.max(out, dim=1)[1]
    else:
      out = distributions.Categorical(probs=out).sample()
    return out.cpu().numpy()

  def configure_optimizers(self):
    optimizer = torch.optim.RMSprop(self.parameters(), lr=1e-3, eps=1e-5)
    return optimizer