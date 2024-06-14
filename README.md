# Tetris Agent using Deep Q Learning

## Packages to install
```
pip install gymnasium torch matplotlib numpy pygame
```

## The Enviroment

* The agent is going to act in the tetris game we made using pygame, all the code of the game is in [Tetris.py](Tetris.py) and it's used to create the custom enviroment in [tetrisEnv.py](tetrisEnv.py) 

```python
        #score / reward
            
        if self._action_to_direction[idx_action] == 'drop':
            self.game.score += (drop_height) - 6
            
        if self.game.score > 5000:
            self.game.score +=500
            terminated = True

        if self.game.score != self.actualscore:
            reward = self.game.score - self.actualscore
            self.actualscore=self.game.score
        else:
            reward = 0
```

* this part of the [tetrisEnv.py](tetrisEnv.py) reward the agent for the lower the piece that was placed.

* The game also increase the score a lot when lines are cleared, a Tspin is made or a perfect clear happend.

## The Model
* The model has 2 Convolutional 2D layers + ReLU, then a flatten layer. Then this is concatenated with other 15 parameters to process it with the a ReLU layer and then a linear for the output:
```python
    def __init__(self, freeze=False):
        super().__init__()
        # Conolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        
        self.field_flatter = nn.Flatten()
        
        # Linear layers
        self.network = nn.Sequential(
            nn.Linear(3599,out_features=512),
            nn.ReLU(),
            nn.Linear(512, 8)
        )
```
the model is defined [model2.py](model2.py)

## The Training with Deep Q Learing
* The memory: It works with a memory as a deque with a max cap. Every state, action, reward, state after and done (if the game ended that moment) is being stored as 1 element in that memeory.
```python
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
```
* Then at the end of an episode, the agent takes (bach size) number of samples of the memory to make predictions, calculate Q values and use them to get a loss to preform backpropagation with the ADAM optimaizer.

```python
for state, action, reward, next_state, done in minibatch:
    field = torch.FloatTensor(state['field']).unsqueeze(0).unsqueeze(0).to(self.get_device())
    other_state = torch.FloatTensor(state['other_state']).unsqueeze(0).to(self.get_device())
    
    next_field = torch.FloatTensor(next_state['field']).unsqueeze(0).unsqueeze(0).to(self.get_device())
    next_other_state = torch.FloatTensor(next_state['other_state']).unsqueeze(0).to(self.get_device())
    target = self.model(field, other_state)#get prediction of action vetor with the main model
    if done:#check if is a terminal state
        target[0][action] = reward #if it is, sets the Q value for this action with the reward
    else:
        next_target = self.target_model(next_field, next_other_state)#get action prection for the nextstate with target model
        target[0][action] = reward + self.gamma * torch.max(next_target[0]).item() # sets the Q value with the formula: reward * discount * Q value of best action
    
        self.optimizer.zero_grad()
        loss = self.criterion(self.model(field, other_state), target)#get the loss with MSE of the model's prediction vs target Q values
        loss.backward()#backpropagation
        self.optimizer.step()
```


## The Hyperparameters
* For the training of the model using Deep Q Learing, The hyperparameters to adjust are the following:
```python
gamma=0.93, # The discount rate used in the Q formula
learning_rate=0.00001, # The factor of a step given during the adjust of weiths
epsilon=0.8, # The initial randomness
epsilon_min=0.075, # The minimum randomness
epsilon_decay=0.955, # The decay of the randomness
batch_size=69, # The amount of samples to take to learn at the end of and episode
max_memory=10_000, # The max number of samples that are stored in the deque
update_target=15, # The episodes to update the target model
epsiont_every = 10 # The episodes to play one episode free of randomness
```
the hyperparameters are declared in [agent.py](agent.py)

## The Data

* The shape of the 2 tensors to give to the model have this structure:
```
input:
tensor([[[[0., 0., 0., 0., 2., 2., 0., 0., 0., 0.],
          [0., 0., 0., 0., 2., 2., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 1., 1., 1., 1., 0., 0., 0.],
          [0., 0., 0., 1., 1., 1., 1., 0., 0., 0.],
          [0., 0., 0., 1., 1., 1., 1., 0., 0., 0.],
          [0., 0., 0., 0., 1., 1., 0., 0., 0., 0.],
          [0., 0., 0., 0., 1., 1., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
          [0., 0., 1., 0., 0., 1., 0., 0., 0., 0.],
          [0., 0., 1., 0., 0., 1., 0., 0., 0., 0.],
          [1., 0., 1., 1., 1., 1., 1., 0., 0., 0.],
          [1., 0., 1., 1., 1., 1., 1., 0., 0., 0.],
          [1., 1., 1., 0., 1., 1., 0., 0., 0., 0.],
          [1., 1., 1., 0., 1., 1., 0., 0., 0., 0.]]]], device='cuda:0')
tensor([[ 3.,  0.,  6.,  0.,  0.,  6.,  6.,  0.,  6.,  0.,  0., 73., 20., 64.,
         26.]], device='cuda:0')
```
* This the tensor form of the data recived the this function _get_obs() found in [tetrisEnv.py](tetrisEnv.py)
```python
    def _get_obs(self):
        obs = {
                "x_piece":self.game.figure.x, # horizontal position of the current piece (-2 to 7)
                "y_piece":self.game.figure.y, # vertical position of the current piece (0 to 17)
                "piece_type":self.game.figure.type, # the type of the current piece (0 to 7)
                "piece_rotation":self.game.figure.rotation, # the rotation of the current piece (-1 to 2)
                "hold":self.game.hold_piece.type, # the type of the hold piece (-1 to 7)
                "queue":self.game.queue, # the queue of the 5 next pieces (5 values of 0 to 7)
                "lines_cleared":self.game.cleared_lines, # the total number of lines cleared (>=0)
                "total_score": self.game.score, # the total gained score (>=0)
                "holes": self.game.holes(), # the total number of holes left in the field (>=0)
                "total_height": self.game.total_height(), # the sum of all the heights (>=0)
                "bumpiness":self.game.bumpiness() # the sum of the differences of height between columns (>=0)
            }
```

