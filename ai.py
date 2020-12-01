#Importing the libraries.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

#Importing packages for OpenAI and Doom.
#Importing the environment for the game.
import gym 
from gym.wrappers import SkipWrapper
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete

#Importing python files.
import experience_replay, image_preprocessing

# Building the Ai
#Making the brain

class Convolutional_Neural_Network(nn.Module):
    
    def __init__(self, num_of_actions):
        
        super(Convolutional_Neural_Network, self).__init__()
        self.convo_connection_1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5)  
        self.convo_connection_2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3)
        self.convo_connection_3 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 2)
        self.full_connection_1 = nn.Linear(in_features = self.count_neurons(1, 80, 80) , out_features = 40)
        self.full_connection_2 = nn.Linear(in_features = 40 , out_features = num_of_actions)
        
    def count_neurons(self, dimensions_of_image):
        #Create a fake image
        x = Variable(torch.rand(1, *dimensions_of_image))
        #Apply Convolution Layers to our images.
        x = F.relu(F.max_pool2d(self.convo_connection_1(x), 3, 2)) 
        x = F.relu(F.max_pool2d(self.convo_connection_2(x), 3, 2)) 
        x = F.relu(F.max_pool2d(self.convo_connection_3(x), 3, 2)) 
        #Flatten the layers using size function
        return x.data.view(1, -1).size(1)
    #Forward the input between the fully connected layers.
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.convo_connection_1(x), 3, 2)) 
        x = F.relu(F.max_pool2d(self.convo_connection_2(x), 3, 2)) 
        x = F.relu(F.max_pool2d(self.convo_connection_3(x), 3, 2)) 
        #Flatten the channels
        x = x.view(x.size(0), -1)
        x = F.relu(self.full_connection_1(x))
        x = self.full_connection_2(x)
        return x

#Making the body
class Body_Softmax(nn.Module):
    
    def __init__(self, temperature):
        super(Body_Softmax, self).__init__()
        self.temperature = temperature
    
    def forward(self, output_signals):
        probabilities_of_actions = F.softmax( output_signals * self.temperature) 
        action = probabilities_of_actions.multinomial() 
        return action
    
#Making an AI
class AI:
    
    def __init__(self, brain, body):
        self.brain = brain
        self.body = body
    
    #Start the propagation process
    def __call__(self, input_images):
        input = Variable(torch.from_numpy(np.array(input_images, dtype = np.float32)))
        output = self.brain(input)
        actions = self.body(output)
        return actions.data.numpy()
    
#Training the AI with DEEP Q LEARNING
doom_env = image_preprocessing.PreprocessImage(SkipWrapper(4)(ToDiscrete("minimal")(gym.make("ppaquette/DoomCorridor-v0"))), width = 80, height = 80, grayscale = True)
doom_env = gym.wrappers.Monitor(doom_env, "videos", force = True)
num_of_actions = doom_env.action_space.n
        
#Building the AI
cnn = Convolutional_Neural_Network(num_of_actions)
softmax_body = Body_Softmax(temperature = 1.0)
ai = AI(brain = cnn, body = softmax_body)

#Setting up experience Replay
n_steps = experience_replay.NStepProgress(env = doom_env, ai = ai, n_step = 10)
memory = experience_replay.ReplayMemory(n_steps = n_steps,capacity = 100000 )

# Implementing Eligibility Trace
# Implemented using Asynchronous n-step Q-Learning-pseudocode for each actor-learner thread.
def eligibility_trace(batch):
    gamma = 0.99
    inputs = []
    targets = []
    
    for series in batch:
        input = Variable(torch.from_numpy(np.array([series[0].state, series[-1].state], dtype = np.float32)))
        output = cnn(input)
        cumul_reward = 0.0 if series[-1].done else output[1].data.max()
        
        for step in reversed(series[:-1]):
            cumul_reward = step.reward + gamma * cumul_reward
        
        state = series[0].state
        target = output[0].data
        target[series[0].action] = cumul_reward
        inputs.append(state)
        targets.append(target)
    return torch.from_numpy(np.array(inputs, dtype = np.float32)), torch.stack(targets)

# Making the moving average on 100 steps
class Moving_Average:
    def __init__(self, size):
        self.list_of_rewards = []
        self.size = size
        
    def  add(self, rewards):
        
        if isinstance(rewards, list):
            self.list_of_rewards += rewards
        else:
            self.list_of_rewards.append(rewards)
        
        while len(self.list_of_rewards) > self.size:
            del self.list_of_rewards[0]
    
    def average(self):
        return np.mean(self.list_of_rewards)
ma = Moving_Average(100)

# Training the AI
loss = nn.MSELoss()
optimizer = optim.Adam(cnn.parameters(), lr = 0.001)
nb_epochs = 100
for epoch in range(1, nb_epochs + 1):
    memory.run_steps(200)
    for batch in memory.sample_batch(128):
        inputs, targets = eligibility_trace(batch)
        inputs, targets = Variable(inputs), Variable(targets)
        predictions = cnn(inputs)
        loss_error = loss(predictions, targets)
        optimizer.zero_grad()
        loss_error.backward()
        optimizer.step()
    rewards_steps = n_steps.rewards_steps()
    ma.add(rewards_steps)
    avg_reward = ma.average()
    print("Epoch: %s, Average Reward: %s" % (str(epoch), str(avg_reward)))
      
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    