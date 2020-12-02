#Importing the Libraries
import os
import cv2
import glob
import gym
import vizdoomgym
import matplotlib.pyplot as plt
from collections import Counter
from gym.wrappers import Monitor

#Printing the input shape and number of possible actions.
env = gym.make('VizdoomCorridor-v0')
action_num = env.action_space.n
print("Number of possible actions: ", action_num)
state = env.reset()
state, reward, done, info = env.step(env.action_space.sample())
print(state.shape)
env.close()

#Displaying a frame of the environment just to see how it looks.
'''
observation = env.reset()
plt.imshow(observation)
plt.show()
print(observation.shape)
'''

#Making a helper function for the visualization.
def wrap_env(env):
  env = Monitor(env, './vid',video_callable=lambda episode_id:True, force=True)
  return env

#Running the AI on one episode.
gym_env = PreprocessImage(gym.make('VizdoomCorridor-v0'), width = 256, height = 256, grayscale = True)
gym_env = wrappers.Monitor(gym_env, "videos", force = True)
environment = wrap_env(gym_env)
done = False
observation = environment.reset()
new_observation = observation
actions_counter = Counter()
prev_input = None
img_array=[]
while True:
    # Feeding the game screen and getting the Q values for each action
    actions = cnn(Variable(Variable(torch.from_numpy(observation.reshape(1, 1, 256, 256)))))
    # Getting the action
    action = np.argmax(actions.detach().numpy(), axis=-1)
    actions_counter[str(action)] += 1 
    # Selecting the action using epsilon greedy policy
    environment.render()
    observation = new_observation        
    # Now performing the action and moving to the next state, next_obs, receive reward
    new_observation, reward, done, _ = environment.step(action)
    img_array.append(new_observation)
    if done: 
        # observation = env.reset()
        break
environment.close()

#Outputting the video of the gameplay.
folder_name = "gameplay_frames"
video_name = "agent_gameplay.avi"

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

for i in range(len(img_array)):
    plt.imsave("{}/{}_frame.jpg".format(folder_name, i), img_array[i].reshape(256, 256))

files = glob.glob(os.path.expanduser("{}/*".format(folder_name)))
frames_array = []

for filename in sorted(files, key=lambda t: os.stat(t).st_mtime):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    frames_array.append(img)

out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for i in range(len(frames_array)):
    out.write(frames_array[i])

out.release()