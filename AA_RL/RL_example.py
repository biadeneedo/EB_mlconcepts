# RL Repository
# https://rubikscode.net/2019/07/08/deep-q-learning-with-python-and-tensorflow-2-0/
# https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/

import gym
from IPython.display import clear_output
from time import sleep
import random
import numpy as np
from collections import deque
import progressbar
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam

env = gym.make("Taxi-v3").env
print("Action Space {}".format(env.action_space)) # print all the possible action spaces
print("State Space {}".format(env.observation_space)) # print the number of all the possible states
env.render() # print the render function, which is a picture in this case (Render is not the reward)
env.s #print the state

state = env.encode(3, 1, 2, 0) # (taxi row, taxi column, passenger index, destination index)
print("State:", state)
state = env.s
env.render()

# The reward dictionary calculates which are all the possible rewards
# for every possible action to take, starting from a certain state 
# This reward dictionary has the structure: {action: [(probability, nextstate, reward, done)]}
env.P[328]



# %% BRUTE-FORCE
# Brute force is similar to a simulation

env.action_space.sample()
epochs = 0
penalties, reward = 0, 0
frames = []
done = False

while not done:
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)

    if reward == -10:
        penalties += 1
    
    # Put each rendered frame into dict for animation
    frames.append({
        'frame': env.render(mode='ansi'),
        'state': state,
        'action': action,
        'reward': reward
        }
    )

    epochs += 1    
print("Timesteps taken: {}".format(epochs))
print("Penalties incurred: {}".format(penalties))

def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'].getvalue())
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)        
print_frames(frames)


# %% Q-LEARNING

""" Definition
Initialize the Q-table by all zeros.
1. Start exploring actions: For each state, select any one among all possible actions for the current state (S).
2. Travel to the next state (S') as a result of that action (a).
3. For all possible actions from the state (S') select the one with the highest Q-value.
4. Update Q-table values using the equation.
5. Set the next state as the current state.
6. If goal state is reached, then end and repeat the process.
QLEARNING EQUATION : Q(state,action)←(1−α)Q(state,action)+α(reward+γmaxaQ(next state,all actions))
"""
q_table = np.zeros([env.observation_space.n, env.action_space.n])

""" Hyperparameters
# A high value for the discount factor (close to 1) captures the long-term effective award, whereas, a discount factor of 0 makes our agent consider only immediate reward, hence making it greedy.
# Ideally, all three should decrease over time because as the agent continues to learn, it actually builds up more resilient priors;
# alpha is the learning rate, which is the extent to which our Q-values are being updated in every iteration
# gamma is the discount factor and it determines how much importance we want to give to future rewards
# epsilon makes the model take random actions without considering the highest Qvalues
"""
alpha = 0.1 
gamma = 0.6 
epsilon = 0.1

# 1. Training 
# This steps aims to train the q TABLE
for i in range(1, 100001):
    state = env.reset()
    epochs, penalties, reward, = 0, 0, 0
    done = False
     
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state]) # Exploit learned values

        next_state, reward, done, info = env.step(action) 
        
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1
        
    if i % 100 == 0: # divides i by 100 and if there is no remainder (0) then do something 
        clear_output(wait=True)
        print(f"Episode: {i}")
print("Training finished.\n")

# 2. Testing
# Evaluate agent's performance after Q-learning - TAKE DECISIONS BASED ON THE TRAINED Q-TABLE
total_epochs, total_penalties = 0, 0
episodes = 100
for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    done = False
    
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")


# %% DEEP Q-LEARNING (DQN)

"""
Experience Replay is the new q table
Steps:
1. Provide the state of the environment to the agent. 
   The agent uses Target Network and Q-Network to get the Q-Values of all possible actions in the defined state.
2. Pick the action a, based on the epsilon value.
   Meaning, either select a random action (exploration) or select the action with the maximum Q-Value (exploitation).
3. Perform action a
4. Observe reward r and the next state s’
5. Store these information in the experience replay memory <s, s’, a, r>
6. Sample random batches from experience replay memory and perform training of the Q-Network.
7. Each Nth iteration, copy the weights values from the Q-Network to the Target Network.
8. Repeat steps 2-7 for each episode
"""


def build_compile_model():
    model = Sequential()
    model.add(Embedding(state_size, 10, input_length=1))
    model.add(Reshape((10,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer = Adam(learning_rate= 0.01))
    return model

def act(state):
    if np.random.rand() <= epsilon:
        return env.action_space.sample()
    q_values = q_network.predict(state)
    return np.argmax(q_values[0])

def store(state, action, reward, next_state, terminated):
    experience_replay.append((state, action, reward, next_state, terminated))

def retrain(batch_size):
    minibatch = random.sample(experience_replay, batch_size)
    
    for state, action, reward, next_state, terminated in minibatch:
        
        target = q_network.predict(state)
        
        if terminated:
            target[0][action] = reward
        else:
            t = target_network.predict(next_state)
            target[0][action] = reward + gamma * np.amax(t)
        
        q_network.fit(state, target, epochs=1, verbose=0)

def alighn_target_model():
       target_network.set_weights(q_network.get_weights())
        

# Reset the env & initialize variables
state_size = env.observation_space.n
action_size = env.action_space.n
experience_replay = deque(maxlen=2000)
gamma = 0.6
epsilon = 0.1
q_network = build_compile_model()
target_network = build_compile_model()
q_network.summary()
target_network.summary()


for e in range(0, 100):
    print(e)
    
    state = env.reset()
    state = np.reshape(state, [1, 1])
    reward = 0
    terminated = False
    
    action = act(state) 
    
    next_state, reward, terminated, info = env.step(action) 
    next_state = np.reshape(next_state, [1, 1])
    store(state, action, reward, next_state, terminated)
    state = next_state
    
    if terminated != False:
        alighn_target_model()
        break
            
    if len(experience_replay) > 32: #batch size
        print('retrain')
        retrain(32)
        

# 2. Testing
# Evaluate agent's performance after Q-learning - TAKE DECISIONS BASED ON THE TRAINED Q-TABLE
total_epochs, total_penalties = 0, 0
episodes = 100

for e in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    done = False
    
    while not done:
        action = np.argmax(experience_replay[state])
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs


