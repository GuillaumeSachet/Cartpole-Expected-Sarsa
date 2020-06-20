import numpy as np
import tensorflow as tf
from replay_buffer import ReplayBuffer
import pickle
from softmax import Softmax

class DeepExpectedSarsaAgent():
    def __init__(self, agent_config, model_config):
        self.model_type = agent_config['model']
        self.num_actions = agent_config['num_actions']
        self.model = self.model_type(model_config)
        self.model.summary()
        self.model2 = self.model_type(model_config)
        self.gamma = agent_config['gamma']
        self.tau = agent_config['tau']
        self.replay_buffer = ReplayBuffer(agent_config['replay_buffer_size'])
        self.total_timestep = 0
        self.last_state = None
        self.last_action = None
            
    def policy(self, state):
        action_values = self.model.predict(state)
        probs_batch = Softmax(action_values,self.tau)
        action = np.random.choice(self.num_actions, p=probs_batch)
        return action
    
    def experience_replay(self, batch_size, num_replay,verb = 0):
        self.total_timestep += num_replay
        if len(self.replay_buffer.memory) > batch_size*4:
            self.model2.set_weights(self.model.get_weights())
            for _ in range(num_replay):
                states,actions,rewards,next_states,terminals = self.replay_buffer.sample(batch_size)
                q_next_mat = self.model2.predict(next_states)
                probs_mat = Softmax(q_next_mat,self.tau)
                v_next_vec = np.sum(probs_mat*q_next_mat,1)*(1 - terminals)
                target_vec = rewards + self.gamma*v_next_vec

                q_mat = self.model.predict(states)
                x_batch = states
                y_batch = q_mat
                for index,a in enumerate(actions):
                    y_batch[index][a] = target_vec[index]
                self.model.fit(x_batch, y_batch, batch_size=batch_size, verbose=verb,epochs=1)
            print('Experience replay done for {} replays'.format(num_replay))
        else : print('Not enough memory in replay buffer')
        
    def reset(self, state):
        self.last_state = state
        self.last_action = self.policy(self.last_state)
        return self.last_action
    
    def step(self, reward, state,done):
        action = self.policy(state)
        self.replay_buffer.add(self.last_state, self.last_action, reward, state, done)
        self.last_state = state
        self.last_action = action
        return action
    
    def save_weights(self, file):
        filehandler = open(file+'_replay_buffer', 'wb') 
        pickle.dump(self.replay_buffer, filehandler)
        filehandler = open(file+'_total_timestep', 'wb') 
        pickle.dump(self.total_timestep, filehandler)
        self.model.save_weights(file)
        print('Saved')
    
    def load_weights(self,file):
        filehandler = open(file+'_replay_buffer', 'rb') 
        self.replay_buffer = pickle.load(filehandler)
        filehandler = open(file+'_total_timestep', 'rb') 
        self.total_timestep = pickle.load(filehandler) 
        self.model.load_weights(file)
        print('Loaded')