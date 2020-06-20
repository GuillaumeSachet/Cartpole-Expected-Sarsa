import gym
import time

class GymRunner:
    def __init__(self, env_id, monitor_dir, max_timesteps=100000):
        self.monitor_dir = monitor_dir
        self.max_timesteps = max_timesteps
        self.env = gym.make(env_id)
    
    def run(self, agent, num_episodes,timestep, save):
        if save:
            self.env = gym.wrappers.Monitor(self.env, self.monitor_dir+'/eval',
                       video_callable=lambda ep_id : True, force = True, mode = 'evaluation')
        for episode in range(num_episodes):
            state = self.env.reset().reshape(1, self.env.observation_space.shape[0])
            for t in range(timestep):
                action = agent.policy(state)
                state, _, done, _ = self.env.step(action)
                state = state.reshape(1, self.env.observation_space.shape[0])
                self.env.render()
                if save and done:
                    print('Done : Recording stopped at timestep {}'.format(t))
                    break
            

    def train(self, agent, num_episodes,monitor,render_freq = 10):
        if monitor:
            self.env = gym.wrappers.Monitor(self.env, self.monitor_dir+'/train', 
                               video_callable=lambda ep_id :(ep_id+1) % render_freq == 0 or ep_id == 0,
                                            resume = True, mode = 'training')  
        for episode in range(num_episodes):
            state = self.env.reset().reshape(1, self.env.observation_space.shape[0])
            total_reward = 0
            time_start = time.time()
            action = agent.reset(state)
            for t in range(self.max_timesteps):
                state, reward, done, _ = self.env.step(action)
                state = state.reshape(1, self.env.observation_space.shape[0])
                action = agent.step(reward,state,done)
                total_reward += reward
                if done:
                    break
            print("episode: {}/{} | score: {} | time: {:.2f}".format(
            episode + 1, num_episodes, total_reward, time.time()-time_start))
    
    def close(self):
        self.env.close()