import pip
import gym
import imageio
import json
import PIL.Image

from tensorflow.compat import v1

from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import trajectory
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

class Dqnnet:
    def __init__(self):
        self.env_t = None
        self.env_e = None
        self.agent = None
        self.settings = {}
        self.policies = {}
        self.returns = []
    
    def conf_settings(self, readpath):
        if readpath == None:
            self.settings['iter'] = 2e4
            self.settings['ic_steps'] = 1e3
            self.settings['rb_cap'] = 1e5
            self.settings['fc_layer'] = (100,)
            self.settings['bsize'] = 64
            self.settings['lr'] = 1e-3
            self.settings['log_int'] = 200
            self.settings['eval'] = 10
            self.settings['eval_int'] = 1e3
        else:
            with open(readpath , 'r') as reader:
                self.settings = json.load(reader.read())

    def create_env(self):
        pip._internal.main(['install','-e','gym_my_maze'])
        env = gym.make('gym_my_maze:mymaze-v0')
        env.reset()
        PIL.Image.fromarray(env.render())
        self.env_t, self.env_e = tf_py_environment.TFPyEnvironment(env), tf_py_environment.TFPyEnvironment(env)

    def create_agent(self):
        q_net = q_network.QNetwork(self.env_t.observation_spec(), self.env_t.action_spec(), fc_layer_params=self.settings['fc_layer'])
        optimizer = v1.train.AdamOptimizer(learning_rate=self.settings['lr'])
        train_step_counter = v1.Variable(0)
        self.agent = dqn_agent.DqnAgent(self.env_t.time_step_spec(), self.env_t.action_spec(), q_network=q_net, optimizer=optimizer, td_errors_loss_fn=common.element_wise_squared_loss, train_step_counter=train_step_counter) 
        self.agent.initialize()

    def create_policy(self):
        self.policies['e'] = self.agent.policy
        self.policies['c'] = self.agent.collect_policy
        self.policies['r'] = random_tf_policy.RandomTFPolicy(self.env_t.time_step_spec(),self.env_t.action_spec())

    def compute_avg_return(self, environment, policy, num_episodes=10):
        total_return = 0.0
        for _ in range(num_episodes):
            time_step = environment.reset()
            episode_return = 0.0
            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return
        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]
    
    def collect_step(self, replay_buffer, environment, policy):
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        replay_buffer.add_batch(traj)


    def train(self):
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=self.agent.collect_data_spec, batch_size=self.settings['bsize'], max_length=self.settings['rb_cap'])
        for _ in range(self.settings['ic_steps']):
            self.collect_step(replay_buffer, self.env_t, self.policies['r'])
        dataset = replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=self.settings['bsize'], num_steps=2).prefetch(3)    
        iterator = iter(dataset)
        self.agent.train = common.function(self.agent.train)
        self.agent.train_step_counter.assign(0)
        avg_return = self.compute_avg_return(self.env_e, self.policies['e'], self.settings['eval'])
        self.returns = [avg_return]
        experience, _ = next(iterator)
        loss_t = self.agent.train(experience)
        step = self.agent.train_step_counter.numpy()
        if step % self.settings['log_int'] == 0:
            print('step = {0}: loss = {1}'.format(step, loss_t.loss))
        if step % self.settings['eval_int'] == 0:
            avg_return = self.compute_avg_return(self.env_e, self.policies['e'], self.settings['eval'])
            print('step = {0}: avg_return = {1}'.format(step, avg_return))
            self.returns.append(avg_return)

    def create_video(self, videopath):
        with imageio.get_writer(videopath, fps=60) as video:
            for _ in range(4):
                time_step = self.env_e.reset()
                video.append_data(self.env_e.render())
                while not time_step.is_last():
                    action_step = self.agent.policy.action(time_step)
                    time_step = self.env_e.step(action_step.action)
                    video.append_data(self.env_e.render())
