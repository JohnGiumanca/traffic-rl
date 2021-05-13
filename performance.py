from environment.sumo_env import SumoEnv
from generator.traffic_generator import TrafficGenerator
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

BASE_PATH = '/Users/johnsmacbook/MyDocs/Facultate/Master/Disertatie/myproj'
NET_PATH = BASE_PATH+ '/junction/clasic3lane.net.xml'
CONFIG_PATH = BASE_PATH + '/junction/classic3lane.sumocfg'

MAX_STEPS = 1000
N_CARS = 500
MAX_OCCUPANCY_TH = 0.5
MIN_GREEN = 3
ROUTE_WEIGHTS = {
    'n_w': 1,
    'n_s': 1,
    'n_e': 1,
    'e_n': 1,
    'e_w': 1,
    'e_s': 1,
    's_e': 1,
    's_n': 1,
    's_w': 1,
    'w_s': 1,
    'w_e': 1,
    'w_n': 1,
}

class Tester:

    def __init__(self, env, model):
        self._env = env
        self._model = model

        self.reward_history_default = None
        self.reward_history_dqn = None
        self.avg_reward_default = None
        self.avg_reward_dqn = None

        self.run_simulations()

    def run_simulations(self):
        done = False
        step = 0
        reward_history = []
        obs = self._env.reset()
        while not done:
            
            action = None
            obs, reward, done, _ = env.step(action)
            step += 1
            reward_history.append(reward)
        
        env.close()

        reward_default = sum(reward_history) / len(reward_history)
        self.reward_history_default = reward_history
        self.avg_reward_default = reward_default

        done = False
        step = 0
        reward_history = []
        obs = self._env.reset()
        while not done:
            
            action = action = dqn.forward(obs)
            obs, reward, done, _ = env.step(action)
            step += 1
            reward_history.append(reward)

        env.close()

        reward_dqn = sum(reward_history) / len(reward_history)
        self.reward_history_dqn = reward_history
        self.avg_reward_dqn = reward_dqn

    
    def plot_reward_history(self,save_path=None):

        fig, (ax1, ax2) = plt.subplots(1,2,sharey=True, figsize=(14,6))

        fig.suptitle('Reward History of SUMO episode')

        ax1.plot(range(len(self.reward_history_default)), self.reward_history_default)
        ax1.grid(True)
        ax2.plot(range(len(self.reward_history_dqn)), self.reward_history_dqn)
        ax2.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_avg_reward(self, save_path=None):
        done = False
        step = 0
        reward_history = []
        obs = self._env.reset()
        while not done:
            
            action = None
            obs, reward, done, _ = env.step(action)
            step += 1
            reward_history.append(reward)

        reward_default = sum(reward_history) / len(reward_history)

        done = False
        step = 0
        reward_history = []
        obs = self._env.reset()
        while not done:
            
            action = action = dqn.forward(obs)
            obs, reward, done, _ = env.step(action)
            step += 1
            reward_history.append(reward)

        reward_dqn = sum(reward_history) / len(reward_history)

        plt.bar(['Default', 'DQN'], [reward_default, reward_dqn])
        if save_path:
            plt.savefig(save_path)
        plt.show()

if __name__ == '__main__':

    new_routefile = BASE_PATH + '/junction/test.rou.xml'
    # generator = TrafficGenerator(new_routefile, MAX_STEPS, N_CARS, ROUTE_WEIGHTS) 
    # generator.generate_routefile()

    env = SumoEnv(net_path=NET_PATH, rou_path=new_routefile, max_steps=MAX_STEPS, sumo_gui=False, 
                    occupancy_threshold=MAX_OCCUPANCY_TH, min_green=MIN_GREEN)

    nb_actions = env.action_space.n
    print(nb_actions)
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions, activation='linear'))
    print(model.summary())

    memory = SequentialMemory(limit=100_000, window_length=1)
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.01,
                              nb_steps=80_000)
    # policy = BoltzmannQPolicy()
    # enable the dueling network
    # you can specify the dueling_type to one of {'avg','max','naive'}
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=2000,
                   enable_dueling_network=True, dueling_type='avg', target_model_update=3000, policy=policy)
    dqn.compile(Adam(lr=0.001), metrics=['mae'])
    dqn.load_weights('dqn_traffic_weights.h5f')

    tester = Tester(env=env, model=dqn)
    tester.plot_reward_history(save_path='reward_history.png')























