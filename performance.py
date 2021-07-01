from environment.sumo_env import SumoEnv
from generator.traffic_generator import TrafficGenerator
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

BASE_PATH = '/Users/johnsmacbook/MyDocs/Facultate/Master/Disertatie/myproj'
# NET_PATH = BASE_PATH+ '/junction/clasic3lane.net.xml'
NET_PATH = BASE_PATH+ '/junction/clasic3lane_leftonly.net.xml'
GEN_ROUTE_PATH = BASE_PATH + '/junction/generated.rou.xml'
CONFIG_PATH = BASE_PATH + '/junction/classic3lane.sumocfg'
WEIGHTS_BASE = 'results/obs3_rwd3_mingreen1/150ksteps_1000s_400t/'
WEIGHTS_PATH = WEIGHTS_BASE + 'dqn_traffic_weights.h5f'
# WEIGHTS_PATH = 'dqn_traffic_weights.h5f'

MAX_STEPS = 500
N_CARS = 100
MAX_OCCUPANCY_TH = 0.5
MIN_GREEN = 3
WINDOW_LENGTH = 4
ROUTE_WEIGHTS = {
    'n_w': 1,
    'n_s': 1,
    'n_e': 0.5,
    'e_n': 1,
    'e_w': 1,
    'e_s': 0.5,
    's_e': 1,
    's_n': 1,
    's_w': 0.5,
    'w_s': 1,
    'w_e': 1,
    'w_n': 0.5,
}
SIM_N = 5


class Tester:

    def __init__(self, sim_n, model):
        self._sim_n = sim_n
        self._model = model

        self.reward_history_default = None
        self.reward_history_dqn = None
        self.avg_reward_default = None
        self.avg_reward_dqn = None

        self.multi_sims_reward_default = None
        self.multi_sims_reward_model = None
        self.run_simulations()

    def run_simulations(self):

        multi_sims_default = []
        multi_sims_model = []
        for _ in range(self._sim_n):
            self.generate_traffic()
            env = self.init_env()

            done = False
            step = 0
            step_reward_history_default = []    
            obs = env.reset()
            while not done:
                
                action = None
                obs, reward, done, _ = env.step(action)
                step += 1
            
            env.close()
            multi_sims_default.append(env.history_reward)
            # reward_default = sum(reward_history) / len(reward_history)
            # self.reward_history_default = reward_history
            # self.avg_reward_default = reward_default

            done = False
            step = 0
            step_reward_history_model = []
            obs = env.reset()
            while not done:
                
                action = dqn.forward(obs)
                obs, reward, done, _ = env.step(action)
                step += 1

            env.close()
            multi_sims_model.append(env.history_reward)
            # reward_dqn = sum(reward_history) / len(reward_history)
            # self.reward_history_dqn = reward_history
            # self.avg_reward_dqn = reward_dqn

        self.multi_sims_reward_default = multi_sims_default
        self.multi_sims_reward_model = multi_sims_model

    def generate_traffic(self):
        generator = TrafficGenerator(GEN_ROUTE_PATH, MAX_STEPS, N_CARS, ROUTE_WEIGHTS) 
        generator.generate_routefile()

    def init_env(self):
        env = SumoEnv(net_path=NET_PATH, rou_path=GEN_ROUTE_PATH, max_steps=MAX_STEPS, sumo_gui=False, 
                    occupancy_threshold=MAX_OCCUPANCY_TH, min_green=MIN_GREEN)
        return env
    
    def plot_reward_history(self, save_path=None):
        fig, (ax1, ax2) = plt.subplots(1,2,sharey=True, figsize=(14,6))

        fig.suptitle('Reward History of SUMO episode')

        ax1.plot(range(len(self.reward_history_default)), self.reward_history_default)
        ax1.grid(True)
        ax2.plot(range(len(self.reward_history_dqn)), self.reward_history_dqn)
        ax2.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_reward_history_mean(self, save_path=None):
        fig, (ax1, ax2) = plt.subplots(1,2,sharey=True, figsize=(14,6))

        fig.suptitle('Reward History of SUMO episode')
        min_run_default = min(map(len, self.multi_sims_reward_default))
        min_len_default = [run[:min_run_default] for run in self.multi_sims_reward_default]
        mean_array_default = np.array(min_len_default).mean(axis=0)
        ax1.plot(range(mean_array_default.shape[0]), mean_array_default)
        ax1.grid(True)
        ax1.set_ylim(-1000,0)
        ax1.set_title('Default')
        min_run_model = min(map(len, self.multi_sims_reward_model))
        min_len_model = [run[:min_run_model] for run in self.multi_sims_reward_model]
        mean_array_model = np.array(min_len_model).mean(axis=0)
        ax2.plot(range(mean_array_model.shape[0]), mean_array_model)
        ax2.grid(True)
        ax2.set_title('DQN')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_avg_reward(self, save_path=None):
        
        min_run_default = min(map(len, self.multi_sims_reward_default))
        min_len_default = [run[:min_run_default] for run in self.multi_sims_reward_default]
        mean_array_default = np.array(min_len_default).mean(axis=0)
        mean_default = mean_array_default.mean()

        min_run_model = min(map(len, self.multi_sims_reward_model))
        min_len_model = [run[:min_run_model] for run in self.multi_sims_reward_model]
        mean_array_model = np.array(min_len_model).mean(axis=0)
        mean_model = mean_array_model.mean()

        plt.bar(['Default', 'DQN'], [mean_default, mean_model])
        if save_path:
            plt.savefig(save_path)
        plt.show()

if __name__ == '__main__':

    envv = SumoEnv(net_path=NET_PATH, rou_path=GEN_ROUTE_PATH, max_steps=MAX_STEPS, sumo_gui=False, 
                    occupancy_threshold=MAX_OCCUPANCY_TH, min_green=MIN_GREEN)


    nb_actions = envv.action_space.n
    print(nb_actions)
    model = Sequential()
    model.add(Flatten(input_shape=(WINDOW_LENGTH,) + envv.observation_space.shape))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions, activation='linear'))
    print(model.summary())

    memory = SequentialMemory(limit=1_000_000, window_length=WINDOW_LENGTH)
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=1_000_000)
    # policy = BoltzmannQPolicy()
    # enable the dueling network
    # you can specify the dueling_type to one of {'avg','max','naive'}
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=50_000,
                   enable_dueling_network=True, dueling_type='avg', target_model_update=10_000, 
                   policy=policy, train_interval=4)
    dqn.compile(Adam(lr=0.0001), metrics=['mae'])
    dqn.load_weights(WEIGHTS_PATH)
    print('WEIGHTS: '+ WEIGHTS_PATH)


    tester = Tester(sim_n=SIM_N, model=dqn)
    avg_path = str(SIM_N)+'sims_avg_'+str(MAX_STEPS)+'s_'+str(N_CARS)+'t.png'
    tester.plot_avg_reward(save_path=avg_path)
    his_path = str(SIM_N)+'sims_his_'+str(MAX_STEPS)+'s_'+str(N_CARS)+'t.png'
    tester.plot_reward_history_mean(save_path=his_path)






















