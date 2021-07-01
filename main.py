import os
import sys
import argparse
import pandas
import pickle

import matplotlib.pyplot as plt

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumolib import checkBinary  
import traci

from environment.sumo_env import SumoEnv
from generator.traffic_generator import TrafficGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

# from tensorflow.python.compiler.tensorrt import trt_convert as trt

BASE_PATH = '/Users/johnsmacbook/MyDocs/Facultate/Master/Disertatie/myproj'
NET_PATH = BASE_PATH+ '/junction/clasic3lane_leftonly.net.xml'
# NET_PATH = BASE_PATH+ '/junction/clasic3lane.net.xml'
ROU_PATH = BASE_PATH  + '/junction/classic3lane.rou.xml'
CONFIG_PATH = BASE_PATH + '/junction/classic3lane.sumocfg'
WEIGHTS_BASE = 'results/obs3_rwd3_mingreen1/150ksteps_1000s_400t/'
WEIGHTS_PATH = WEIGHTS_BASE + 'dqn_traffic_weights.h5f'

MAX_STEPS = 1000
N_CARS = 600
MAX_OCCUPANCY_TH = 0.5
MIN_GREEN = 3
WINDOW_LENGTH = 1
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



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true",
                         default=False, help="Run sumo with graphical interface")
    parser.add_argument("--train", action="store_true",
                         default=False, help="Train the DQN Agent")
    parser.add_argument("--test", action="store_true",
                         default=False, help="Test the DQN Agent")
    parser.add_argument("--random", action="store_true",
                         default=False, help="Run SUMO with random actions")
    parser.add_argument("--default", action="store_true",
                         default=False, help="Run SUMO with default traffic light programme")
    args = parser.parse_args()
    return args

# def run():
#     step=0
#     road_id = 'wptocp'
#     while traci.simulation.getMinExpectedNumber() > 0:
#         if step % 10 == 0:
#             print('Nr of vehicles on wptocp road: ')
#             print(get_nr_vehicles_stopped_road(road_id))
#             print('  ')
#         traci.simulationStep()
#         step += 1


    
if __name__ == '__main__':

    args = get_args()

    gui = False
    if args.gui:
        gui = True

    new_routefile = BASE_PATH + '/junction/test.rou.xml'
    generator = TrafficGenerator(new_routefile, MAX_STEPS, N_CARS, ROUTE_WEIGHTS) 
    generator.generate_routefile()

    env = SumoEnv(net_path=NET_PATH, rou_path=new_routefile, max_steps=MAX_STEPS, 
                    sumo_gui=gui, occupancy_threshold=MAX_OCCUPANCY_TH, min_green=MIN_GREEN)

    

    if args.random or args.default:
        
        done = False
        step = 0
        env.reset()
        new_record_n = 0
        last_record = 0
        while not done:
            action = None
            if args.random:
                action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
            # print(env.get_trafficlight_phase('cp'))
            # env.get_road_info(road_id)
            print(f'Observation: {obs}')
            print(f'Reward: {reward}')
            print(f'Done: {done}')
            print(f'Max Occupancy: {env.last_max_occupancy}')
            print(f'Last Step: {env.last_step}')
            print(f'Global Waiting Time: {env.get_total_waiting_time()}')
            if env.last_max_occupancy > last_record:
                last_record = env.last_max_occupancy
                new_record_n += 1
            step += 1
        print('--')
        print(new_record_n)
        print('--')
        env.close()


    # check_env(env, warn=True)
    # model = DQN('MlpPolicy', env, verbose=2).learn(50000).save("dqn_traffic")

    # model = DQN.load("dqn_traffic", env=env)
    # done = False
    # obs = env.reset()
    # while not done:
    #     action, _ = model.predict(obs, deterministic=True)
    #     print(action)
    #     obs, rewards, done, _ = env.step(action)
    #     env.render()

    # env.close()


    nb_actions = env.action_space.n
    print(nb_actions)
    model = Sequential()
    model.add(Flatten(input_shape=(WINDOW_LENGTH,) + env.observation_space.shape))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions, activation='linear'))
    print(model.summary())

    memory = SequentialMemory(limit=50_000, window_length=WINDOW_LENGTH)
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=50_000)
    # policy = BoltzmannQPolicy()
    # enable the dueling network
    # you can specify the dueling_type to one of {'avg','max','naive'}
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=2_000,
                   enable_dueling_network=True, dueling_type='avg', target_model_update=1_000, 
                   policy=policy, train_interval=4)
    dqn.compile(Adam(lr=0.001), metrics=['mae'])

    if args.train:  
        # dqn.load_weights('dqn_traffic_weights.h5f')
        history = dqn.fit(env, nb_steps=100_000, visualize=False, verbose=2)

        env.close()
        dqn.save_weights('dqn_traffic_weights.h5f', overwrite=True)
        pandas.DataFrame(history.history['episode_reward']).plot(figsize=(16, 5) )
        plt.savefig('output.png')
    
    
    if args.test:
        dqn.load_weights(WEIGHTS_PATH)
        done = False
        step = 0
        obs = env.reset()
        while not done:
            action = dqn.forward(obs)
            obs, reward, done, _ = env.step(action)
            # print(env.get_trafficlight_phase('cp'))
            # env.get_road_info(road_id)
            print(f'Action: {action}')
            print(f'Observation: {obs}')
            print(f'Reward: {reward}')
            print(f'Done: {done}')
            print(f'Max Occupancy: {env.get_max_occupancy()}')
            step += 1

        env.close()

    









