import os
import sys

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumolib import checkBinary  
import traci
from gym.spaces import Discrete, Box
from gym import Env
import numpy as np


class SumoEnv(Env):

    def __init__(self, sumocfg_path=None, net_path=None, rou_path=None, max_steps=1000, 
                    sumo_gui=False, yellow_time_limit=3, occupancy_threshold=0.5, 
                    min_green=1, multi_junction_net=False, car_size=5, car_gap=2.5):
        super(SumoEnv, self).__init__()
        
        self._sumocfg_path = sumocfg_path
        self._net_path = net_path
        self._rou_path = rou_path
        self._max_steps = max_steps
        self._sumo_gui = sumo_gui
        self._multi_junction = multi_junction_net
        self._yellow_time_limit = yellow_time_limit
        self._occupancy_threshold = occupancy_threshold
        self._min_green = min_green
        self._car_size = car_size
        self._car_gap = car_gap

        self.traci_init = None
        self.trafficlight_ids = []
        self.trafficlight_phases = dict()
        self.trafficlight_n_green_phases = dict()
        self.trafficlight_lanes = dict()
        
        self._observation_space = dict()
        self._action_space = dict()

        self.last_observation = dict()
        self.last_action = dict()
        self.last_reward = dict()
        
        self.last_phase = 0
        self.next_phase = None
        self.last_max_occupancy = 0

        self.last_step = 0
        self.running = False

        self.reset(init_use=True)
        self.close()

    def init_atrib(self):
        self.trafficlight_ids = traci.trafficlight.getIDList()

        for tl_id in self.trafficlight_ids:
            self.trafficlight_phases[tl_id] = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)[0].phases
            self.trafficlight_n_green_phases[tl_id] = round(len(self.trafficlight_phases[tl_id]) / 2)
            self.trafficlight_lanes[tl_id] = list(dict.fromkeys(traci.trafficlight.getControlledLanes('cp')))
            
            observation_n = len(self.trafficlight_phases[tl_id]) + 2* len(self.trafficlight_lanes[tl_id]) # one for tl phase digit  
            box_low = np.zeros(observation_n)
            box_high = np.full(observation_n, np.inf)
            box_high[0] = self.trafficlight_n_green_phases[tl_id] * 2 # initialy we also include yellow lights in the obs
            self._observation_space[tl_id] = Box(box_low, box_high) # 0-7 for a classic TL and 0-inf for nr os stopped cars

            self._action_space[tl_id] = Discrete(self.trafficlight_n_green_phases[tl_id])


    @property
    def observation_space(self):
        return self._observation_space[self.trafficlight_ids[0]]
    @property
    def action_space(self):
        return self._action_space[self.trafficlight_ids[0]]
    
    

    def reset(self, init_use=False):
        
        self.last_step = 0
        self.next_phase = None
        self.last_max_occupancy = 0

        if self.running:
            traci.close()
            self.running = False

        sumo_binary = None
        if self._sumo_gui and not init_use:
            sumo_binary = checkBinary('sumo-gui')
        else:
            sumo_binary = checkBinary('sumo')

        if self._sumocfg_path:
            sumo_config = [sumo_binary, "-c", self._sumocfg_path]
        else:
            sumo_config = [sumo_binary, '-n', self._net_path, '-r', self._rou_path,]
        
        traci.start(sumo_config)
        self.running = True

        if self.traci_init is None:
            self.init_atrib()
            self.traci_init = True

        obs = self.update_observation()

        if not self._multi_junction:
            tl_id = self.trafficlight_ids[0]
            return obs[tl_id]
    

    # def step(self):
    #     ...
    #     return observation, reward, done, inf

    def render(self):
        pass

    def close(self):
        traci.close()
        sys.stdout.flush()
        self.running = False



    def step(self, action):
        
        if not self._multi_junction:
            tl_id = self.trafficlight_ids[0]

            if action is not None and not self.is_yellow(self.last_phase):
                self.do_action(action)
        
            obs, reward, done, info = self.sumo_step()
            self.last_phase = obs[tl_id][0]

            return obs[tl_id], reward[tl_id], done, info

    def sumo_step(self, n_steps=1):

        info = {}
        for step in range(n_steps):
            traci.simulationStep()
            self.last_step += 1
            self.update_max_occupancy()

            obs = self.update_observation()
            reward = self.update_reward()
            done = self.check_done()
            
            if done:
                return obs, reward, done, info
        
        return obs, reward, done, info

    def get_lanes(self, tl_id):
        return self.trafficlight_lanes[tl_id]
    
    def get_trafficlight_ids(self):
        return self.trafficlight_ids


    # def get_trafficlight_phase(self, id):
    #     if self.running:
    #         return traci.trafficlight.getPhase(id)
    #     else:
    #         print("Can't retrieve Traffic Light phase when simulation is not running!")
    #         return None

    def get_halting_vehicles(self, tl_id):
        stopped_cars_each_lane = []
        for lane in self.trafficlight_lanes[tl_id]:
            stopped_cars = traci.lane.getLastStepHaltingNumber(lane)
            stopped_cars_each_lane.append(stopped_cars)

        return stopped_cars_each_lane

    def get_halting_vehicles_norm(self, tl_id):
        stopped_cars_each_lane = []
        for lane in self.trafficlight_lanes[tl_id]:
            stopped_cars = traci.lane.getLastStepHaltingNumber(lane)
            car_space = self._car_size + self._car_gap
            lane_length = traci.lane.getLength(lane)
            stopped_cars_space = stopped_cars * car_space
            stopped_cars_each_lane.append(stopped_cars_space / lane_length)

        return stopped_cars_each_lane

    def get_total_vehicles_norm(self, tl_id):
        stopped_cars_each_lane = []
        for lane in self.trafficlight_lanes[tl_id]:
            stopped_cars = traci.lane.getLastStepVehicleNumber(lane)
            car_space = self._car_size + self._car_gap
            lane_length = traci.lane.getLength(lane)
            stopped_cars_space = stopped_cars * car_space
            stopped_cars_each_lane.append(stopped_cars_space / lane_length)

        return stopped_cars_each_lane

    def do_action(self, action): # for single agent use!
        tl_id =self.trafficlight_ids[0]
        self.next_phase = action * 2

        if self.next_phase !=  self.last_phase:
            traci.trafficlight.setPhase(tl_id,  self.last_phase + 1) 
            self.sumo_step(n_steps=self._yellow_time_limit)
            traci.trafficlight.setPhase(tl_id,  self.next_phase) 
            # self.sumo_step(n_steps=self._min_green)

    # def change_tl_phase(action, tl_id):
    #     next_phase = action * 2
    #     last_phase = self.last_observation[tl_id][0]
    #     if 


    def is_yellow(self, tl_phase):
        return tl_phase % 2 != 0

    def check_done(self):
        if self.last_step >= self._max_steps:
            return True

        if self.get_max_occupancy() > self._occupancy_threshold:
            return True

        return False
    

    def update_reward(self):
        for tl_id in self.trafficlight_ids:  
            reward = self.calculate_reward_3(tl_id)
            self.last_reward[tl_id] = reward

        return self.last_reward

    def update_observation(self):
        for tl_id in self.trafficlight_ids: 
            obs = self.compute_observation_3(tl_id)
            self.last_observation[tl_id] = obs

        return self.last_observation

    def update_max_occupancy(self):
        self.last_max_occupancy = self.get_max_occupancy()


    def calculate_reward(self, tl_id):
        #choose reward calculation
        return self.calculate_reward_3(tl_id)


    def calculate_reward_1(self, tl_id):
        # -1 for each stopped car on each lane 
        stopped_cars_each_lane = self.last_observation[tl_id][1:]
        reward = stopped_cars_each_lane.sum() * -1

        if self.get_max_occupancy() > self._occupancy_threshold:
            return -1000

        return reward

    def calculate_reward_2(self, tl_id):
        # -1 for each stopped car on each lane 
        tl_id = self.trafficlight_ids[0]
        stopped_cars_idx = len(self.trafficlight_phases[tl_id])  
        stopped_cars_each_lane = self.last_observation[tl_id][stopped_cars_idx:]
        reward = stopped_cars_each_lane.sum() * -1

        if self.get_max_occupancy() > self._occupancy_threshold:
            return -1000

        return reward

    def calculate_reward_3(self, tl_id): 

        total_waiting_time = self.get_total_waiting_time()
        reward = -total_waiting_time / 12 

        return reward


    def compute_observation_1(self):
        for tl_id in self.trafficlight_ids:  
            tl_phase = traci.trafficlight.getPhase(tl_id)
            stopped_cars = np.array(self.get_halting_vehicles(tl_id))
            obs = np.append(tl_phase, stopped_cars)
            self.last_observation[tl_id] = obs.astype(np.float32)
        
        return self.last_observation

    def compute_observation_2(self):
        for tl_id in self.trafficlight_ids: 
            #traffic light phase one hhot
            n_phase = len(self.trafficlight_phases[tl_id])
            last_phase = traci.trafficlight.getPhase(tl_id)
            tl_phase_one_hot = np.zeros(n_phase)
            tl_phase_one_hot[last_phase] = 1
            #lane n cars stops normalized 0-1
            stopped_cars_norm= np.array(self.get_halting_vehicles_norm(tl_id))

            obs = np.append(tl_phase_one_hot, stopped_cars_norm)
            self.last_observation[tl_id] = obs.astype(np.float32)

        return self.last_observation

    def compute_observation_3(self, tl_id):
        n_phase = len(self.trafficlight_phases[tl_id])
        last_phase = traci.trafficlight.getPhase(tl_id)
        tl_phase_one_hot = np.zeros(n_phase)
        tl_phase_one_hot[last_phase] = 1
        #lane n cars stops normalized 0-1
        stopped_cars_norm = np.array(self.get_halting_vehicles_norm(tl_id))
        #lane n cars total normalized 0-1
        total_cars_norm = np.array(self.get_total_vehicles_norm(tl_id))

        obs = np.concatenate((tl_phase_one_hot, stopped_cars_norm, total_cars_norm))
        
        return obs.astype(np.float32)



    def get_max_occupancy(self):
        tl_id =self.trafficlight_ids[0]
        occ = [traci.lane.getLastStepOccupancy(lane) for lane in self.trafficlight_lanes[tl_id]]
        return max(occ)
        

    def get_lane_info(self, lane_id):
        print('---')
        print(f'Occupancy: {traci.lane.getLastStepOccupancy(lane_id)}')
        print(f'Lane length: {traci.lane.getLength(lane_id)}')
        print(f'Vehicle mean length: {traci.lane.getLastStepLength(lane_id)}')
        print(f'Mean speed: {traci.lane.getLastStepMeanSpeed(lane_id)}')
        print(f'Waiting time: {traci.lane.getWaitingTime(lane_id)}')
        print(f'Nr of stopped vehicles: {traci.lane.getLastStepHaltingNumber(lane_id)}')
        print(f'Cars waiting time: {self.get_waiting_time_lane(lane_id)}')
    
    def get_road_info(self, edge_id):
        print('------')
        for lane_nr in range(3):
            print(f'Lane {lane_nr}')
            self.get_lane_info(edge_id + '_' + str(lane_nr))

    def get_tl_info(self):
        tl_id =self.trafficlight_ids[0]
        print(traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id))

    def get_total_waiting_time(self):
        tl_id =self.trafficlight_ids[0]
        total_time = 0 
        for lane_id in self.trafficlight_lanes[tl_id]:
            total_time_lane = sum(self.get_waiting_times_lane(lane_id))
            total_time += total_time_lane
        return total_time

    def get_waiting_times_lane(self, lane_id):
        cars = traci.lane.getLastStepVehicleIDs(lane_id)
        waiting_times = []
        for car in cars:
            # waiting_time_acc = traci.vehicle.getAccumulatedWaitingTime(car)
            waiting_time = traci.vehicle.getWaitingTime(car)
            waiting_times.append((waiting_time))
        return waiting_times































   
