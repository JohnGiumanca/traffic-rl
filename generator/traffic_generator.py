
import numpy as np

class TrafficGenerator:

	def __init__(self, routefile_path, max_steps, n_cars, route_weights_dict):

		self._routefile_path = routefile_path
		self._max_steps=max_steps
		self._n_cars = n_cars
		self._weights_dict = route_weights_dict



	def generate_routefile(self):

		car_spawn = np.linspace(0, self._max_steps, self._n_cars).astype(int)
		routes = list(self._weights_dict.keys())
		weights = np.array(list(self._weights_dict.values()))
		probs = weights / weights.sum()


		with open(self._routefile_path, 'w') as routefile:
			print(''' <routes>
            <vType id="car" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="20.0" />
            <route id="n_w" edges="nptocp cptowp"/>
            <route id="n_s" edges="nptocp cptosp"/>
            <route id="n_e" edges="nptocp cptoep"/>
            <route id="e_n" edges="eptocp cptonp"/>
            <route id="e_w" edges="eptocp cptowp"/>
            <route id="e_s" edges="eptocp cptosp"/>
            <route id="s_e" edges="sptocp cptoep"/>
            <route id="s_n" edges="sptocp cptonp"/>
            <route id="s_w" edges="sptocp cptowp"/>
            <route id="w_s" edges="wptocp cptosp"/>
            <route id="w_e" edges="wptocp cptoep"/>
            <route id="w_n" edges="wptocp cptonp"/>''', file=routefile)


			for idx, spawn_step in enumerate(car_spawn):
				route = np.random.choice(routes, p=probs)
				veh_id = route+'_'+str(idx)
				print('    <vehicle id="%s" type="car" route="%s" depart="%s" departLane="random" departSpeed="10" />' 
					%(veh_id, route,  spawn_step), file=routefile)





			print("</routes>", file=routefile)