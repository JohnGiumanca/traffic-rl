import os
import sys
import optparse
import random


if 'SUMO_HOME' in os.environ:
    print('HERE')
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  
import traci  

def run():
    step=0
    road_id = 'wptocp'
    while traci.simulation.getMinExpectedNumber() > 0:
        if step % 10 == 0:
            print('Nr of vehicles on wptocp road: ')
            print(get_nr_vehicles_stopped_road(road_id))
            print('  ')
        traci.simulationStep()
        step += 1
    traci.close()
    sys.stdout.flush()

def get_nr_vehicles_stopped_lane(lane_id):
    n = 0
    for k in traci.lane.getLastStepVehicleIDs(lane_id):
        if traci.vehicle.getSpeed(k) == 0:
             n += 1
    return n

def get_nr_vehicles_stopped_road(edge_id):
    n = 0
    for lane_nr in range(3):
        n += get_nr_vehicles_stopped_lane(edge_id + '_' + str(lane_nr))
    return n


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options


# this is the main entry point of this script
if __name__ == "__main__":

    options = get_options()
    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    traci.start([sumoBinary, "-c", "classic3lane.sumocfg",
                             "--tripinfo-output", "tripinfo.xml"])
    run()