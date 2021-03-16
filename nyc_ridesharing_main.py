'''
1 read TAZ zones get the edges
2 read demand files get distributions
3 sample enough demands as backgrounds
4 sample from and to edges according to TAZ zones (need to check network)
5 write trip file
6 compute route file with DUA
7 generate new demand trip files
'''

import pandas as pd
import numpy as np
from collections import defaultdict
from queue import PriorityQueue
import pickle
import sys, os
TOOLS = "/home/boqi/Downloads/sumo-1.8.0/tools"
SUMO_GUI = "/home/boqi/Downloads/sumo-1.8.0/bin/sumo-gui"
SUMO = "/home/boqi/Downloads/sumo-1.8.0/bin/sumo"
sys.path.append(TOOLS)
import sumolib
from toy_ridesharing_main import Vehicle, State, Request, travel
from simple_rl.mdp.MDPClass import MDP

TRAIN = False
if TRAIN:
    import libsumo as traci
    SUMOEXE = SUMO
else:
    import traci
    SUMOEXE = SUMO_GUI

TIME_STEP = 60
DEMAND_GEN = 15*60
MAX_DELAY_SEC = 5*60
MAX_WAIT_SEC = 5*60
MAX_CAPACITY = 4
MAX_V_PER_REQ = 10
PENALTY = 10*60

ZONEIDS = [12, 88, 13, 261, 87, 209, 231, 45, 125, 211, 144, 148, 232, 158, 249, 114, 113, 79, 4,
           246, 68, 90, 234, 107, 224, 186, 164, 170, 137, 100, 233, 50, 48, 230, 161, 162, 229, 143, 142, 163,
           237, 141, 140, 239, 236, 263, 262, 238, 151, 24, 75, 43, 166, 41, 74, 152, 42, 116, 244, 120, 243,
           127, 128]

def readTAZfile(filename):
    tazs = {}
    file = open(filename, 'r')
    lines = file.readlines()
    for line in lines:
        tokens = line.split("\"")
        if "<taz id=" in tokens[0]:
            taz_id = int(tokens[1])
            if taz_id not in tazs:
                tazs[taz_id] = []
        elif "tazSource" in tokens[0]:
            if "id" in tokens[2]:
                tazs[taz_id].append(tokens[3])
            else:
                tazs[taz_id].append(tokens[1])
    file.close()
    return tazs

def readDemandfiles(filenames):
    df = pd.concat((pd.read_csv(f, usecols=['pickup_datetime', 'PULocationID', 'DOLocationID', 'group']) for f in filenames), ignore_index=True)
    df['pickup_date'] = pd.to_datetime(df.pickup_datetime)
    df['time_int'] = (df.pickup_date.dt.minute + 60*df.pickup_date.dt.hour) // 15
    df['day_of_week'] = df.pickup_date.dt.day_of_week
    demand_by_day = []
    for day in range(1, 32):
        demand_by_day.append(len(df[(df.pickup_date.dt.day==day) &
                                    (df.PULocationID.isin(ZONEIDS))&
                                    (df.DOLocationID.isin(ZONEIDS))])
        )
    demand_by_interval_from_to = dict()
    for interval in range(96):
        for start in ZONEIDS:
            for dest in ZONEIDS:
                print(interval, start, dest)
                num = len(df[(df.PULocationID==start) &
                             (df.DOLocationID==dest) &
                             (df.time_int==interval) &
                             (df.day_of_week!=5) &
                             (df.day_of_week!=6)])
                if num > 0:
                    demand_by_interval_from_to[(interval, start, dest)] = num

    return df, demand_by_day, demand_by_interval_from_to

def sampleTrips(demand_by_day, demand_by_interval_from_to, tazs, net):
    mean_demands = np.max(demand_by_day)
    rng = np.random.RandomState(0)
    iod_indices = dict()
    ind = 0
    weights = []
    for i, s, d in demand_by_interval_from_to:
        iod_indices[ind] = (i, s, d)
        weights.append(demand_by_interval_from_to[(i,s,d)])
        ind += 1
    weights = np.array(weights, dtype=np.float)
    weights = weights/np.sum(weights)
    sampled_demand_indices = rng.choice(len(iod_indices), size=int(mean_demands), p=weights)
    trips = PriorityQueue()
    num = 0
    print(mean_demands)
    for ind in sampled_demand_indices:
        i, s, d = iod_indices[ind]
        time = rng.randint(i*15*60, (i+1)*15*60)
        print(num)
        while True:
            start_edge = tazs[s][rng.choice(len(tazs[s]))]
            dest_edge = tazs[d][rng.choice(len(tazs[d]))]
            path = net.getShortestPath(net.getEdge(start_edge), net.getEdge(dest_edge))
            if path[0]:
                trips.put((time, [start_edge, dest_edge]))
                break
        num += 1
    return trips, sampled_demand_indices

def writeTripfile(filename, sampled_trips):
    f = open(filename, "a")
    f.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
    f.write("<routes xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:noNamespaceSchemaLocation=\"http://sumo.dlr.de/xsd/routes_file.xsd\">\n")
    id = 0
    while not sampled_trips.empty():
        next_trip = sampled_trips.get()
        f.write("      <trip id=\"{}\" depart=\"{}\" from=\"{}\" to=\"{}\"/>\n".format(id, next_trip[0], next_trip[1][0], next_trip[1][1]))
        id += 1
    f.write("</routes>\n")
    f.close()

class SUMOVehicle(Vehicle):
    def __init__(self, unique, loc):
        super(SUMOVehicle, self).__init__("sav_{}".format(unique), loc)
        traci.route.add("sav_init_{}".format(unique), [loc])
        traci.vehicle.add("sav_{}".format(unique), "sav_init_{}".format(unique), typeID="taxi")
        #traci.vehicle.setStop("sav_{}".format(unique), loc, flags=3)
        self.new_req = None
        self.last_pas = None
        self.dispatch_order = []

    def head_for(self, loc, now_time, traffic):
        self.new_req = None
        self.last_pas = None
        self.dispatch_order = []
        traci.vehicle.changeTarget(self.unique, loc)
        self.scheduleroute = traci.getRoute(self.unique)
        if traci.vehicle.isStopped(self.unique):
            traci.vehicle.resume(self.unique)

    def update(self, now_time, newRequests, traffic):
        loc = traci.vehicle.getRoudID(self.unique)
        if loc in traffic.edges:
            self.location = loc
        else:
            # internal edge at intersection
            for idx, edge in enumerate(self.scheduleroute):
                if edge == self.location:
                    self.location = self.scheduleroute[idx+1]
                    break
            else:
                raise
        if self.location != self.scheduleroute[0]:
            assert(self.location == self.scheduleroute[1])
            self.scheduleroute.pop(0)

        # dropoff
        dropped_passengers = []
        if traci.vehicle.isStopped(self.unique):
            dropped = []
            for pas in self.passengers:
                if pas.onBoard:
                    if pas.dest == self.location:
                        dropped.append(pas.unique)

            if len(dropped) > 0:
                while self.dropoff_order[0] in dropped:
                     dropped_passengers.append(self.dropoff_order.pop(0))
                     self.served_reqs += 1

            if len(self.scheduleroute) > 1:
                traci.vehicle.resume(self.unique) # collision might happen but for now it is ignored

        # pickup
        if self.new_req:
            if self.location == self.new_req.start:
                traci.vehicle.dispatchTaxi(self.unique, [self.new_req.unique]+list(self.dropoff_order))
                traci.vehicle.setRoute(self.unique, self.scheduleroute)
                picked_passengers.append(self.new_req.unique)
                self.new_req.onBoard = True

    def finish_update(self):
        newPassengers = []
        schPassengers = []
        newRequests = []
        for pas in self.passengers:
            if pas.onBoard:
                if pas.unique not in dropped_passengers:
                    newPassengers.append(pas)
            else:
                newRequests.append(pas)
                if len(schPassengers) == 0:
                    schPassengers.append(pas)
                elif pas.scheduledOnTime < schPassengers[0].scheduledOnTime:
                    schPassengers[0] = pas

        self.passengers = newPassengers
        self.schedulepassengers = schPassengers
        return newRequests

    def set_path(self, now_time, path, traffic):
        # this is after the setting of passengers/ requests
        # The order of passengers are now in dropoff order
        # Find the new request
        #self.last_pas = None
        self.new_req = None
        self.dropoff_order = deque()
        for idx, pas in enumerate(self.passengers):
            if pas.onBoard == False:
                 self.new_req = pas
                 #if idx != 0:
                 #    self.last_pas = self.passengers[idx-1]
            #else:
                #self.dispatch_order_old.append(pas.unique)
            self.dropoff_order.append(pas.unique)
        # new_req  = last pas = None dispath != []

        # new_req  != None last pas = None dispath = []
        self.scheduleroute = path
        traci.vehicle.setRoute(self.unique, self.scheduleroute)
        if traci.vehicle.isStopped(self.unique):
            traci.vehicle.resume(self.unique)

        # new_req  != None last pas = None dispath != []

        # new_req  != None last pas != None dispath != []

        # new_req  = None last pas != None dispath != []


        # check if vehicle need to resume
        # first just give route
        # check if on same street then update pick up order
        # if no new request then if has passengers just dispath them
        # need to maintain a pickup dropoff order in terms of requests / passengers

    def get_time_to_next_node(self):
        pos = traci.vehicle.getLanePosition(self.unique)
        time_remain = traci.edge.getTraveltime(self.location) - pos/traci.vehicle.getAllowedSpeed(self.unique)
        assert(time_remain >= 0.0)
        return time_remain

    def finish_route(self, traffic):
        raise(NotImplementedError)

class SUMOTraffic(object):
    def __init__(self, sumonet):
        #super(SUMOTraffic, self).__init__()
        self.sumonet = sumonet
        self.sumoedges = sumonet.getEdges()

    def get_traveltime(self, start, dest, include_start=False):
        stage = traci.simulation.findRoute(start, dest)
        if not include_start:
            return stage.travelTime - traci.edge.getTraveltime(start)

        return stage.travelTime

    def find_path(self, start, dest):
        stage = traci.simulation.findRoute(start, dest)
        path = list(stage.edges)
        return path

class SUMORequest(Request):
    def __init__(self, r):
        super(SUMORequest, self).__init__(r.persons[0], r.fromEdge, r.toEdge, int(r.reservationTime))

class DemandModel(object):
    def __init__(self, filename):
        f = open(filename, "rb")
        demands = pickle.load(f)
        self.num_demands = np.mean(demands['dbd'])
        self.demand_by_interval = [0 for _ in range(96)]
        self.od_by_interval = {i:{"od":[], "w":[]} for i in range(96)}
        for i, s, d in demands['dbift']:
            self.demand_by_interval[i] += demands['dbift'][(i, s, d)]
            self.od_by_interval[i]["od"].append((s,d))
            self.od_by_interval[i]["w"].append(demands['dbift'][(i, s, d)])
    def numDemands(self, interval):
        return int(self.num_demands * self.demand_by_interval[interval] / np.sum(self.demand_by_interval))

    def od_dists(self, interval):
        ods = self.od_by_interval[interval]["od"]
        weights = np.array(self.od_by_interval[interval]["w"], dtype=np.float)
        weights = weights/np.sum(weights)
        return ods, weights

class SUMORideSharing(MDP):
    def __init__(self, net, demands, tazs, sumocfg, num_vehicles=100, min_Tint=24, max_Tint=48):
        self.traffic = SUMOTraffic(net)
        self.sumocfg = sumocfg
        self.tazs = tazs
        self.num_vehicles = num_vehicles
        self.demands = demands
        self.rng = np.random.RandomState(0)
        self.minTint = min_Tint
        self.maxTint = max_Tint
        self.presimulation_steps = 1800
        self.delta_T = TIME_STEP

    def reset(self):
        # pre-simulation
        self.uniques = 0
        self.vehicles = []
        self.requests = []
        self.presimulation()
        # get state
        obs = self.current_state = State(self.requests, self.vehicles)
        #obs, self.current_state = self.get_state()
        return obs

    def execute_agent_action(self, actions):
        self._apply_rl_actions(actions)

        # get next state

        #obs, rew, self.current_state = self.get_state_reward()
        #return obs, rew
        self.current_state = State(self.requests, self.vehicles)
        return 0, self.current_state

    def generate_demands(self):
        interval = int(self.sumostep/60/15)
        numSamples = self.demands.numDemands(interval)
        ods, weights = self.demands.od_dists(interval)
        sampled_demand_indices = self.rng.choice(len(ods), size=numSamples, p=weights)
        print(len(sampled_demand_indices))
        for ind in sampled_demand_indices:
            time = self.rng.randint(interval*15*60, (interval+1)*15*60)
            if time >= self.sumostep:
                o, d = ods[ind]
                start_edge_name = self.tazs[o][self.rng.choice(len(self.tazs[o]))]
                dest_edge_name = self.tazs[d][self.rng.choice(len(self.tazs[d]))]
                start_edge = self.traffic.sumonet.getEdge(start_edge_name)
                dest_edge = self.traffic.sumonet.getEdge(dest_edge_name)
                path = self.traffic.sumonet.getShortestPath(start_edge, dest_edge)
                if path[0]:
                    start_edge_length = start_edge.getLength()
                    traci.person.add('{}'.format(self.uniques), start_edge_name, start_edge_length-0.5, depart=time)
                    new_stage = traci.simulation.Stage(type=3, line="taxi", edges=[start_edge_name, dest_edge_name], description="waiting for taxi")
                    traci.person.appendStage('{}'.format(self.uniques), new_stage)
                    self.uniques += 1

    def presimulation(self):
        if traci.isLoaded():
            traci.close()
        # resample the begin time
        begin_int = self.rng.randint(self.minTint, self.maxTint)
        traci.start([SUMOEXE, "-c", self.sumocfg, "-b", "{}".format(begin_int*15*60)])
        # populate the traffic
        for j in range(self.presimulation_steps):
            if j == self.presimulation_steps-1:
                # generate vehicles
                for n in range(self.num_vehicles):
                    edge_ind = self.rng.choice(len(self.traffic.sumoedges))
                    self.vehicles.append(SUMOVehicle(n, loc=self.traffic.sumoedges[edge_ind].getID()))

            traci.simulationStep()
            self.sumostep = int(traci.simulation.getTime())
            # generate_demands
            if self.sumostep % DEMAND_GEN == 0: # new demands for the next 15 min are created

                print("here", self.sumostep)
                self.generate_demands()

        sumo_reqs = traci.person.getTaxiReservations(0)
        for r in sumo_reqs:
            if self.sumostep - r.reservationTime > MAX_WAIT_SEC:
                #delete
                for p in r.persons:
                    traci.person.removeStages(p) #TODO fatal error may occur when reservation id is dispatched
            else:
                req = SUMORequest(r)
                req.expectedOffTime = self.traffic.get_traveltime(req.start, req.dest, include_start=True) + req.reqTime
                self.requests.append(req)

    def _apply_rl_actions(self, rl_actions):
        """
        Apply individual agent actions.

        :param rl_actions: dictionary of format {agent_id : action vector}.
        """
        served_inds = []
        for veh_ind, action in rl_actions:
            veh = self.vehicles[veh_ind]
            if type(action) == list:
                # trip plan
                #time_to_next_junction = veh.get_time_to_next_junction()
                travel(veh, self.sumostep, trip, self.traffic, True)
                for req in action:
                    served_inds.append(req.unique)
            else:
                # rebalance plan
                veh.head_for(action, self.sumostep, self.traffic)

        # step forward
        new_requests = []
        for j in range(self.delta_T):
            traci.simulationStep()
            self.sumostep = int(traci.simulation.getCurrentTime()/1000)
            # some thing need to be checked every step
            # update vehicles
            for veh in self.vehicles:
                veh.update(self.sumostep, self.requests, self.traffic)
            # generate_demands
            if self.sumostep % DEMAND_GEN == 0: # new demands for the next 15 min are created
                self.generate_demands()
            # collect requests
            sumo_reqs = traci.person.getTaxiReservations(1)
            for r in sumo_reqs:
                req = SUMORequest(r)
                req.expectedOffTime = self.traffic.get_traveltime(req.start, req.dest, include_start=True) + req.reqTime
                new_requests.append(req)


        for req in self.requests:
            if req.unique not in served_inds:
                new_requests.append(req)

        for veh in self.vehicles:
            new_requests.extend(veh.finish_update(self.sumostep, self.requests, self.traffic))

        # deal with timed out request
        self.requests = []
        for req in new_requests:
            if self.sumostep - req.reqTime > MAX_WAIT_SEC:
                # remove these request
                traci.person.removeStages(req.unique) #TODO one person per request
            else:
                self.requests.append(req)


    def compute_reward(self, rl_actions, **kwargs):
        """In this example, all agents receive a reward of 10"""
        reward_dict = {}
        for rl_id, action in rl_actions.items():
            reward_dict[rl_id] = 10
        return reward_dict

    def get_state(self, **kwargs):
        """Every agent observes its own speed"""
        obs_dict = {}
        for rl_id in self.k.vehicle.get_rl_ids():
            obs_dict[rl_id] = self.k.vehicle.get_speed(rl_id)
        return obs_dict
