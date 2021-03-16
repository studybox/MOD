import os, sys, random, time, math
import numpy as np
from numpy.random import randint
from collections import defaultdict
from simple_rl.tasks import FourRoomMDP
from simple_rl.planning import ValueIteration
from simple_rl.mdp.MDPClass import MDP
import networkx as nx
import pygame
from collections import defaultdict
from itertools import permutations
import gurobipy as gp
from gurobipy import GRB
import scipy.sparse as sp

TIME_STEP = 1
MAX_DELAY_SEC = 10
MAX_WAIT_SEC = 15
MAX_CAPACITY = 4
MAX_V_PER_REQ = 10
PENALTY = 25



class Vehicle(object):
    def __init__(self, unique, loc):
        self.timeToNextNode =  0
        self.location = loc
        self.available = True
        self.passengers = []
        self.scheduleroute = []
        self.schedulepassengers = []
        self.unique = unique
        self.served_reqs = 0
        self.draw_route = False

    def isAvailable(self):
        return self.available
    def get_location(self):
        return self.location
    def set_location(self, loc):
        self.location = loc

    def get_time_to_next_node(self):
        return 0#self.timeToNextNode

    def get_num_passengers(self):
        return len(self.passengers)
    def __repr__(self):
        return "veh {}: at {} num of pass {}".format(self.unique, self.location, len(self.passengers))

    def __str__(self):
        ret = "veh {}: at {} num of pass {}".format(self.unique, self.location, len(self.passengers))

        for pas in self.passengers:
            if pas.onBoard:
                ret += " {} onboard".format(pas.unique)
            else:
                ret += "{} offboard".format(pas.unique)
        return ret

    def head_for(self, loc, now_time, traffic):
        path = traffic.find_path(self.location, loc)
        self.scheduleroute = []
        if self.timeToNextNode <= TIME_STEP:
            baseTime = now_time #self.timeToNextNode +
        else:
            baseTime = -self.timeToNextNode + now_time
        tmpTime = baseTime
        for idx in range(1, len(path)):
            tmpTime += traffic.get_traveltime(path[idx-1], path[idx])
            self.scheduleroute.append((tmpTime, path[idx]))

    def update(self, nowTime, newRequests, traffic):
        #if len(self.scheduleroute) == 0:
        #    return []
        #print("veh ", self.unique)
        if self.timeToNextNode <= TIME_STEP:
            while len(self.scheduleroute) > 0:
                schedTime, nextLoc = self.scheduleroute[0]
                #print("sch, loc ", schedTime, nextLoc)
                if schedTime <= nowTime:# or schedTime - nowTime <= TIME_STEP:
                    #if len(self.passengers) > 0:
                    #    onboardCnt = 0
                    #    for pas in self.passengers:
                    #        if pas.scheduledOnTime < schedTime:
                    #            onboardCnt += 1
                    #        if pas.scheduledOffTime < schedTime:
                    #            onboardCnt -= 1
                    self.location = nextLoc
                    self.scheduleroute.pop(0)
                    #print("here1")
                if schedTime > nowTime:
                    self.timeToNextNode = schedTime - nowTime
                    self.available = self.timeToNextNode <= TIME_STEP
                    #print("here2")
                    break
        else:
            #print("here3")
            self.timeToNextNode -= TIME_STEP
            self.available = self.timeToNextNode < TIME_STEP
            if self.available:
                schedTime, nextLoc = self.scheduleroute[0]
                #if len(self.passengers) > 0:
                #    onboardCnt = 0
                #    for pas in self.passengers:
                #        if pas.scheduledOnTime < schedTime:
                #            onboardCnt += 1
                #        if pas.scheduledOffTime < schedTime:
                #            onboardCnt -= 1
                self.location = nextLoc
                self.scheduleroute.pop(0)
        if self.available:
            baseTime = nowTime# + self.timeToNextNode
        else:
            baseTime =  nowTime

        newPassengers = []
        schPassengers = []
        newRequests = []
        for pas in self.passengers:
            if pas.scheduledOnTime > baseTime:
                assert pas.onBoard ==False
                newRequests.append(pas)
                if len(schPassengers) == 0:
                    schPassengers.append(pas)
                elif pas.scheduledOnTime < schPassengers[0].scheduledOnTime:
                    schPassengers[0] = pas

            elif pas.scheduledOffTime <= baseTime:
                self.served_reqs += 1
                #total_wait_time += pas.scheduledOnTime - pas.reqTime
            else:
                pas.onBoard = True
                newPassengers.append(pas)
        self.passengers = newPassengers
        self.schedulepassengers = schPassengers
        return newRequests

    def set_path(self, now_time, path, traffic):
        self.scheduleroute = []

        if self.timeToNextNode < TIME_STEP:
            baseTime = self.timeToNextNode + now_time
        else:
            baseTime = -self.timeToNextNode + now_time
        tmpTime = now_time
        assert(path[0] == self.location)
        for idx in range(1, len(path)):
            tmpTime += traffic.get_traveltime(path[idx-1], path[idx])
            self.scheduleroute.append((tmpTime, path[idx]))

    def set_passengers(self, schedule):
        self.passengers = [req for req in schedule]
    def finish_route(self, traffic):
        if len(self.passengers) > 0:
            for pas in self.passengers:
                served_reqs += 1
                total_wait_time += pas.scheduledOnTime - pas.reqTime
            while len(self.scheduleroute) > 0:
                schedTime, nextLoc = self.scheduleroute[0]
                if len(self.passengers) > 0:
                    onboardCnt = 0
                    for pas in self.passengers:
                        if pas.scheduledOnTime < schedTime:
                            onboardCnt += 1
                        if pas.scheduledOffTime < schedTime:
                            onboardCnt -= 1
                self.location = nextLoc
                self.scheduleroute.pop(0)

class State(object):
    def __init__(self, requests, vehicles):
        self.requests = requests
        self.vehicles = vehicles
        self.done = False
    def is_terminal(self):
        return self.done
    def __str__(self):
        return ""

class Traffic(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    def get_traveltime(self, start, dest):
        return np.abs(start[0]-dest[0]) + np.abs(start[1]-dest[1])
    def find_path(self, start, dest):
        path = []
        if dest[0] < start[0]:
            x = start[0]
            while x >= dest[0]:
                path.append((x, start[1]))
                x -= 1
        else:
            x = start[0]
            while x <= dest[0]:
                path.append((x, start[1]))
                x += 1
        if dest[1] <= start[1]:
            y = start[1]-1
            while y >= dest[1]:
                path.append((dest[0], y))
                y -= 1
        else:
            y = start[1]+1
            while y <= dest[1]:
                path.append((dest[0], y))
                y += 1
        return path

class Request(object):
    def __init__(self, unique, start,  dest, reqTime):
        self.start = start
        self.dest = dest
        self.reqTime = reqTime
        self.onBoard = False
        self.scheduledOnTime = -1
        self.scheduledOffTime = -1

        self.expectedOffTime = -1
        self.unique = unique
        self.assign = 0

    def __repr__(self):
        if self.onBoard:
            ret = " pass {} from {} to {} started at {}".format(self.unique, self.start, self.dest, self.reqTime)
        else:
            ret = " reqs {} from {} to {} sent at {}".format(self.unique, self.start, self.dest, self.reqTime)
        return ret

class GridWorldRideSharing(MDP):
    def __init__(self, width=20, height=20, num_vehicles=10):
        self.width = width
        self.height = height
        self.traffic = Traffic(width, height)
        self.num_vehicles = num_vehicles
        self.draw_routes = [i for i in range(min(num_vehicles, 1))]
        self.num_samples = 20000
        self.rng = np.random.RandomState(0)
        self.ODs = defaultdict(lambda: 0)
        random_ODs = []
        while len(random_ODs) < self.num_samples:
            o , d = self.rng.randint(0,width*height), self.rng.randint(0,height*width)
            x, y = o//self.height, o%self.width
            if x >= 8 and x <= 12 and y >=8 and y <= 12:
                random_ODs.append((o, d))
        #random_ODs = [(self.rng.randint(0,width*height), self.rng.randint(0,height*width)) for _ in range(self.num_samples)]
        for sample in random_ODs:
            self.ODs[sample] += 1

        self.requests = []
        self.uniques = 0
        self.cur_step = 0
        self.generate_new_requests()

        self.vehicles = [Vehicle(idx, (self.rng.randint(0,width), self.rng.randint(0,height))) for idx in range(self.num_vehicles)]
        for veh_ind in self.draw_routes:
            self.vehicles[veh_ind].draw_route = True
        #self.vehicles[0].passengers.append(1)
        #self.vehicles[1].passengers.extend([1,2])
        #self.vehicles[2].passengers.extend([1,2,3])
        #self.vehicles[3].passengers.extend([1,2,3,4])

        self.current_state = State(self.requests, self.vehicles)
    def get_request_locs(self):
        locs = []
        for req in self.requests:
            locs.append(req.start)
        return locs

    def get_actions(self):
        return [1,2,3,4]

    def get_vehicle_locs(self):
        locs = []
        for veh in self.vehicles:
            locs.append(veh.get_location())
        return locs

    def reset(self):
        self.vehicles = [Vehicle(idx, (self.rng.randint(0,self.width), self.rng.randint(0,self.height))) for idx in range(self.num_vehicles)]
        self.requests = []
        self.uniques = 0
        self.cur_step = 0
        self.generate_new_requests()
        self.current_state = State(self.requests, self.vehicles)
        return self.current_state

    def generate_new_requests(self):
        for o in range(self.width*self.height):
            for d in range(self.width*self.height):
                if o != d:
                    theta = self.rng.beta(1+self.ODs[(o,d)], 1+self.num_samples-self.ODs[(o,d)])
                    if self.rng.uniform() < theta:
                        req = Request(self.uniques,
                                        (o//self.height, o%self.width),
                                        (d//self.height, d%self.width),
                                        self.cur_step)
                        req.expectedOffTime = self.traffic.get_traveltime(req.start, req.dest) + self.cur_step
                        self.requests.append(req)
                        self.uniques += 1

    def execute_agent_action(self, action):
        # update the vehicles (they moves)
        self.update_vehicles(action)
        # update the requests (new generated)
        if self.cur_step % 5 == 0 and self.cur_step <= 50:
            #print("New requests")
            self.generate_new_requests()
        if self.cur_step % 150 == 0:
            for idx, veh_ind in enumerate(self.draw_routes):
                self.vehicles[veh_ind].draw_route = False
                self.draw_routes[idx] += len(self.draw_routes)
                self.draw_routes[idx] = self.draw_routes[idx] % len(self.vehicles)
                self.vehicles[self.draw_routes[idx]].draw_route = True

        self.current_state = State(self.requests, self.vehicles)
        return 0, self.current_state

    def update_vehicles(self, action):
        served_inds = []
        # plans for every vehicle
        for veh in self.vehicles:
            veh_ind = veh.unique
            ret = action[veh_ind]
            if type(ret) == list:
                # trips plan
                trip = ret
                travel(veh, self.cur_step, trip, self.traffic, True)
                for req in trip:
                    served_inds.append(req.unique)
            else:
                # rebalance plan
                loc = ret
                veh.head_for(loc, self.cur_step, self.traffic)
        # move to next step
        self.cur_step += 1

        new_requests = []
        for req in self.requests:
            if req.unique not in served_inds:

                new_requests.append(req)

        # execution
        for veh in self.vehicles:
            new_requests.extend(veh.update(self.cur_step, self.requests, self.traffic))
        self.requests = new_requests
        for req in self.requests:
            if self.cur_step - req.reqTime > MAX_WAIT_SEC:
                assert(self.cur_step - req.reqTime == MAX_WAIT_SEC+1)
                assert(req.assign == 0)
                req.reqTime += 1
                req.expectedOffTime += 1
                req.scheduledOnTime = req.scheduledOffTime = -1

    def visualize_interaction(self):
        from simple_rl.utils import mdp_visualizer as mdpv
        mdpv.visualize_interaction(self, _draw_state, self.current_state)

    def visualize_agent(self, agent):

        from simple_rl.utils import mdp_visualizer as mdpv
        mdpv.visualize_agent(self, agent, _draw_state, self.current_state)


def _draw_state(screen, grid_mdp, state, agent_shape=None, draw_statics=False):
    # Prep some dimensions to make drawing easier.
    scr_width, scr_height = screen.get_width(), screen.get_height()
    width_buffer = scr_width / 10.0
    height_buffer = 30 + (scr_height / 10.0) # Add 30 for title.
    cell_width = (scr_width - width_buffer * 2) / grid_mdp.width
    cell_height = (scr_height - height_buffer * 2) / grid_mdp.height

    #request_locs = grid_mdp.get_request_locs()
    #vehicle_locs = grid_mdp.get_vehicle_locs()

    font_size = int(min(cell_width, cell_height) / 4.0)
    reg_font = pygame.font.SysFont("CMU Serif", font_size)
    cc_font = pygame.font.SysFont("Courier", font_size*2 + 2)

    # Draw the static entities.
    if draw_statics:
        # For each row:
        for i in range(grid_mdp.width):
            # For each column:
            for j in range(grid_mdp.height):
                top_left_point = width_buffer + cell_width*i, height_buffer + cell_height*j
                r = pygame.draw.rect(screen, (46, 49, 49), top_left_point + (cell_width, cell_height), 3)

        # current_state
        location_dict = defaultdict(lambda:[[],[]])
        for req in state.requests:
            loc = req.start
            location_dict[loc][1].append(req)
        for veh in state.vehicles:
            loc = veh.get_location()
            location_dict[loc][0].append(veh)
        agent_shape = []
        for loc in location_dict:
            if len(location_dict[loc][0]) > 0 and len(location_dict[loc][1]) > 0:
                # vehicles and requests
                top_left_point = width_buffer + cell_width*loc[1], height_buffer + cell_height*(loc[0]+1)
                veh_center = int(top_left_point[0] + cell_width/4.0), int(top_left_point[1] - cell_height/2.0)
                num_pass = len(location_dict[loc][0][0].passengers)
                vehicle_shape = _draw_vehicle(veh_center, num_pass, screen, base_size=min(cell_width, cell_height)/5.0)
                req_center = int(top_left_point[0] + cell_width/4.0*3.0), int(top_left_point[1] - cell_height/2.0)
                request_shape =_draw_request(req_center, screen, base_size=min(cell_width, cell_height)/5.0)
                agent_shape.extend(vehicle_shape)
                agent_shape.append(request_shape)
            elif len(location_dict[loc][0]) > 0:
                # vehicles
                top_left_point = width_buffer + cell_width*loc[1], height_buffer + cell_height*(loc[0]+1)
                veh_center = int(top_left_point[0] + cell_width/2.0), int(top_left_point[1] - cell_height/2.0)
                num_pass = len(location_dict[loc][0][0].passengers)
                vehicle_shape = _draw_vehicle(veh_center, num_pass, screen, base_size=min(cell_width, cell_height)/2.5)
                agent_shape.extend(vehicle_shape)
            else:
                # requests
                top_left_point = width_buffer + cell_width*loc[1], height_buffer + cell_height*(loc[0]+1)
                req_center = int(top_left_point[0] + cell_width/2.0), int(top_left_point[1] - cell_height/2.0)
                request_shape =_draw_request(req_center, screen, base_size=min(cell_width, cell_height)/2.5)
                agent_shape.append(request_shape)

    if agent_shape is not None:
        # Clear the old shape.
        for agent in agent_shape:
            pygame.draw.rect(screen, (255,255,255), agent)

        # current_state
        location_dict = defaultdict(lambda:[[],[]])
        route_locs = []
        for req in state.requests:
            loc = req.start
            location_dict[loc][1].append(req)
        for veh in state.vehicles:
            loc = veh.get_location()
            location_dict[loc][0].append(veh)
            if veh.draw_route:
                route_locs.append((veh.get_location(), (0,0), (0,0)))
                for lidx, (sch, loc) in enumerate(veh.scheduleroute):
                    if lidx == 0:
                        if len(veh.scheduleroute) >= 2:
                            route_locs.append((loc, (veh.get_location()[0]-loc[0],veh.get_location()[1]-loc[1]),
                                                    (veh.scheduleroute[lidx+1][1][0]-loc[0], veh.scheduleroute[lidx+1][1][1]-loc[1])))
                        else:
                            route_locs.append((loc, (veh.get_location()[0]-loc[0],veh.get_location()[1]-loc[1]), (0,0)))
                    elif lidx == len(veh.scheduleroute)-1:
                        route_locs.append((loc, (veh.scheduleroute[lidx-1][1][0]-loc[0], veh.scheduleroute[lidx-1][1][1]-loc[1]),
                                                (0,0)))
                    else:
                        route_locs.append((loc, (veh.scheduleroute[lidx-1][1][0]-loc[0], veh.scheduleroute[lidx-1][1][1]-loc[1]),
                                                (veh.scheduleroute[lidx+1][1][0]-loc[0], veh.scheduleroute[lidx+1][1][1]-loc[1])))

        agent_shape = []

        for loc in location_dict:
            if len(location_dict[loc][0]) > 0 and len(location_dict[loc][1]) > 0:
                # vehicles and requests
                top_left_point = width_buffer + cell_width*loc[1], height_buffer + cell_height*(loc[0]+1)
                veh_center = int(top_left_point[0] + cell_width/4.0), int(top_left_point[1] - cell_height/2.0)
                num_pass = len(location_dict[loc][0][0].passengers)
                vehicle_shape = _draw_vehicle(veh_center, num_pass, screen, base_size=min(cell_width, cell_height)/5.0)
                req_center = int(top_left_point[0] + cell_width/4.0*3.0), int(top_left_point[1] - cell_height/2.0)
                request_shape =_draw_request(req_center, screen, base_size=min(cell_width, cell_height)/5.0)
                agent_shape.extend(vehicle_shape)
                agent_shape.append(request_shape)
            elif len(location_dict[loc][0]) > 0:
                # vehicles
                top_left_point = width_buffer + cell_width*loc[1], height_buffer + cell_height*(loc[0]+1)
                veh_center = int(top_left_point[0] + cell_width/2.0), int(top_left_point[1] - cell_height/2.0)
                num_pass = len(location_dict[loc][0][0].passengers)
                vehicle_shape = _draw_vehicle(veh_center, num_pass, screen, base_size=min(cell_width, cell_height)/2.5)
                agent_shape.extend(vehicle_shape)
            else:
                # requests
                top_left_point = width_buffer + cell_width*loc[1], height_buffer + cell_height*(loc[0]+1)
                req_center = int(top_left_point[0] + cell_width/2.0), int(top_left_point[1] - cell_height/2.0)
                request_shape =_draw_request(req_center, screen, base_size=min(cell_width, cell_height)/2.5)
                agent_shape.append(request_shape)

        for loc, last, next  in route_locs:
            path_color = (98, 140, 150, 10)
            base_size=min(cell_width, cell_height)/5.0
            top_left_point = width_buffer + cell_width*loc[1] + cell_width//2.0 - base_size , height_buffer + cell_height*(loc[0]+1) - cell_height//2.0 - base_size
            agent_shape.append(pygame.draw.rect(screen, path_color, top_left_point+(base_size*2,base_size*2)))
            if last == (1,0):
                top_left_point = width_buffer + cell_width*loc[1] + cell_width//2.0 - base_size , height_buffer + cell_height*(loc[0]+1) - cell_height//2.0
                agent_shape.append(pygame.draw.rect(screen, path_color, top_left_point+(base_size*2,base_size*2)))
            elif last == (-1,0):
                top_left_point = width_buffer + cell_width*loc[1] + cell_width//2.0 - base_size , height_buffer + cell_height*(loc[0]+1) - cell_height//2.0 - 2*base_size
                agent_shape.append(pygame.draw.rect(screen, path_color, top_left_point+(base_size*2,base_size*2)))
            elif last == (0,-1):
                top_left_point = width_buffer + cell_width*loc[1] + cell_width//2.0 - 2*base_size , height_buffer + cell_height*(loc[0]+1) - cell_height//2.0 - base_size
                agent_shape.append(pygame.draw.rect(screen, path_color, top_left_point+(base_size*2,base_size*2)))
            elif last == (0,1):
                top_left_point = width_buffer + cell_width*loc[1] + cell_width//2.0 , height_buffer + cell_height*(loc[0]+1) - cell_height//2.0 - base_size
                agent_shape.append(pygame.draw.rect(screen, path_color, top_left_point+(base_size*2,base_size*2)))

            if next == (1,0):
                top_left_point = width_buffer + cell_width*loc[1] + cell_width//2.0 - base_size , height_buffer + cell_height*(loc[0]+1) - cell_height//2.0
                agent_shape.append(pygame.draw.rect(screen, path_color, top_left_point+(base_size*2,base_size*2)))
            elif next == (-1,0):
                top_left_point = width_buffer + cell_width*loc[1] + cell_width//2.0 - base_size , height_buffer + cell_height*(loc[0]+1) - cell_height//2.0 - 2*base_size
                agent_shape.append(pygame.draw.rect(screen, path_color, top_left_point+(base_size*2,base_size*2)))
            elif next == (0,-1):
                top_left_point = width_buffer + cell_width*loc[1] + cell_width//2.0 - 2*base_size , height_buffer + cell_height*(loc[0]+1) - cell_height//2.0 - base_size
                agent_shape.append(pygame.draw.rect(screen, path_color, top_left_point+(base_size*2,base_size*2)))
            elif next == (0,1):
                top_left_point = width_buffer + cell_width*loc[1] + cell_width//2.0 , height_buffer + cell_height*(loc[0]+1) - cell_height//2.0 - base_size
                agent_shape.append(pygame.draw.rect(screen, path_color, top_left_point+(base_size*2,base_size*2)))

    pygame.display.flip()

    return agent_shape

def _draw_request(center_point, screen, base_size=20):
    circler_color = (255,127,80)
    request_shape = pygame.draw.circle(screen, circler_color,  center_point, base_size)
    return request_shape

def _draw_vehicle(center_point, num_pass, screen, base_size=20):
    vehicle_shape = []
    vehicle_color = (98, 140, 190, 10)
    request_color = (154, 195, 157, 10)
    # TODO: draw route out side vehicle box, draw intended passengers
    top_left_point = (center_point[0]-base_size, center_point[1]-base_size)
    vehicle_shape.append(pygame.draw.rect(screen, vehicle_color, top_left_point+(base_size*2,base_size*2), 2))
    if num_pass == 0:
        pass
    elif num_pass == 1:
        vehicle_shape.append(pygame.draw.circle(screen, request_color, center_point, base_size))
    elif num_pass == 2:
        vehicle_shape.append(pygame.draw.circle(screen, request_color, (center_point[0]-base_size//2, center_point[1]), base_size//2))
        vehicle_shape.append(pygame.draw.circle(screen, request_color, (center_point[0]+base_size//2, center_point[1]), base_size//2))
    elif num_pass == 3:
        vehicle_shape.append(pygame.draw.circle(screen, request_color, (center_point[0]-base_size//2, center_point[1]-base_size//2), base_size//3))
        vehicle_shape.append(pygame.draw.circle(screen, request_color, (center_point[0]+base_size//2, center_point[1]-base_size//2), base_size//3))
        vehicle_shape.append(pygame.draw.circle(screen, request_color, (center_point[0], center_point[1]+base_size//2), base_size//3))
    else:

        vehicle_shape.append(pygame.draw.circle(screen, request_color, (center_point[0]-base_size//2, center_point[1]-base_size//2), base_size//4))
        vehicle_shape.append(pygame.draw.circle(screen, request_color, (center_point[0]+base_size//2, center_point[1]-base_size//2), base_size//4))
        vehicle_shape.append(pygame.draw.circle(screen, request_color, (center_point[0]-base_size//2, center_point[1]+base_size//2), base_size//4))
        vehicle_shape.append(pygame.draw.circle(screen, request_color, (center_point[0]+base_size//2, center_point[1]+base_size//2), base_size//4))
    return vehicle_shape


def travel(vehicle, now_time, requests, traffic, decided, first_req=False):
    targets = set()
    tmp_targets = set()
    src_dst = defaultdict(lambda:[])
    numReqs = len(requests)
    for req in requests:
        targets.add(req.start)
        src_dst[req.start].append(req.dest)
        tmp_targets.add(req.start)
        tmp_targets.add(req.dest)
    for pas in vehicle.passengers:
        targets.add(pas.dest)
        tmp_targets.add(pas.dest)

    #path, schedule
    now_time = now_time + vehicle.get_time_to_next_node()

    visited = {t:0 for t in targets}
    best_path = None
    best_schedule = None
    best_cost = float('inf')#MAX_DELAY_SEC * numReqs +1


    for path in permutations(tmp_targets, len(tmp_targets)):
        # some path are invalid due to wrong ori-dest
        visitable_locs = [t for t in targets]
        #getOns = []
        #getOffs = []
        req_schonTimes = []
        req_schoffTimes = []
        total_cost = 0.0
        schedule = []
        cur_time = now_time
        prevloc = vehicle.get_location()
        wait_delays = {}
        for cur_loc in path:
            exceeded = False
            cur_time += traffic.get_traveltime(prevloc, cur_loc)
            if cur_loc in src_dst:
                for loc in src_dst[cur_loc]:
                    visitable_locs.append(loc)
                for req in requests:
                    if req.start == cur_loc and not req.onBoard:
                        if cur_time - req.reqTime > MAX_WAIT_SEC:
                            exceeded = True
                            break
                        req.onBoard = True
                        wait_delays[req.unique] = cur_time - req.reqTime
                        if decided:
                            req_schonTimes.append((req, cur_time))
                        #getOns.append(req)

            #print(getOns)
            if cur_loc not in visitable_locs or exceeded:
                #invalid route
                break


            for req in requests+vehicle.passengers:
                if req.onBoard:
                    if req.unique in wait_delays:
                        wait_time = wait_delays[req.unique]
                    else:
                        assert req.scheduledOnTime != -1
                        wait_time = req.scheduledOnTime - req.reqTime
                    if cur_time - req.expectedOffTime - wait_time > MAX_DELAY_SEC:
                        exceeded = True
                        break
                    if req.dest == cur_loc:
                        req.onBoard = False
                        if decided:
                            req_schoffTimes.append((req, cur_time))
                        schedule.append(req)
                        total_cost += cur_time - req.expectedOffTime
                        #getOffs.append(req)
            prevloc = cur_loc
            if exceeded:
                break
            if total_cost >= best_cost:
                break
        else:
            best_cost = total_cost
            best_path = path
            best_schedule = schedule
            #print("::", req_schonTimes)
            for req, ontime in req_schonTimes:
                req.scheduledOnTime = ontime
            for req, offtime in req_schoffTimes:
                req.scheduledOffTime = offtime

        for req in requests:
            req.onBoard = False
        for req in vehicle.passengers:
            req.onBoard = True
            #
    #print([req.scheduledOnTime for req in best_schedule])
    if best_cost < float('inf'):#MAX_DELAY_SEC * numReqs+1:
        if first_req:
            prevloc = vehicle.get_location()
            for pidx in range(len(best_path)):
                if pidx == 0:
                    for req in requests:
                        if req.start == prevloc:
                            return req
                for req in requests:
                    if req.start == best_path[pidx]:
                        return req
            raise
        if decided:
            prevloc = vehicle.get_location()
            finalPath = []
            for pidx in range(len(best_path)):
                if pidx == 0:
                    rou = traffic.find_path(prevloc, best_path[pidx])
                    finalPath.extend(rou)
                else:
                    rou = traffic.find_path(best_path[pidx-1], best_path[pidx])
                    finalPath.extend(rou[1:])
            assert(len(finalPath) > 0)
            vehicle.set_path(now_time, finalPath, traffic)
            vehicle.set_passengers(best_schedule)
        return best_cost
    else:
        return -1

class Agent(object):
    def __init__(self, mdp):
        self.mdp = mdp

    def act(self, state, reward):
        self.rvgraph = RVGraph(self.mdp.cur_step, state.vehicles, state.requests, self.mdp.traffic)
        self.rtvgraph = RTVGraph(self.rvgraph, self.mdp.cur_step, state.vehicles, state.requests, self.mdp.traffic)
        self.action = self.rtvgraph.solve()
        #print(self.action)
        return self.action

class RVGraph(object):
    def __init__(self, cur_time, vehicles, requests, traffic):
        #create request to request edges.
        self.req_inter = []
        self.car_req_cost = defaultdict(lambda:dict())
        self.req_cost_car = defaultdict(lambda:[])
        if len(requests) > 0:
            virtualCar = Vehicle(-1, requests[0].start)
        for i in range(len(requests)):
            for j in range(i+1, len(requests)):
                virtualCar.set_location(requests[i].start)
                cost = travel(virtualCar, cur_time, [requests[i], requests[j]], traffic, False)
                if cost >= 0:
                    self.req_inter.append((i, j, cost))
                else:
                    virtualCar.set_location(requests[j].start)
                    cost = travel(virtualCar, cur_time, [requests[j], requests[i]], traffic, False)
                    if cost >= 0 :
                        self.req_inter.append((i, j, cost))
        #create request to vehicle edges
        for i, veh in enumerate(vehicles):
            if veh.isAvailable() and veh.get_num_passengers() < MAX_CAPACITY:
                for j, req in enumerate(requests):
                    cost = travel(veh, cur_time, [req], traffic, False)
                    if cost >= 0:
                        self.car_req_cost[i][j] = cost
                        self.req_cost_car[j].append((i, cost))

        for req_ind in self.req_cost_car:
            self.req_cost_car[req_ind].sort(key=lambda x: x[1])
            if len(self.req_cost_car[req_ind]) <= MAX_V_PER_REQ:
                continue
            #print(req_ind, self.req_cost_car[req_ind])
            #print(self.car_req_cost)
            for idx in range(MAX_V_PER_REQ, len(self.req_cost_car[req_ind])):
                veh_ind = self.req_cost_car[req_ind][idx][0]
                #print(veh_ind, req_ind)
                #print(self.car_req_cost[veh_ind][req_ind])
                del self.car_req_cost[veh_ind][req_ind]
            self.req_cost_car[req_ind] = self.req_cost_car[req_ind][:MAX_V_PER_REQ]

    def has_vehicle(self, veh_ind):
        return len(self.car_req_cost[veh_ind]) > 0
    def get_vehicle_num(self):
        return len(self.car_req_cost)
    def has_reqs_edge(self, r1_ind, r2_ind):
        for i, j, c in self.req_inter:
            if i == r1_ind and j == r2_ind:
                return True
            if j == r1_ind and i == r2_ind:
                return True
        return False
    def get_vehicle_edges(self, veh_ind):
        edges = []
        for req_ind in self.car_req_cost[veh_ind]:
            edges.append((req_ind, self.car_req_cost[veh_ind][req_ind]))
        return edges

class RTVGraph(object):
    def __init__(self, rvgraph, cur_time, vehicles, requests, traffic):
        self.trips = []
        self.trip_inds = dict()
        self.numReqs = len(requests)
        self.numTrips = 0
        self.numVehicles = 0
        self.vehicles = vehicles
        self.requests = requests
        self.traffic = traffic
        self.veh_inds = []
        self.veh_ind_trip_inds = defaultdict(lambda: [])
        self.req_ind_trip_inds = defaultdict(lambda: set())
        self.trip_ind_cost_inds = defaultdict(lambda: [])
        self.VKO_assigned = []
        self.VKO_idle = []
        self.RKO = []
        for i in range(len(vehicles)):
            if rvgraph.has_vehicle(i):
                self.build_single_vehicle(i, rvgraph, cur_time, vehicles, requests, traffic)
            else:
                if len(vehicles[i].passengers) == 0:
                    self.VKO_idle.append(i)
                else:
                    self.VKO_assigned.append(i)
        self.sort_edges()

    def build_single_vehicle(self, veh_ind, rvgraph, cur_time, vehicles, requests, traffic):
        veh = vehicles[veh_ind]
        self.veh_inds.append(veh_ind)
        self.numVehicles += 1

        tIdxListOfCapacity = [[] for _ in range(MAX_CAPACITY+1)]
        # add trips of size 1
        if MAX_CAPACITY - veh.get_num_passengers() > 0:
            edges = rvgraph.get_vehicle_edges(veh_ind)
            for req_ind, cost in edges:
                trip_ind = self.get_trip_ind((req_ind,))
                tIdxListOfCapacity[1].append(trip_ind)
                self.add_edge_trip_vehicle(trip_ind, (cost, veh_ind))
        # add trips of size 2
        prevSize = len(tIdxListOfCapacity[1])
        for i in range(len(tIdxListOfCapacity[1])):
            for j in range(i+1, len(tIdxListOfCapacity[1])):
                req_ind1 = self.trips[tIdxListOfCapacity[1][i]][0]
                req_ind2 = self.trips[tIdxListOfCapacity[1][j]][0]
                if rvgraph.has_reqs_edge(req_ind1, req_ind2):
                    cost = travel(veh, cur_time, [requests[req_ind1], requests[req_ind2]], traffic, False)
                    if cost >= 0:
                        trip = (req_ind1, req_ind2)
                        trip_ind = self.get_trip_ind(trip)
                        tIdxListOfCapacity[2].append(trip_ind)
                        self.add_edge_trip_vehicle(trip_ind, (cost, veh_ind))
        # add trips of size k
        for k in range(3, MAX_CAPACITY-veh.get_num_passengers()+1):
            #print("trip size :", k, "previous size: ", len(tIdxListOfCapacity[k-1]))
            prevSize = len(tIdxListOfCapacity[k-1])
            trip_list = []
            for i in range(len(tIdxListOfCapacity[k-1])):
                for j in range(i+1, len(tIdxListOfCapacity[k-1])):
                    trip1 = self.trips[tIdxListOfCapacity[k-1][i]]
                    trip2 = self.trips[tIdxListOfCapacity[k-1][j]]
                    trip = set()
                    for req in trip1+trip2:
                        trip.add(req)
                    if len(trip) != k:
                        continue
                    if trip in trip_list:
                        continue
                    else:
                        trip_list.append(trip)
                    allSubsExist = True

                    #print(trip1, trip2, trip)
                    for req_ex in trip:
                        subExist = False
                        for sub in tIdxListOfCapacity[k-1]:
                            if self.equal_to_sub(self.trips[sub], trip, req_ex):
                                subExist = True
                                break
                        if not subExist:
                            allSubsExist = False
                            break
                    if not allSubsExist:
                        continue
                    reqs = [requests[tr] for tr in trip]
                    cost = travel(veh, cur_time, reqs, traffic, False)
                    if cost >= 0:
                        trip_ind = self.get_trip_ind(tuple(trip))
                        tIdxListOfCapacity[k].append(trip_ind)
                        self.add_edge_trip_vehicle(trip_ind, (cost, veh_ind))

    def equal_to_sub(self, compared, origin_set, exclude):
        origin = set(origin_set)
        origin.remove(exclude)
        if origin.union(compared) == origin:
            return True
        else:
            return False
        #origin = list(origin_set)
        #for idx in range(len(origin)):
        #    if idx == excludeidx:
        #        continue
        #    if origin[idx] not in compared:
        #        return False
        #else:
        #    return True

    def solve(self):
        ROK = []
        self.actions = [None for _ in range(len(self.VKO_idle)+len(self.VKO_assigned)+self.numVehicles)]
        if self.numVehicles > 0:
            model = gp.Model("ilp")
            model.setParam('OutputFlag', 0)

            self.num_edges = 0
            self.edge_dict = {}
            for veh_ind in self.veh_ind_trip_inds:
                for trip_ind in self.veh_ind_trip_inds[veh_ind]:
                    self.edge_dict[veh_ind, trip_ind] = self.num_edges
                    self.num_edges += 1

            epsilon_chi = model.addMVar(shape=self.num_edges+self.numReqs,
                                        vtype=GRB.BINARY, name="epsilon_chi")
            #chi = model.addMVar(shape=self.numReqs, vtype=GRP.BINARY, name="chi")

            row, col, val = [], [], []
            for i, veh_ind in enumerate(self.veh_inds):
                for trip_ind in self.veh_ind_trip_inds[veh_ind]:
                    row.append(i)
                    col.append(self.edge_dict[veh_ind,trip_ind])
                    val.append(1.)
            A = sp.csr_matrix((val, (row, col)), shape=(self.numVehicles, self.num_edges+self.numReqs))
            rhs_A = np.ones(self.numVehicles)
            model.addConstr(A @ epsilon_chi <= rhs_A, name="c1")

            row, col, val = [], [], []
            for req_ind in range(self.numReqs):
                for trip_ind in self.req_ind_trip_inds[req_ind]:
                    for _, veh_ind in self.trip_ind_cost_inds[trip_ind]:
                        row.append(req_ind)
                        col.append(self.edge_dict[veh_ind,trip_ind])
                        val.append(1.)
                row.append(req_ind)
                col.append(self.num_edges+req_ind)
                val.append(1.)
            B = sp.csr_matrix((val, (row, col)), shape=(self.numReqs, self.num_edges+self.numReqs))
            rhs_B = np.ones(self.numReqs)
            model.addConstr(B @ epsilon_chi == rhs_B, name="c2")
            # greedy assignment
            epsilon_chi.start = self.greedy_assignment()
            model.update()
            # build objective function
            objective = np.zeros(self.num_edges+self.numReqs)
            for trip_ind in range(self.numTrips):
                for cost, veh_ind in self.trip_ind_cost_inds[trip_ind]:
                    objective[self.edge_dict[veh_ind,trip_ind]] = cost
            for req_ind in range(self.numReqs):
                objective[self.num_edges+req_ind] = PENALTY

            model.setObjective(objective @ epsilon_chi, GRB.MINIMIZE)
            # solve
            #print("solve")
            model.optimize()
            #print("solved")
            solution = epsilon_chi.x
            # return trips
            for veh_ind in self.veh_inds:
                for trip_ind in self.veh_ind_trip_inds[veh_ind]:
                    if solution[self.edge_dict[veh_ind, trip_ind]] >= 0.9:
                        trip = self.trips[trip_ind]
                        self.actions[veh_ind] = [self.requests[req_ind] for req_ind in trip]
                        for req_ind in trip:
                            ROK.append(req_ind)
                        break
                else:
                    if len(self.vehicles[veh_ind].passengers) == 0:
                        self.VKO_idle.append(veh_ind)
                    else:
                        self.VKO_assigned.append(veh_ind)

        for req_ind in range(self.numReqs):
            if req_ind not in ROK:
                self.RKO.append(req_ind)

        if len(self.VKO_idle) > 0 and len(self.RKO) > 0:
            self.rebalance()
        elif len(self.VKO_idle) > 0:
            for i in range(len(self.VKO_idle)):
                veh_ind = self.VKO_idle[i]
                self.actions[veh_ind] = self.vehicles[veh_ind].get_location()

        for i in range(len(self.VKO_assigned)):
            veh_ind = self.VKO_assigned[i]
            self.actions[veh_ind] = []

        return self.actions
    def rebalance(self):
        model =  gp.Model("lp")
        model.setParam('OutputFlag', 0)

        y = model.addMVar(shape=len(self.VKO_idle)*len(self.RKO),
                                    vtype=GRB.BINARY, name="y")

        row, col, val, rhs_A = [], [], [], []
        objective = np.zeros(len(self.VKO_idle)*len(self.RKO))
        for i in range(len(self.VKO_idle)):
            for j in range(len(self.RKO)):
                start = self.vehicles[self.VKO_idle[i]].get_location()
                dest = self.requests[self.RKO[j]].start
                objective[i+len(self.VKO_idle)*j] = self.traffic.get_traveltime(start, dest)
                row.append(i)
                col.append(i+len(self.VKO_idle)*j)
                val.append(1)
            rhs_A.append(1)
        for j in range(len(self.RKO)):
            for i in range(len(self.VKO_idle)):
                row.append(len(self.VKO_idle)+j)
                col.append(i+len(self.VKO_idle)*j)
                val.append(1)
            rhs_A.append(1)
        A = sp.csr_matrix((val, (row, col)), shape=(len(self.VKO_idle)+len(self.RKO), len(self.VKO_idle)*len(self.RKO)))
        rhs_A = np.array(rhs_A)
        model.addConstr(A @ y <= rhs_A, name='c1')

        row, col, val, rhs_B = [], [], [], []
        for i in range(len(self.VKO_idle)):
            for j in range(len(self.RKO)):
                row.append(0)
                col.append(i+len(self.VKO_idle)*j)
                val.append(1)
        rhs_B.append(np.minimum(len(self.RKO),len(self.VKO_idle)))
        B = sp.csr_matrix((val, (row, col)), shape=(1, len(self.VKO_idle)*len(self.RKO)))
        rhs_B = np.array(rhs_B)
        model.addConstr(B @ y == rhs_B, name='c2')
        model.setObjective(objective @ y, GRB.MINIMIZE)
        model.optimize()

        solution = y.x
        for i in range(len(self.VKO_idle)):
            for j in range(len(self.RKO)):
                if solution[i+len(self.VKO_idle)*j] >= 0.9:
                    veh_ind = self.VKO_idle[i]
                    req = self.requests[j]
                    self.actions[veh_ind] = req.start
                    break
            else:
                veh_ind = self.VKO_idle[i]
                self.actions[veh_ind] = self.vehicles[veh_ind].get_location()


    def greedy_assignment(self):
        trip_size = MAX_CAPACITY
        initial_assignment = np.zeros(self.num_edges+self.numReqs)
        VOK, ROK = [], []
        while trip_size > 0:
            for trip_ind in self.trip_ind_cost_inds:
                trip = self.trips[trip_ind]
                if len(trip) == trip_size:
                    for req_ind in trip:
                        if req_ind in ROK:
                            break
                    else:
                        for cost, veh_ind in self.trip_ind_cost_inds[trip_ind]:
                            if veh_ind not in VOK:
                                initial_assignment[self.edge_dict[veh_ind,trip_ind]] = 1.0
                                VOK.append(veh_ind)
                                for req_ind in trip:
                                    ROK.append(req_ind)
                                break
            trip_size -= 1
            if len(VOK) == self.numVehicles:
                break
        for req_ind in range(self.numReqs):
            if req_ind not in ROK:
                initial_assignment[self.num_edges+req_ind] = 1.0
        return initial_assignment

    def sort_edges(self):
        for trip_ind in self.trip_ind_cost_inds:
            self.trip_ind_cost_inds[trip_ind].sort(key=lambda x: x[0])

    def add_edge_trip_vehicle(self, trip_ind, cost_veh_ind):
        self.trip_ind_cost_inds[trip_ind].append(cost_veh_ind)
        self.veh_ind_trip_inds [cost_veh_ind[1]].append(trip_ind)

    def get_trip_ind(self, trip):
        if trip in self.trip_inds:
            return self.trip_inds[trip]
        self.trip_inds[trip] = self.numTrips
        self.trips.append(trip)
        for req_ind in trip:
            self.req_ind_trip_inds[req_ind].add(self.numTrips)

        self.numTrips += 1
        return self.numTrips-1

def main():
    mdp = GridWorldRideSharing()
    while True:
        pass
        # get state
        # compute RV graph
        # compute RTV graph
        # rebalance
        # get next state
