import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from toy_ridesharing_main import *
from torch_geometric.data import Data, Dataset, Batch, DataLoader
from torch_scatter import scatter_mean, scatter_max, scatter_sum, scatter_softmax, scatter_log_softmax
import gc
from collections import deque
import os
from a2c_ppo_acktr import utils
from a2c_ppo_acktr.utils import init
from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.algo.kfac import KFACOptimizer

PENALTY = PENALTY + 10
# define RolloutBuffer
class MGRolloutBuffer(object):
    def __init__(self, num_steps, num_agents, batch_size=32):
        self.obs = [[] for _ in range(num_steps+1)]
        self.rewards = torch.zeros(num_steps, num_agents, 1).to("cuda:0")
        self.value_preds = torch.zeros(num_steps+1, num_agents, 1).to("cuda:0")
        self.returns = torch.zeros(num_steps + 1, num_agents, 1).to("cuda:0")
        self.action_log_probs = torch.zeros(num_steps, num_agents, 1).to("cuda:0")
        #if action_space.__class__.__name__ == 'Discrete':
        action_shape = 1
        #else:
        #    action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_agents, action_shape).to("cuda:0")
        self.masks = torch.ones(num_steps + 1, num_agents, 1).to("cuda:0")

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_agents, 1).to("cuda:0")

        self.num_agents = num_agents
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.step = 0
    def to(self, device):
        self.obs = [[self.obs[i][j].to(device) for j in range(self.num_agents)] for i in range(self.num_steps)]
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

    def insert(self, obs, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks):
        self.obs[self.step+1] = obs
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)

        self.step = (self.step + 1) % self.num_steps

    def feed_forward_dataloader(self):
        datas = []
        for step in range(self.num_steps):
            for agent_ind in range(self.num_agents):
                self.obs[step][agent_ind].returns = self.returns[step][agent_ind]
                datas.append(self.obs[step][agent_ind])
        dataloader = DataLoader(datas,
                                batch_size=self.batch_size,
                                shuffle=True,
                                drop_last=False,
                                follow_batch=['requests_x','vehicles_x','passengers_x' ])
        return dataloader

    def after_update(self):
        self.obs[0] = self.obs[-1]
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda,
                        use_proper_time_limits=True):

        if use_proper_time_limits:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[
                        step + 1] * self.masks[step +
                                               1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step +
                                                                  1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                        + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[
                        step + 1] * self.masks[step +
                                               1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step +
                                                                  1] * gae
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]

PAS_LOW = torch.tensor([0,  0,  0,  0,  0,             0,  0,  0,  0,            0], dtype=torch.float32)
PAS_HIG = torch.tensor([20, 20, 20, 20, MAX_WAIT_SEC, 40, 40, 40, MAX_WAIT_SEC, MAX_DELAY_SEC], dtype=torch.float32)

REQ_LOW = torch.tensor([0,  0,  0,  0,  0,            0,  0,            0,              0, 0], dtype=torch.float32)
REQ_HIG = torch.tensor([20, 20, 20, 20, MAX_WAIT_SEC, 40, MAX_WAIT_SEC, MAX_DELAY_SEC, 1, 1], dtype=torch.float32)

VEH_LOW = torch.tensor([0, 0, 0, 0 , 0, 0, 0, 0], dtype=torch.float32)
VEH_HIG = torch.tensor([1, 1, 1, 1, 20, 20, 4, 4], dtype=torch.float32)

REQ2VEH_LOW = torch.tensor([0, 0], dtype=torch.float32)
REQ2VEH_HIG = torch.tensor([MAX_WAIT_SEC+MAX_DELAY_SEC, 1], dtype=torch.float32)

REQ2REQ_LOW = torch.tensor([0], dtype=torch.float32)
REQ2REQ_HIG = torch.tensor([MAX_WAIT_SEC+MAX_DELAY_SEC], dtype=torch.float32)


def log_passenger(pas_list, pas, cur_step, wait_tolerance=MAX_WAIT_SEC, delay_tolerance=MAX_DELAY_SEC):
    pas_list.append([pas.start[0],
                     pas.start[1],
                     pas.dest[0],
                     pas.dest[1],
                     pas.scheduledOnTime - pas.reqTime,
                     pas.expectedOffTime - pas.reqTime,
                     cur_step - pas.scheduledOnTime,
                     pas.scheduledOffTime - pas.scheduledOnTime,
                     wait_tolerance,
                     delay_tolerance])

def log_request(req_list, req, cur_step, centrality, wait_tolerance=MAX_WAIT_SEC, delay_tolerance=MAX_DELAY_SEC):
    req_list.append([req.start[0],
                     req.start[1],
                     req.dest[0],
                     req.dest[1],
                     cur_step - req.reqTime,
                     req.expectedOffTime - req.reqTime,
                     wait_tolerance,
                     delay_tolerance,
                     centrality,
                     req.assign])

def log_vehicle(veh_feat, veh, ego):
    veh_feat.append([ego, ego, ego, ego, veh.location[0], veh.location[1], len(veh.passengers), MAX_CAPACITY])

def make_rebalance_requests(location, time):
    #left
    left_req = Request(-1, (location[0], location[1]),
                           (location[0]-1, location[1]),
                           time)
    left_req.expectedOffTime = left_req.reqTime+1
    #right
    right_req = Request(-1, (location[0], location[1]),
                           (location[0]+1, location[1]),
                           time)
    right_req.expectedOffTime =  right_req.reqTime+1
    #up
    up_req = Request(-1, (location[0], location[1]),
                           (location[0], location[1]+1),
                           time)
    up_req.expectedOffTime =  up_req.reqTime+1
    #down
    down_req = Request(-1, (location[0], location[1]),
                           (location[0], location[1]-1),
                           time)
    down_req.expectedOffTime =  down_req.reqTime+1
    #stay
    stay_req = Request(-1, (location[0], location[1]),
                           (location[0], location[1]),
                           time)
    stay_req.expectedOffTime =  stay_req.reqTime
    return [left_req,right_req,up_req,down_req,stay_req]

def make_assign_request(start, dest, time):
    virtual_req = Request(-1, start, dest, time)
    virtual_req.expectedOffTime =  virtual_req.reqTime+1
    return virtual_req

def make_virtual_passenger(unique, location, time):
    virtual_pas = Request(unique, location, location, time)
    virtual_pas.onBoard = True
    virtual_pas.scheduledOnTime = time
    virtual_pas.scheduledOffTime = time
    virtual_pas.expectedOffTime = time
    return virtual_pas

def get_info(env):
    return [{} for _ in env.vehicles]

def get_done(env):
    done = False
    if env.cur_step > 50 and len(env.requests) == 0:
        Complete = True
        for veh in env.vehicles:
            if len(veh.passengers) != 0:
                Complete = False
                break
        if Complete:
            done = True
    if done:
        env.reset()
    return env.current_state, [done for _ in env.vehicles]

def get_reward_cluster(env, all_graph_s, all_graph_a, device="cuda:0"):
    rs = [[0.] for _ in range(len(all_graph_s))]
    for veh_ind in range(len(all_graph_s)):
        veh = env.vehicles[veh_ind]
        loc = veh.get_location()
        total_dist = 0
        for nei_ind in all_graph_s[veh].vehicle_inds:
            if nei_ind != veh_ind:
                nei = env.vehicles[nei_ind]
                total_dist += np.abs(loc[0] - nei.location[0]) + np.abs(loc[1] - nei.location[1])
        rs[veh_ind][0] = np.exp(-total_dist)
    return torch.tensor(rs, dtype=torch.float).to(device)

def cost(env, act):
    all_cost = 0.0
    req_uniques = []
    for veh, a in zip(env.vehicles, act):
        if type(a) == list:
            if len(a) > 0:
                req_uniques.append(a[0].unique)
            c = travel(veh, env.cur_step, a, env.traffic, False)
            assert(c != -1)
            all_cost += c
    for req in env.requests:
        if req.unique not in req_uniques:
            all_cost += MAX_WAIT_SEC + MAX_DELAY_SEC + 1
    return req_uniques, all_cost

def alt_action(env, all_graph_s, all_graph_a, veh_ind, alt_req_unique):
    alt_graph_a = all_graph_a.detach().clone()
    req_uniques = all_graph_s[veh_ind].request_uniques.cpu().numpy().astype(int)
    if alt_req_unique in req_uniques:
        index = np.where(req_uniques==alt_req_unique)[0][0]
    else:
        index = len(req_uniques)-1
    alt_graph_a[veh_ind][0] = index
    return alt_graph_a

def get_request_reward(env, all_graph_s, all_graph_a, device="cuda:0"):
    rs = [[0.] for _ in range(len(all_graph_s))]
    # global request rewards
    env_acts = get_env_action(env, all_graph_s, all_graph_a, decided=False)

    req_uniques, all_cost = cost(env, env_acts)

    for veh_ind in range(len(all_graph_s)):
        veh = env.vehicles[veh_ind]
        a = env_acts[veh_ind]
        if len(veh.schedulepassengers) == 1:
            sch_req = veh.schedulepassengers[0]
            non_acts = alt_action(env, all_graph_s, all_graph_a, veh_ind, sch_req.unique)
            non_env_acts = get_env_action(env, all_graph_s, non_acts, decided=False)
            _, non_cost = cost(env, non_env_acts)
            rs[veh_ind][0] = non_cost - all_cost
        else:
            if len(veh.passengers) > 0:
                non_acts = alt_action(env, all_graph_s, all_graph_a, veh_ind, -10)
                non_env_acts = get_env_action(env, all_graph_s, non_acts, decided=False)
                _, non_cost = cost(env, non_env_acts)
                rs[veh_ind][0] = non_cost - all_cost
            else:
                non_acts = alt_action(env, all_graph_s, all_graph_a, veh_ind, -5)
                non_env_acts = get_env_action(env, all_graph_s, non_acts, decided=False)
                _, non_cost = cost(env, non_env_acts)
                rs[veh_ind][0] = non_cost - all_cost
    return torch.tensor(rs, dtype=torch.float).to(device)/15.


def get_passenger_reward(env, all_graph_s, all_graph_a, device="cuda:0"):
    rs = [[0.] for _ in range(len(all_graph_s))]
    for veh_ind in range(len(all_graph_s)):
        veh = env.vehicles[veh_ind]
        rs[veh_ind][0] = len(veh.passengers)/4.0
    return torch.tensor(rs, dtype=torch.float).to(device)

def get_centrality_reward(env, all_graph_s, all_graph_a, device="cuda:0"):
    rs = [[0.] for _ in range(len(env.vehicles))]
    for veh, s, a, r in zip(env.vehicles, all_graph_s, all_graph_a, rs):
        req_ind = s.request_inds[a]
        if req_ind < 0 and req_ind > -10:
            req_feat = s.requests_x[a].squeeze()
            cen = req_feat[-2].cpu().item() + 0.5
            r[0] = cen

    return torch.tensor(rs, dtype=torch.float).to(device)


'''
def get_reward_center(env, all_graph_s, all_graph_a, device="cuda:0"):
    rs = [[0.] for _ in range(len(all_graph_s))]
    for veh_ind in range(len(all_graph_s)):
        veh = env.vehicles[veh_ind]
        loc = veh.get_location()
        a = all_graph_a[veh_ind][0]
        req_ind = a
        #req_ind = all_graph_s[veh_ind].request_inds[a]
        #if req_ind > 0:
        #    rs[veh_ind][0] = -1.
        #    continue
        if req_ind == 0:#-1:
            new_loc = (max(loc[0]-1,0), loc[1])
        elif req_ind == 1:#-2:
            new_loc = (min(veh.location[0]+1, env.width-1), loc[1])
        elif req_ind == 2:#-3:
            new_loc = (veh.location[0], min(veh.location[1]+1,env.height-1))
        elif req_ind == 3:#-4:
            new_loc = (veh.location[0], max(veh.location[1]-1,0))
        else:
            new_loc = (veh.location[0], veh.location[1])
        if new_loc[0] >= 8 and new_loc[0] <=12 and new_loc[1] >=8 and new_loc[1] <=12:
            rs[veh_ind][0] = 1.
    return torch.tensor(rs, dtype=torch.float).to(device)

def get_reward_center(env, all_graph_s, all_graph_a, device="cuda:0"):
    rs = [[0.] for _ in range(len(all_graph_s))]
    for veh_ind in range(len(all_graph_s)):
        veh = env.vehicles[veh_ind]
        loc = veh.get_location()
        a = all_graph_a[veh_ind][0]
        req_ind = a
        req_ind = all_graph_s[veh_ind].request_inds[a]
        req = all_graph_s[veh_ind].requests_x[a]
        if req_ind >= 0:
            rs[veh_ind][0] = -1.
            continue
        if req_ind == -1:
            #print(req[2] - req[0])
            assert req[2] - req[0] < 0
            assert req[1] == req[3]
            new_loc = (max(loc[0]-1,0), loc[1])
        elif req_ind == -2:
            #print(req[2] - req[0])
            assert req[2] - req[0] > 0
            assert req[1] == req[3]
            new_loc = (min(loc[0]+1, env.width-1), loc[1])
        elif req_ind == -3:
            #print(req[3] - req[1])
            assert req[3] - req[1] > 0
            assert req[2] == req[0]
            new_loc = (loc[0], min(loc[1]+1,env.height-1))
        elif req_ind == -4:
            #print(req[3] - req[1])
            assert req[3] - req[1] < 0
            assert req[2] == req[0]
            new_loc = (loc[0], max(loc[1]-1,0))
        else:
            new_loc = (loc[0], loc[1])
        if new_loc[0] >= 8 and new_loc[0] <=12 and new_loc[1] >=8 and new_loc[1] <=12:
            rs[veh_ind][0] = 1.
    return torch.tensor(rs, dtype=torch.float).to(device)
'''

'''
def get_reward_right(env, all_graph_s, all_graph_a, device="cuda:0"):
    rs = [[0.] for _ in range(len(all_graph_s))]
    for veh_ind in range(len(all_graph_s)):
        veh = env.vehicles[veh_ind]
        loc = veh.get_location()
        a = all_graph_a[veh_ind][0]
        req_ind = a
        req_ind = all_graph_s[veh_ind].request_inds[a]
        req = all_graph_s[veh_ind].requests_x[a]
        if req_ind >= 0:
            rs[veh_ind][0] = -1.
            continue
        if req_ind == -1:
            #print(req[2] - req[0])
            assert req[2] - req[0] < 0
            assert req[1] == req[3]
            new_loc = (max(loc[0]-1,0), loc[1])
        elif req_ind == -2:
            #print(req[2] - req[0])
            assert req[2] - req[0] > 0
            assert req[1] == req[3]
            new_loc = (min(loc[0]+1, env.width-1), loc[1])
        elif req_ind == -3:
            #print(req[3] - req[1])
            assert req[3] - req[1] > 0
            assert req[2] == req[0]
            new_loc = (loc[0], min(loc[1]+1,env.height-1))
        elif req_ind == -4:
            #print(req[3] - req[1])
            assert req[3] - req[1] < 0
            assert req[2] == req[0]
            new_loc = (loc[0], max(loc[1]-1,0))
        else:
            new_loc = (loc[0], loc[1])
        if new_loc[0] - loc[0] < 0:
            rs[veh_ind][0] = 1.
    return torch.tensor(rs, dtype=torch.float).to(device)
'''

'''
def get_reward_right(env, all_graph_s, all_graph_a, device="cuda:0"):
    rs = [[0.] for _ in range(len(all_graph_s))]
    for veh_ind in range(len(all_graph_s)):
        veh = env.vehicles[veh_ind]
        loc = veh.get_location()
        a = all_graph_a[veh_ind][0]
        req_ind = a
        #req_ind = all_graph_s[veh_ind].request_inds[a]
        #if req_ind > 0:
        #    rs[veh_ind][0] = -1.
        #    continue
        if req_ind == 0:#-1:
            new_loc = (max(loc[0]-1,0), loc[1])
        elif req_ind == 1:#-2:
            new_loc = (min(loc[0]+1, env.width-1), loc[1])
        elif req_ind == 2:#-3:
            new_loc = (loc[0], min(loc[1]+1,env.height-1))
        elif req_ind == 3:#-4:
            new_loc = (loc[0], max(loc[1]-1,0))
        else:
            new_loc = (loc[0], loc[1])
        if new_loc[1] - loc[1] < 0:
            rs[veh_ind][0] = 1.
    return torch.tensor(rs, dtype=torch.float).to(device)
'''
#def get_reward_left(env, all_graph_s, all_graph_a, device="cuda:0"):
#    rs = [[0.] for _ in range(len(all_graph_s))]
#    for veh_ind in range(len(all_graph_s)):
#        veh = env.vehicles[veh_ind]
#        loc = veh.get_location()
#        if loc[0] == 0:
#            rs[veh_ind][0] = 1.
#    return torch.tensor(rs, dtype=torch.float).to(device)

#def get_reward_left2(env, all_graph_s, all_graph_a, device="cuda:0"):
#    rs = [[0.] for _ in range(len(all_graph_s))]
#    for veh_ind in range(len(all_graph_s)):
#        veh = env.vehicles[veh_ind]
#        a = all_graph_a[veh_ind][0]
#        req_ind = all_graph_s[veh_ind].request_inds[a]
#        if req_ind == -1:
#            rs[veh_ind][0] = 1.
#    return torch.tensor(rs, dtype=torch.float).to(device)

'''
def get_reward(env, all_graph_s, all_graph_a_rl, device="cuda:0"):
    env_action = get_env_action(env, all_graph_s, all_graph_a_rl)
    all_graph_a = convert_expert_action(env, all_graph_s, env_action)
    rs = [[0.] for _ in range(len(all_graph_s))] # np.zeros(len(all_graph_s), 1)
    for veh_ind in range(len(all_graph_s)):
        veh = env.vehicles[veh_ind]
        a = all_graph_a[veh_ind][0]
        a_rl = all_graph_a_rl[veh_ind][0]
        if a != a_rl:
            #print("not same: ", all_graph_s[veh_ind].request_uniques[a], all_graph_s[veh_ind].request_uniques[a_rl])
            rs[veh_ind][0] = 0
            continue
        #else:
            #print("same: ", all_graph_s[veh_ind].request_uniques[a], all_graph_s[veh_ind].request_uniques[a_rl])

        req_ind = all_graph_s[veh_ind].request_inds[a]
        req_unique = all_graph_s[veh_ind].request_uniques[a]

        if req_ind >= 0:
            #actual req
            req_new = env.requests[req_ind]
            if len(veh.schedulepassengers) != 0:
                # has current schedule
                if req_unique == veh.schedulepassengers[0].unique: #TODO we assume there is only one scheduled req
                    # current schedule == new schedule
                    rs[veh_ind][0] = 1.
                else:
                    # current schedule != new schedule
                    for nei_ind in all_graph_s[veh_ind].vehicle_inds.cpu().numpy().astype(int):
                        if nei_ind != veh_ind:
                            nei = env.vehicles[nei_ind]
                            if len(nei.schedulepassengers) >0 and nei.schedulepassengers[0].unique == req_unique:
                                # new schedule conflict with another veh schedule
                                # determine who will get the request first
                                cost_sch = travel(veh, env.cur_step, veh.schedulepassengers, env.traffic, False)
                                assert(cost_sch != -1)
                                cost_sch_nei = travel(nei, env.cur_step, nei.schedulepassengers, env.traffic, False)
                                assert(cost_sch_nei != -1)
                                cost_new = travel(veh, env.cur_step, [req_new], env.traffic, False)
                                assert(cost_new != -1)
                                if cost_new < cost_sch_nei:
                                    cost_new_nei = travel(nei, env.cur_step, [], env.traffic, False)
                                    rs[veh_ind][0] = cost_sch + cost_sch_nei - (cost_new + cost_new_nei + PENALTY)
                                else:
                                    cost_new = travel(veh, env.cur_step, [], env.traffic, False)
                                    rs[veh_ind][0] = cost_sch + cost_sch_nei - (cost_sch_nei + cost_new + PENALTY)
                                break
                    else:
                        # new schedule is unassigned
                        cost_sch = travel(veh, env.cur_step, veh.schedulepassengers, env.traffic, False)
                        assert(cost_sch != -1)
                        cost_new = travel(veh, env.cur_step, [req_new], env.traffic, False)
                        assert(cost_new != -1)
                        rs[veh_ind][0] = cost_sch - cost_new
            else:
                # no current schedule
                for nei_ind in all_graph_s[veh_ind].vehicle_inds.cpu().numpy().astype(int):
                    if nei_ind != veh_ind:
                        nei = env.vehicles[nei_ind]
                        if len(nei.schedulepassengers) >0 and nei.schedulepassengers[0].unique == req_unique:
                            # new schedule conflict with another veh schedule
                            # determine who will get the request first
                            cost_sch = travel(veh, env.cur_step, [], env.traffic, False)
                            cost_sch_nei = travel(nei, env.cur_step, nei.schedulepassengers, env.traffic, False)
                            assert(cost_sch_nei != -1)
                            cost_new = travel(veh, env.cur_step, [req_new], env.traffic, False)
                            assert(cost_new != -1)
                            if cost_new < cost_sch_nei:
                                cost_new_nei = travel(nei, env.cur_step, [], env.traffic, False)
                                rs[veh_ind][0] = cost_sch + cost_sch_nei - (cost_new + cost_new_nei)
                            else:
                                #if len(veh.passengers) == 0 and all_graph_s[veh_ind].request_inds[0] > 0:
                                #    rs[veh_ind][0] = -1
                                #else:
                                #    rs[veh_ind][0] = 0
                                rs[veh_ind][0] = 0
                            break
                else:
                    # new schedule is unassigned
                    cost_new = travel(veh, env.cur_step, [req_new], env.traffic, False)
                    cost_sch = travel(veh, env.cur_step, [], env.traffic, False)
                    assert(cost_new != -1)
                    rs[veh_ind][0] = PENALTY + cost_sch - cost_new
        elif req_ind == -10:
            # pas dropping
            if len(veh.schedulepassengers) != 0:
                # has current schedule
                cost_sch = travel(veh, env.cur_step, veh.schedulepassengers, env.traffic, False)
                cost_new = travel(veh, env.cur_step, [], env.traffic, False)
                rs[veh_ind][0] = cost_sch - (cost_new + PENALTY)
            else:
                # no current schedule
                rs[veh_ind][0] = 1.
        else:
            # rebalance
            if len(veh.schedulepassengers) != 0:
                # has current schedule
                cost_sch = travel(veh, env.cur_step, veh.schedulepassengers, env.traffic, False)
                rs[veh_ind][0] = cost_sch - PENALTY
            else:
                # no current schedule
                rs[veh_ind][0] = 0
                #if all_graph_s[veh_ind].request_inds[0] > 0:
                #    rs[veh_ind][0] = -1.
                #else:
                #    rs[veh_ind][0] = 0.
    return torch.tensor(rs, dtype=torch.float).to(device)/10.
'''
def convert_expert_action(env, all_graph_s, all_expert_a, device="cuda:0"):
    actions = []
    for veh_ind in range(len(all_graph_s)):
        veh = env.vehicles[veh_ind]
        req_feats = all_graph_s[veh_ind].requests_x.cpu().numpy()
        req_uniques = all_graph_s[veh_ind].request_uniques.cpu().numpy().astype(int)
        a = all_expert_a[veh_ind]
        if type(a) == list:
            if len(a) > 0:
                req = travel(veh, env.cur_step, a, env.traffic, False, first_req=True)
                actions.append([np.where(req_uniques==req.unique)[0][0]])
                #print(req_feats[np.where(req_uniques==req.unique)[0][0]])
                #print(req, "first trip")
            else:
                #print(req_feats[-1])
                #print(veh.scheduleroute[0][1], "assign")
                actions.append([len(req_feats)-1])
        else:
            path = env.traffic.find_path(veh.location, a)
            if len(path) == 1:
                #print(req_feats[-1])
                #print(veh.location, "stay")
                actions.append([len(req_feats)-1])
            else:
                if path[1][0] == path[0][0]-1:
                    #print(req_feats[-5])
                    #print(path[1], "left")
                    actions.append([len(req_feats)-5])
                elif path[1][0] == path[0][0]+1:
                    #print(req_feats[-4])
                    #print(path[1], "right")
                    actions.append([len(req_feats)-4])
                elif path[1][1] == path[0][1]+1:
                    #print(req_feats[-3])
                    #print(path[1], "up")
                    actions.append([len(req_feats)-3])
                else:
                    #print(req_feats[-2])
                    #print(path[1], "down")
                    actions.append([len(req_feats)-2])
    return torch.tensor(actions, dtype=torch.long).to(device)


def get_env_action(env, all_graph_s, all_graph_a, decided=True):
    expert_actions = []
    requ_veh = {}
    veh_reqi = {}
    for veh_ind in range(len(all_graph_s)):
        veh = env.vehicles[veh_ind]
        req_inds = all_graph_s[veh_ind].request_inds.cpu().numpy().astype(int)
        a = all_graph_a[veh_ind][0]
        req_ind = req_inds[a]
        if req_ind >= 0:
            req = env.requests[req_ind]
            if req.unique not in requ_veh:
                # not conflict\
                cost = travel(veh, env.cur_step, [req], env.traffic, False)
                assert(cost != -1)
                requ_veh[req.unique] = (veh_ind, cost)
                veh_reqi[veh_ind] = req_ind
            else:
                # conflict
                cost = travel(veh, env.cur_step, [req], env.traffic, False)
                assert(cost != -1)
                if cost < requ_veh[req.unique][1]:
                    veh_reqi[veh_ind] = req_ind
                    if len(env.vehicles[requ_veh[req.unique][0]].passengers) > 0:
                        veh_reqi[requ_veh[req.unique][0]] = -10
                    else:
                        veh_reqi[requ_veh[req.unique][0]] = -5
                    requ_veh[req.unique] = (veh_ind, cost)
                else:
                    if len(veh.passengers) > 0:
                        veh_reqi[veh_ind] = -10
                    else:
                        veh_reqi[veh_ind] = -5
        else:
            veh_reqi[veh_ind] = req_ind
    # final assignments
    for veh_ind in range(len(all_graph_s)):
        veh = env.vehicles[veh_ind]
        req_ind = veh_reqi[veh_ind]
        if req_ind >= 0:
            req = env.requests[req_ind]
            if len(veh.schedulepassengers) > 0 and veh.schedulepassengers[0].unique == req.unique:
                # sch == new
                sch_req = veh.schedulepassengers[0]
                #print("sch", sch_req.unique)
                if decided:
                    assert(req.assign > 0)
                expert_actions.append([req])
            elif len(veh.schedulepassengers) > 0:
                # has schedule req and not same as new
                sch_req = veh.schedulepassengers[0]
                #print("sch", sch_req.unique)
                if sch_req.unique in requ_veh:
                    # schedule switched to others
                    expert_actions.append([req])
                    if decided:
                        assert(sch_req.assign!=2)#
                        sch_req.assign = 2
                        if req.assign == 0:
                            req.assign = 1
                else:
                    # schedule abandoned
                    expert_actions.append([req])
                    if decided:
                        assert(sch_req.assign!=2)
                        sch_req.assign = 0
                        sch_req.scheduledOnTime = -1
                        sch_req.scheduledOffTime = -1
                        if req.assign == 0:
                            req.assign = 1
            else:
                # no schedule
                expert_actions.append([req])
                if decided:
                    if req.assign == 0:
                        req.assign = 1

        elif req_ind == -10:
            if len(veh.schedulepassengers) > 0:
                sch_req = veh.schedulepassengers[0]
                #print("sch", sch_req.unique)
                if decided:
                    assert(sch_req.assign != 2)
                if sch_req.unique in requ_veh:
                    # schedule switched to others
                    expert_actions.append([])
                    if decided:
                        sch_req.assign = 2
                else:
                    # back to that request
                    expert_actions.append([sch_req])
            else:
                expert_actions.append([])
        else:
            if len(veh.schedulepassengers) > 0:
                sch_req = veh.schedulepassengers[0]
                if decided:
                    assert(sch_req.assign != 2)
                if sch_req.unique in requ_veh:
                    # schedule switched to others
                    if req_ind == -1:
                        expert_actions.append((max(veh.location[0]-1,0), veh.location[1]))
                    elif req_ind == -2:
                        expert_actions.append((min(veh.location[0]+1, env.width-1), veh.location[1]))
                    elif req_ind == -3:
                        expert_actions.append((veh.location[0], min(veh.location[1]+1,env.height-1)))
                    elif req_ind == -4:
                        expert_actions.append((veh.location[0], max(veh.location[1]-1,0)))
                    else:
                        expert_actions.append((veh.location[0], veh.location[1]))
                    if decided:
                        sch_req.assign = 2
                else:
                    # back to that request
                    expert_actions.append([sch_req])
            else:
                if req_ind == -1:
                    expert_actions.append((max(veh.location[0]-1,0), veh.location[1]))
                elif req_ind == -2:
                    expert_actions.append((min(veh.location[0]+1, env.width-1), veh.location[1]))
                elif req_ind == -3:
                    expert_actions.append((veh.location[0], min(veh.location[1]+1,env.height-1)))
                elif req_ind == -4:
                    expert_actions.append((veh.location[0], max(veh.location[1]-1,0)))
                else:
                    expert_actions.append((veh.location[0], veh.location[1]))

    return expert_actions

def get_mean_action(all_graph_s, all_graph_a):
    # compute the additional information
    # gather the request that each agent plan
    veh_feat_dict = {}
    for veh_ind in range(len(all_graph_s)):
        req_feats = all_graph_s[veh_ind].requests_x
        a = all_graph_a[veh_ind][0]
        veh_feat_dict[veh_ind] = req_feats[a]
    mean_actions = []
    for veh_ind in range(len(all_graph_s)):
        veh_inds = all_graph_s[veh_ind].vehicle_inds.cpu().numpy().astype(int)
        mean_action = []
        for nei_ind in veh_inds:
            mean_action.append(veh_feat_dict[nei_ind])
        mean_action = torch.stack(mean_action)
        mean_actions.append(mean_action)
    return mean_actions



#def log_vehicle(veh_feat, veh, ego, step):
#    veh_feat.append([step, ego, veh.location[0], veh.location[1], len(veh.passengers)])

def compute_centrality(env, req, rebalance=False):
    #
    if rebalance:
        start = req.start
        dest = req.dest
        if dest[0] > start[0]:
            #right
            cen = 0
            for i in range(1, env.width):
                N = 0
                dist = 1
                if start[0]+i >= env.width:
                    break
                for req in env.requests:
                    if req.assign <= 1:
                        N += 1
                        dist += env.traffic.get_traveltime((start[0]+i, start[1]), req.start)
                if cen < N/dist:
                    cen = N/dist
        elif dest[0] < start[0]:
            #left
            cen = 0
            for i in range(1, env.width):
                N = 0
                dist = 1
                if start[0]-i < 0:
                    break
                for req in env.requests:
                    if req.assign <= 1:
                        N += 1
                        dist += env.traffic.get_traveltime((start[0]-i, start[1]), req.start)
                if cen < N/dist:
                    cen = N/dist
        elif dest[1] > start[0]:
            #up
            cen = 0
            for i in range(1, env.height):
                N = 0
                dist = 1
                if start[1]+i >= env.height:
                    break
                for req in env.requests:
                    if req.assign <= 1:
                        N += 1
                        dist += env.traffic.get_traveltime((start[0], start[1]+i), req.start)
                if cen < N/dist:
                    cen = N/dist
        elif dest[1] < start[1]:
            #down
            cen = 0
            for i in range(1, env.height):
                N = 0
                dist = 1
                if start[1]-i < 0:
                    break
                for req in env.requests:
                    if req.assign <= 1:
                        N += 1
                        dist += env.traffic.get_traveltime((start[0], start[1]-i), req.start)
                if cen < N/dist:
                    cen = N/dist
        else:
            #stay
            N = 0
            dist = 1
            for req in env.requests:
                if req.assign <= 1:
                    N += 1
                    dist += env.traffic.get_traveltime(start, req.start)
            cen = N/dist
    else:
        dest = req.dest
        dist = 1
        N = 0
        for req in env.requests:
            if req.assign <= 1:
                N += 1
                dist += env.traffic.get_traveltime(dest, req.start)
        cen = N/dist
    return cen


def get_state(env, s, device="cuda:0"):
    rvgraph = RVGraph(env.cur_step, s.vehicles, s.requests, env.traffic)
    agents = []
    # seperate each agent
    for veh_ind in range(len(s.vehicles)):
        uniques = 0
        veh = s.vehicles[veh_ind]
        req_feat, req_uniques, req_inds = [], [], []
        veh_feat = []
        log_vehicle(veh_feat, veh, 1)
        #veh_feat = [[env.cur_step, 1, 1, 1, 1, veh.location[0], veh.location[1], len(veh.passengers)]]
        veh_inds = [veh_ind]
        pas_feat = []
        veh_unique_inds = {veh_ind:0}
        req_unique_inds = {}
        pas_unique_inds = {}
        req2req_sender, req2req_receiver, req2req_edge_attr = [], [], []
        req2veh_sender, req2veh_receiver, req2veh_edge_attr = [], [], []
        veh2pas_sender, veh2pas_receiver = [], []

        for pas in veh.passengers:
            assert(pas.unique not in pas_unique_inds)
            pas_unique_inds[pas.unique] = len(pas_feat)
            log_passenger(pas_feat, pas, env.cur_step)
            veh2pas_sender.append(veh_unique_inds[veh_ind])
            veh2pas_receiver.append(pas_unique_inds[pas.unique])

        if len(veh.passengers) == 0:
            # add a virtual passenger
            virtual_pas = make_virtual_passenger(env.uniques+uniques, veh.get_location(), env.cur_step)
            pas_unique_inds[virtual_pas.unique] = len(pas_feat)
            log_passenger(pas_feat, virtual_pas, env.cur_step, wait_tolerance=0, delay_tolerance=0)
            veh2pas_sender.append(veh_unique_inds[veh_ind])
            veh2pas_receiver.append(pas_unique_inds[virtual_pas.unique])
            uniques += 1

        if rvgraph.has_vehicle(veh_ind):
            # vehicle has active requests nearby
            if len(veh.passengers) < MAX_CAPACITY:
                # there is capability
                # log the requests
                if len(veh.schedulepassengers) > 0 and veh.schedulepassengers[0].assign == 2:
                    # the scheduled request has assign status 2
                    # it must be pickup by this vehicle
                    sch_req = veh.schedulepassengers[0]
                    for req_ind, cost in rvgraph.get_vehicle_edges(veh_ind):
                        req = s.requests[req_ind]
                        if req.unique == sch_req.unique:
                            sch_req_ind = req_ind
                            sch_cost = cost
                            break
                    assert(sum([r.unique == req.unique for r in s.requests]) > 0)
                    req_unique_inds[sch_req_ind] = len(req_feat)
                    cen = compute_centrality(env, sch_req, rebalance=False)
                    log_request(req_feat, sch_req, env.cur_step, cen)
                    req_uniques.append(sch_req.unique)
                    req_inds.append(sch_req_ind)
                    #log req self edge
                    req2req_sender.append(req_unique_inds[sch_req_ind])
                    req2req_receiver.append(req_unique_inds[sch_req_ind])
                    req2req_edge_attr.append([0])
                    #log req to veh edge
                    req2veh_sender.append(req_unique_inds[sch_req_ind])
                    req2veh_receiver.append(veh_unique_inds[veh_ind])
                    req2veh_edge_attr.append([sch_cost, 1.])

                else:
                    #
                    for req_ind, cost in rvgraph.get_vehicle_edges(veh_ind):
                        # ignore all requests with status 2
                        req = s.requests[req_ind]
                        if req.assign == 2:
                            continue
                        assert(req_ind not in req_unique_inds)
                        req_unique_inds[req_ind] = len(req_feat)
                        cen = compute_centrality(env, req, rebalance=False)
                        log_request(req_feat, req, env.cur_step, cen)
                        req_uniques.append(req.unique)
                        req_inds.append(req_ind)
                        #log req self edge
                        req2req_sender.append(req_unique_inds[req_ind])
                        req2req_receiver.append(req_unique_inds[req_ind])
                        req2req_edge_attr.append([0])
                        #log req to veh edge
                        req2veh_sender.append(req_unique_inds[req_ind])
                        req2veh_receiver.append(veh_unique_inds[veh_ind])
                        if len(veh.schedulepassengers) > 0 and req.unique == veh.schedulepassengers[0].unique:
                            assert(veh.schedulepassengers[0].assign==1)
                            req2veh_edge_attr.append([cost, 1.])
                        else:
                            req2veh_edge_attr.append([cost, 0.])

                        # log the nearby vehicles
                        for nei_ind, cost in rvgraph.req_cost_car[req_ind]:
                            nei = s.vehicles[nei_ind]
                            if len(nei.schedulepassengers) > 0 and nei.schedulepassengers[0].assign == 2:
                                continue
                            if nei_ind not in veh_unique_inds and len(nei.passengers) < MAX_CAPACITY:
                                veh_unique_inds[nei_ind] = len(veh_feat)
                                log_vehicle(veh_feat, nei, 0)
                                #veh_feat.append([env.cur_step, 0, 0, 0, 0, nei.location[0], nei.location[1], len(nei.passengers)])
                                veh_inds.append(nei_ind)
                                # log passengers
                                for pas in nei.passengers:
                                    assert(pas.unique not in pas_unique_inds)
                                    pas_unique_inds[pas.unique] = len(pas_feat)
                                    log_passenger(pas_feat, pas, env.cur_step)
                                    # log vehicle to passenger edges
                                    veh2pas_sender.append(veh_unique_inds[nei_ind])
                                    veh2pas_receiver.append(pas_unique_inds[pas.unique])
                                # add virtual passenger
                                if len(veh.passengers) == 0:
                                    virtual_pas = make_virtual_passenger(env.uniques+uniques, nei.get_location(), env.cur_step)
                                    pas_unique_inds[virtual_pas.unique] = len(pas_feat)
                                    log_passenger(pas_feat, virtual_pas, env.cur_step, wait_tolerance=0, delay_tolerance=0)
                                    veh2pas_sender.append(veh_unique_inds[nei_ind])
                                    veh2pas_receiver.append(pas_unique_inds[virtual_pas.unique])
                                    uniques += 1

                                # log request to nei vehicle edges
                                req2veh_sender.append(req_unique_inds[req_ind])
                                req2veh_receiver.append(veh_unique_inds[nei_ind])
                                if len(nei.schedulepassengers) > 0 and req.unique == nei.schedulepassengers[0].unique:
                                    assert(nei.schedulepassengers[0].assign==1)
                                    req2veh_edge_attr.append([cost, 1.])
                                else:
                                    req2veh_edge_attr.append([cost, 0.])

                    # log request to request edges
                    for req1_ind, req2_ind, cost in rvgraph.req_inter:
                        if req1_ind in req_unique_inds and req2_ind in req_unique_inds:
                            req1 = s.requests[req1_ind]
                            req2 = s.requests[req2_ind]
                            virtualCar = Vehicle(-1, req1.start)
                            cost = travel(virtualCar, env.cur_step, [req1,req2], env.traffic, False)
                            if cost >= 0:
                                req2req_sender.append(req_unique_inds[req1_ind])
                                req2req_receiver.append(req_unique_inds[req2_ind])
                                req2req_edge_attr.append([cost])

                            virtualCar.set_location(req2.start)
                            cost = travel(virtualCar, env.cur_step, [req2,req1], env.traffic, False)
                            if cost >=0:
                                req2req_sender.append(req_unique_inds[req2_ind])
                                req2req_receiver.append(req_unique_inds[req1_ind])
                                req2req_edge_attr.append([cost])

        else:
            # vehicle has no active requests nearby
            # perhaps it should be told where the nearest requests are
            #assert(len(veh.schedulepassengers) == 0)
            pass
        # add rebalance options
        if len(veh.schedulepassengers) == 0 or (len(veh.schedulepassengers) == 1 and veh.schedulepassengers[0].assign == 1):
            if len(veh.passengers) == 0:
            # add five rebalance virtual requests
                virtual_reqs = make_rebalance_requests(veh.get_location(), env.cur_step)
                for i, vir_req in enumerate(virtual_reqs):
                    cen = compute_centrality(env, vir_req, rebalance=True)
                    log_request(req_feat, vir_req, env.cur_step, cen, wait_tolerance = 0, delay_tolerance = 0)
                    req_uniques.append(-1-i)
                    req_inds.append(-1-i)
                    # log virtual req self edge
                    req2req_sender.append(len(req_feat)-1)
                    req2req_receiver.append(len(req_feat)-1)
                    req2req_edge_attr.append([0])

                    # log virtual req to req
                    #for req_ind, _ in rvgraph.get_vehicle_edges(veh_ind):
                    #    req = s.requests[req_ind]
                    #    virtualCar = Vehicle(-1, vir_req.start)
                    #    cost = travel(virtualCar, env.cur_step, [vir_req,req], env.traffic, False)
                    #    if cost >= 0:
                    #        req2req_sender.append(len(req_feat)-1)
                    #        req2req_receiver.append(req_unique_inds[req_ind])
                    #        req2req_edge_attr.append([cost])

                    # log virtual req to veh
                    req2veh_sender.append(len(req_feat)-1)
                    req2veh_receiver.append(veh_unique_inds[veh_ind])
                    req2veh_edge_attr.append([0, 0.])

            else:
                # add one directional virtual request
                if len(veh.scheduleroute) == 0:
                    virtual_req = make_assign_request(veh.get_location(), veh.get_location(), env.cur_step)
                else:
                    virtual_req = make_assign_request(veh.get_location(), veh.scheduleroute[0][1], env.cur_step)
                cen = compute_centrality(env, virtual_req, rebalance=False)
                log_request(req_feat, virtual_req, env.cur_step, cen, wait_tolerance=0, delay_tolerance=0)
                req_uniques.append(-10)
                req_inds.append(-10)
                # log virtual req self edge
                # log virtual req self edge
                req2req_sender.append(len(req_feat)-1)
                req2req_receiver.append(len(req_feat)-1)
                req2req_edge_attr.append([0])

                # log virtual_ req to req
                #for req_ind, _ in rvgraph.get_vehicle_edges(veh_ind):
                #    req = s.requests[req_ind]
                #    virtualCar = Vehicle(-1, virtual_req.start)
                #    cost = travel(virtualCar, env.cur_step, [virtual_req,req], env.traffic, False)
                #    if cost >= 0:
                #        req2req_sender.append(len(req_feat)-1)
                #        req2req_receiver.append(req_unique_inds[req_ind])
                #        req2req_edge_attr.append([cost])
                # log virtual req to veh
                req2veh_sender.append(len(req_feat)-1)
                req2veh_receiver.append(veh_unique_inds[veh_ind])
                req2veh_edge_attr.append([0., 0.])


        graph_s = RideShareState(request_uniques = torch.tensor(req_uniques, dtype=torch.long),
                        request_inds = torch.tensor(req_inds, dtype=torch.long),
                        requests_x = torch.tensor(req_feat,dtype=torch.float),
                        req2req_edge_attr = torch.tensor(req2req_edge_attr,dtype=torch.float),
                        req2req_edge_index = torch.tensor([req2req_sender,req2req_receiver],dtype=torch.long),
                        vehicle_inds = torch.tensor(veh_inds,dtype=torch.float),
                        vehicles_x = torch.tensor(veh_feat,dtype=torch.float),
                        req2veh_sender_edge_index = torch.tensor(req2veh_sender,dtype=torch.long),
                        req2veh_receiver_edge_index = torch.tensor(req2veh_receiver,dtype=torch.long),
                        req2veh_edge_attr = torch.tensor(req2veh_edge_attr,dtype=torch.float),
                        passengers_x = torch.tensor(pas_feat,dtype=torch.float),
                        veh2pas_sender_edge_index = torch.tensor(veh2pas_sender,dtype=torch.long),
                        veh2pas_receiver_edge_index = torch.tensor(veh2pas_receiver,dtype=torch.long)).to(device)
        agents.append(graph_s)
    return agents

class RideShareState(Data):
    def __init__(self, request_uniques = None,
                    request_inds = None,
                    requests_x = None,
                    req2req_edge_attr = None,
                    req2req_edge_index = None,
                    vehicle_inds = None,
                    vehicles_x = None,
                    req2veh_sender_edge_index=None,
                    req2veh_receiver_edge_index=None,
                    req2veh_edge_attr=None,
                    passengers_x = None,
                    veh2pas_sender_edge_index = None,
                    veh2pas_receiver_edge_index = None):
        super(RideShareState, self).__init__()
        self.request_uniques = request_uniques
        self.request_inds = request_inds
        self.requests_x =  (requests_x - ((REQ_LOW+REQ_HIG)*0.5))/(REQ_HIG - REQ_LOW)
        self.req2req_edge_attr = (req2req_edge_attr - ((REQ2REQ_LOW+REQ2REQ_HIG)*0.5))/(REQ2REQ_HIG-REQ2REQ_LOW)
        self.req2req_edge_index = req2req_edge_index
        self.req2req_start_index = torch.tensor([0], dtype=torch.long)

        self.vehicle_inds = vehicle_inds
        self.vehicles_x = (vehicles_x - ((VEH_LOW+VEH_HIG)*0.5))/(VEH_HIG - VEH_LOW)
        self.req2veh_receiver_start_index = torch.tensor([0], dtype=torch.long)
        self.req2veh_sender_edge_index = req2veh_sender_edge_index
        self.req2veh_receiver_edge_index = req2veh_receiver_edge_index
        self.req2veh_edge_attr = (req2veh_edge_attr - ((REQ2VEH_LOW+REQ2VEH_HIG)*0.5))/(REQ2VEH_HIG - REQ2VEH_LOW)

        self.passengers_x = (passengers_x - ((PAS_LOW+PAS_HIG)*0.5))/ (PAS_HIG - PAS_LOW)
        self.veh2pas_sender_edge_index = veh2pas_sender_edge_index
        self.veh2pas_receiver_edge_index = veh2pas_receiver_edge_index


    def __inc__(self, key, value):
        if "index" in key or "face" in key:
            if "req2req" in key:
                return self.requests_x.size(0)
            if "req2veh" in key:
                if "sender" in key:
                    return self.requests_x.size(0)
                else:
                    return self.vehicles_x.size(0)
            if "veh2pas" in key:
                if "sender" in key:
                    return self.vehicles_x.size(0)
                else:
                    return self.passengers_x.size(0)
            else:
                return super(RideShareState, self).__inc__(key, value)
        else:
            return 0


# define models
class TripModel(nn.Module):
    def __init__(self, obs_shape, configs):
        super(TripModel, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        self.edge_mlp1 = nn.Sequential(
                            init_(nn.Linear(2*obs_shape['req_x']+obs_shape['req_e'], configs['trip_hidden'])),
                            nn.Tanh()
        )
        self.node_mlp1 = nn.Sequential(
                            init_(nn.Linear(obs_shape['req_x']+configs['trip_hidden'],configs['trip_hidden']))
        )
        #self.edge_mlp2
        #self.node_mlp2
    def forward(self, inputs):
        src, dest = inputs.req2req_edge_index
        x = torch.cat([inputs.requests_x[src],
                       inputs.requests_x[dest],
                       inputs.req2req_edge_attr], dim=1)
        try:
            edge = self.edge_mlp1(x)
        except:
            print(x)
            print(inputs)
            print(inputs.requests_x_batch)
            raise
        edge_mean = scatter_mean(edge, src, dim=0, dim_size=inputs.requests_x.size(0))
        return self.node_mlp1(torch.cat([inputs.requests_x, edge_mean], dim=1))

class VehAddAtt(nn.Module):
    def __init__(self, obs_shape, configs):
        super(VehAddAtt, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        self.edge_mlp = init_(nn.Linear(obs_shape['req_x'] + \
                                  obs_shape['veh_x'] + \
                                  obs_shape['req2veh'],\
                                  configs['req2veh_hidden']))
        self.att = nn.Sequential(init_(nn.Linear(configs['veh_hidden']+configs['trip_hidden']+configs['req2veh_hidden'],
                                         configs['veh_hidden']+configs['trip_hidden']+configs['req2veh_hidden'])),
                               nn.Tanh(),
                               init_(nn.Linear(configs['veh_hidden']+configs['trip_hidden']+configs['req2veh_hidden'], 1))
        )

    def forward(self, inputs, veh_feat):
        src, dest = inputs.req2veh_sender_edge_index, inputs.req2veh_receiver_edge_index
        x = torch.cat([inputs.requests_x[src],
                       inputs.vehicles_x[dest],
                       inputs.req2veh_edge_attr], dim=1)
        req2veh_feat = self.edge_mlp(x)
        att_score = self.att(torch.cat([veh_feat[dest],req2veh_feat], dim=1))
        att_score = scatter_softmax(att_score, src, dim=0)
        return scatter_mean(att_score*veh_feat[dest], src, dim=0, dim_size=inputs.requests_x.size(0))

class ReqAddAtt(nn.Module):
    def __init__(self, obs_shape, configs):
        super(ReqAddAtt, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        self.att = nn.Sequential(init_(nn.Linear(configs['pas_hidden']+\
                                           configs['veh_hidden']+\
                                           configs['trip_hidden']+\
                                           configs['req_hidden'],
                                           configs['pas_hidden']+\
                                           configs['veh_hidden']+\
                                           configs['trip_hidden']+\
                                           configs['req_hidden'])),
                                 nn.Tanh(),
                                 init_(nn.Linear(configs['pas_hidden']+\
                                           configs['veh_hidden']+\
                                           configs['trip_hidden']+\
                                           configs['req_hidden'], 1))
        )
    def forward(self, inputs, act_feat):
        src = inputs.requests_x_batch
        att_score = self.att(act_feat)
        att_score = scatter_softmax(att_score, src, dim=0)
        return scatter_mean(att_score*act_feat, src, dim=0)

class GraphActor_d(nn.Module):
    def __init__(self, obs_shape, configs):
        super(GraphActor_d, self).__init__()
        #init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
        #                       constant_(x, 0), np.sqrt(2))
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('tanh'))
        self.phi_req = nn.Sequential(init_(nn.Linear(obs_shape['req_x'], configs['req_hidden'])), nn.Tanh())
        self.phi_veh = nn.Sequential(init_(nn.Linear(obs_shape['veh_x'], configs['veh_hidden'])), nn.Tanh())
        self.phi_pas = nn.Sequential(init_(nn.Linear(obs_shape['pas_x'], configs['pas_hidden'])), nn.Tanh())

        self.actor_linear = nn.Sequential(init_(nn.Linear(configs['req_hidden']+configs['veh_hidden']+configs['pas_hidden'],
                                                     configs['act_hidden'])),
                                           nn.Tanh(),
                                           init_(nn.Linear(configs['act_hidden'],
                                                     configs['act_hidden'])),
                                           nn.Tanh(),
                                           init_(nn.Linear(configs['act_hidden'], 1))
        )
    def forward(self, inputs):
        req_feat = self.phi_req(inputs.requests_x)
        veh_feat = self.phi_veh(inputs.vehicles_x)
        pas_feat = self.phi_pas(inputs.passengers_x)
        pas_mean = scatter_mean(pas_feat[inputs.veh2pas_receiver_edge_index],
                                inputs.veh2pas_sender_edge_index,
                                dim=0,
                                dim_size=inputs.vehicles_x.size(0))
        veh_feat = torch.cat([veh_feat, pas_mean], dim=-1)
        src, dest = inputs.req2veh_sender_edge_index, inputs.req2veh_receiver_edge_index
        veh_feat = scatter_mean(veh_feat[dest], src, dim=0, dim_size=inputs.requests_x.size(0))
        self.action_feat = torch.cat([req_feat, veh_feat], dim=-1)
        #self.action_feat = req_feat
        return self.actor_linear(self.action_feat)

class GraphCritic_d(nn.Module):
    def __init__(self, obs_shape, configs):
        super(GraphCritic_d, self).__init__()
        #init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
        #                       constant_(x, 0), np.sqrt(2))
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('tanh'))

        # request network
        self.phi_req = nn.Sequential(init_(nn.Linear(obs_shape['req_x'], configs['req_hidden'])), nn.Tanh())
        self.phi_veh = nn.Sequential(init_(nn.Linear(obs_shape['veh_x'], configs['veh_hidden'])), nn.Tanh())
        self.phi_pas = nn.Sequential(init_(nn.Linear(obs_shape['pas_x'], configs['pas_hidden'])), nn.Tanh())
        # critic network
        self.critic_linear = nn.Sequential(init_(nn.Linear(configs['req_hidden']+configs['veh_hidden']*2+configs['pas_hidden'],
                                                     configs['act_hidden'])),
                                           nn.Tanh(),
                                           init_(nn.Linear(configs['act_hidden'],
                                                     configs['act_hidden'])),
                                           nn.Tanh(),
                                           init_(nn.Linear(configs['act_hidden'], 1))
        )

    def forward(self, inputs):
        req_feat = self.phi_req(inputs.requests_x)
        req_feat = scatter_mean(req_feat, inputs.requests_x_batch, dim=0)
        #req_feat = req_feat[inputs.req2req_action_index]
        #src, dest = inputs.req2veh_sender_edge_index, inputs.req2veh_receiver_edge_index
        veh_feat = self.phi_veh(inputs.vehicles_x)

        pas_feat = self.phi_pas(inputs.passengers_x)
        pas_mean = scatter_mean(pas_feat[inputs.veh2pas_receiver_edge_index],
                                inputs.veh2pas_sender_edge_index,
                                dim=0,
                                dim_size=inputs.vehicles_x.size(0))
        ego_feat = torch.cat([veh_feat, pas_mean], dim=-1)[inputs.req2veh_receiver_start_index]
        veh_mean = scatter_mean(veh_feat, inputs.vehicles_x_batch, dim=0)
        #veh_feat = scatter_mean(veh_feat[dest], src, dim=0, dim_size=inputs.requests_x.size(0))
        #self.action_feat = torch.cat([req_feat, veh_feat], dim=-1)[inputs.req2req_action_index]
        #self.action_feat = req_feat[inputs.req2req_action_index]
        self.action_feat = torch.cat([req_feat, ego_feat, veh_mean], dim=-1)
        #self.action_feat,_ = scatter_max(req_feat, inputs.requests_x_batch, dim=0)

        return self.critic_linear(self.action_feat)

class GraphActor(nn.Module):
    def __init__(self, obs_shape, configs):
        super(GraphActor, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        # passenger_network
        self.phi_pas = init_(nn.Linear(obs_shape['pas_x'], configs['pas_hidden']))
        # vehicle network
        self.phi_veh = init_(nn.Linear(obs_shape['veh_x'], configs['veh_hidden']))
        # request network
        self.phi_req = init_(nn.Linear(obs_shape['req_x'], configs['req_hidden']))
        # trip network
        self.phi_trip = TripModel(obs_shape, configs)
        # vehicle attention
        self.att_veh = VehAddAtt(obs_shape, configs)
        # actor network
        self.actor_linear = nn.Sequential(init_(nn.Linear(configs['pas_hidden']+\
                                                     configs['veh_hidden']+\
                                                     configs['trip_hidden']+\
                                                     configs['req_hidden'],
                                                     configs['act_hidden'])),
                                           nn.Tanh(),
                                           init_(nn.Linear(configs['act_hidden'],
                                                     configs['act_hidden'])),
                                           nn.Tanh(),
                                           init_(nn.Linear(configs['act_hidden'], 1))
        )

    def forward(self, inputs):
        pas_feat = self.phi_pas(inputs.passengers_x)
        pas_mean = scatter_mean(pas_feat[inputs.veh2pas_receiver_edge_index],
                                inputs.veh2pas_sender_edge_index,
                                dim=0,
                                dim_size=inputs.vehicles_x.size(0))
        veh_feat = self.phi_veh(inputs.vehicles_x)

        veh_feat = torch.cat([veh_feat, pas_mean], dim=1)

        req_feat = self.phi_req(inputs.requests_x)
        trip_feat = self.phi_trip(inputs)
        veh_feat = self.att_veh(inputs, veh_feat)
        action_feat = torch.cat([req_feat, trip_feat, veh_feat], dim=1)
        return self.actor_linear(action_feat)

class GraphCritic(nn.Module):
    def __init__(self, obs_shape, configs):
        super(GraphCritic, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        # passenger_network
        self.phi_pas = init_(nn.Linear(obs_shape['pas_x'], configs['pas_hidden']))
        # vehicle network
        self.phi_veh = init_(nn.Linear(obs_shape['veh_x']+obs_shape['veh_act'], configs['veh_hidden']))
        # request network
        self.phi_req = init_(nn.Linear(obs_shape['req_x'], configs['req_hidden']))
        # trip network
        self.phi_trip = TripModel(obs_shape, configs)
        # vehicle attention
        self.att_veh = VehAddAtt(obs_shape, configs)
        # request attention
        self.att_req = ReqAddAtt(obs_shape, configs)
        # critic network
        self.critic_linear = nn.Sequential(init_(nn.Linear(configs['pas_hidden']+\
                                                     configs['veh_hidden']+\
                                                     configs['trip_hidden']+\
                                                     configs['req_hidden'],
                                                     configs['act_hidden'])),
                                           nn.Tanh(),
                                           init_(nn.Linear(configs['act_hidden'],
                                                     configs['act_hidden'])),
                                           nn.Tanh(),
                                           init_(nn.Linear(configs['act_hidden'], 1))
        )

    def forward(self, inputs):
        pas_feat = self.phi_pas(inputs.passengers_x)
        pas_mean = scatter_mean(pas_feat[inputs.veh2pas_receiver_edge_index],
                                inputs.veh2pas_sender_edge_index,
                                dim=0,
                                dim_size=inputs.vehicles_x.size(0))
        veh_feat = self.phi_veh(torch.cat([inputs.vehicles_x, inputs.vehicle_action], dim=1))

        veh_feat = torch.cat([veh_feat, pas_mean], dim=1)

        req_feat = self.phi_req(inputs.requests_x)
        trip_feat = self.phi_trip(inputs)
        veh_feat = self.att_veh(inputs, veh_feat)
        action_feat = torch.cat([req_feat, trip_feat, veh_feat], dim=1)

        action_feat = self.att_req(inputs, action_feat)
        return self.critic_linear(action_feat)


class GraphACModel(nn.Module):
    def __init__(self, obs_shape, configs):
        super(GraphACModel, self).__init__()

        self.actor = GraphActor_d(obs_shape, configs)
        self.critic = GraphCritic_d(obs_shape, configs)


    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def sample(self, probs, inds, batch_size):
        actions = []
        log_probs = []
        for i in range(batch_size):
            probs_ = probs[inds==i].squeeze(1)
            #print("P", probs_)
            dist = torch.distributions.Categorical(probs_)
            a = dist.sample()
            #print("A", a)
            actions.append(a)
            #print("L", dist.log_prob(a))
            log_probs.append(dist.log_prob(a))
        actions = torch.stack(actions).unsqueeze(1)
        log_probs = torch.stack(log_probs).unsqueeze(1)
        #print("AC", actions)
        #print("LP ", log_probs)
        #raise
        return actions, log_probs

    def act(self, inputs, deterministic=False):
        actor_logits = self.actor(inputs)

        log_probs = scatter_log_softmax(actor_logits,
                                           inputs.requests_x_batch,
                                           dim=0)
        probs = scatter_softmax(actor_logits, inputs.requests_x_batch, dim=0)
        #print("inds ", inputs.requests_x_batch)
        #print("probs", [probs[inputs.requests_x_batch==i].squeeze() for i in range(inputs.req2req_start_index.size(0))])
        #print("log_probs", [log_probs[inputs.requests_x_batch==i].squeeze() for i in range(inputs.req2req_start_index.size(0))])
        action, action_log_probs = self.sample(probs, inputs.requests_x_batch, inputs.req2req_start_index.size(0))
        #action_log_probs, action = scatter_max(log_probs, inputs.requests_x_batch, dim=0)
        #if deterministic:
        #    action = dist.mode()
        #else:
        #    action = dist.sample()
        #action = action - inputs.req2req_start_index.unsqueeze(1)
        return action, action_log_probs

    def get_value(self, inputs):
        value = self.critic(inputs)
        return value

    def evaluate_actions(self, inputs):
        value = self.critic(inputs)
        actor_logits = self.actor(inputs)

        log_probs = scatter_log_softmax(actor_logits,
                                           inputs.requests_x_batch,
                                           dim=0)
        #print("action we get: ", inputs.req2req_action_index)
        #print("action we get: ", inputs.req2req_action_index - inputs.req2req_start_index)

        action_log_probs = log_probs[inputs.req2req_action_index]
        probs = scatter_softmax(actor_logits, inputs.requests_x_batch, dim=0)
        dist_entropy = scatter_sum(-probs*log_probs, inputs.requests_x_batch, dim=0).mean()

        return value, action_log_probs, dist_entropy

class MLPActor_d(nn.Module):
    def __init__(self, obs_shape, configs):
        super(MLPActor_d, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        self.phi_veh = init_(nn.Linear(obs_shape['veh_x'], configs['veh_hidden']))

        self.actor_linear = nn.Sequential(init_(nn.Linear(configs['veh_hidden'],
                                                     configs['act_hidden'])),
                                           nn.Tanh(),
                                           init_(nn.Linear(configs['act_hidden'],
                                                     configs['act_hidden'])),
                                           nn.Tanh())

    def forward(self, inputs):
        veh_feat = self.phi_veh(inputs.vehicles_x[inputs.req2veh_receiver_start_index])

        return self.actor_linear(veh_feat)

class MLPCritic_d(nn.Module):
    def __init__(self, obs_shape, configs):
        super(MLPCritic_d, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        self.phi_veh = init_(nn.Linear(obs_shape['veh_x'], configs['veh_hidden']))
        # critic network
        self.critic_linear = nn.Sequential(init_(nn.Linear(configs['veh_hidden'],
                                                     configs['act_hidden'])),
                                           nn.Tanh(),
                                           init_(nn.Linear(configs['act_hidden'],
                                                     configs['act_hidden'])),
                                           nn.Tanh(),
                                           init_(nn.Linear(configs['act_hidden'], 1))
        )

    def forward(self, inputs):
        veh_feat = self.phi_veh(inputs.vehicles_x[inputs.req2veh_receiver_start_index])
        return self.critic_linear(veh_feat)

class MLPACModel(nn.Module): # we need to first make sure the A2C is correct
    def __init__(self, obs_shape, configs):
        super(MLPACModel, self).__init__()

        self.actor = MLPActor_d(obs_shape, configs)
        self.critic = MLPCritic_d(obs_shape, configs)
        self.dist = Categorical(configs['act_hidden'], 5)

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, deterministic=False):
        actor_feat = self.actor(inputs)
        dist  = self.dist(actor_feat)
        #print("probs", dist.probs)
        #action = dist.mode()
        action = dist.sample()
        #if deterministic:
        #    action = dist.mode()
        #else:
        #    action = dist.sample()
        #action = action - inputs.req2req_start_index.unsqueeze(1)
        action_log_probs = dist.log_probs(action)
        return action, action_log_probs

    def get_value(self, inputs):
        value = self.critic(inputs)
        return value

    def evaluate_actions(self, inputs):
        value = self.critic(inputs)
        actor_feat = self.actor(inputs)

        dist = self.dist(actor_feat)

        action = inputs.req2req_action_index - inputs.req2req_start_index
        action = action.unsqueeze(1)
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        return value, action_log_probs, dist_entropy

# define agent
class MGACK(object):
    def __init__(self, actor_critic, configs, acktr=False):
        #value_loss_coef,
        #entropy_coef,
        #lr=None,
        #eps=None,
        #alpha=None,
        #max_grad_norm=None
        self.actor_critic = actor_critic
        self.acktr = acktr

        self.value_loss_coef = configs['value_loss_coef']
        self.entropy_coef = configs['entropy_coef']

        self.max_grad_norm = configs['max_grad_norm']

        if acktr:
            self.optimizer = KFACOptimizer(actor_critic, lr=configs['lr'], weight_decay=configs['weight_decay'])
        else:
            self.optimizer = optim.RMSprop(
                actor_critic.parameters(), configs['lr'], eps=configs['eps'], alpha=configs['alpha'])

    def update(self, rollouts):
        dataloader = rollouts.feed_forward_dataloader()
        for sample in dataloader:
            #print("action we give :", rollouts.actions)
            values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(sample)
            #print("v", values.size())
            #print("ret", sample.returns.unsqueeze(1).size())
            #print("ap", action_log_probs.size())
            advantages = sample.returns.unsqueeze(1) - values
            value_loss = advantages.pow(2).mean()
            #print("  act: ", sample.request_inds[sample.req2req_action_index])
            #print("  act: ", sample.req2req_action_index - sample.req2req_start_index)
            #print("  ret: ", sample.returns.squeeze())
            #print("  val: ", values.squeeze())
            #print("  ad: ", advantages.squeeze())
            #print("  ap: ", action_log_probs.squeeze())
            action_loss = -(advantages.detach() * action_log_probs).mean()
            #print("  lo:", action_loss)
            '''
            with torch.no_grad():
                print("  cri act feat: ", self.actor_critic.critic.action_feat)
                x = self.actor_critic.critic.action_feat
                for module in self.actor_critic.critic.critic_linear:
                    x = module(x)
                    print(" cri act linear feat: ", x)
                print("  ac act feat: ", self.actor_critic.actor.action_feat)
                x = self.actor_critic.actor.action_feat
                for module in self.actor_critic.actor.actor_linear:
                    x = module(x)
                    print(" ac act linear feat: ", x)
            '''

            if self.acktr and self.optimizer.steps % self.optimizer.Ts == 0:
                # Compute fisher, see Martens 2014
                self.actor_critic.zero_grad()
                pg_fisher_loss = -action_log_probs.mean()

                value_noise = torch.randn(values.size())
                if values.is_cuda:
                    value_noise = value_noise.cuda()

                sample_values = values + value_noise
                vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()

                fisher_loss = pg_fisher_loss + vf_fisher_loss
                self.optimizer.acc_stats = True
                fisher_loss.backward(retain_graph=True)
                self.optimizer.acc_stats = False

            self.optimizer.zero_grad()
            (value_loss * self.value_loss_coef + action_loss -
            dist_entropy * self.entropy_coef).backward()

            if self.acktr == False:
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                        self.max_grad_norm)

            self.optimizer.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item()
#define runner

class Runner(object):
    def __init__(self, env, rl_agent, expert_agent, num_steps, train=True):
        self.env = env
        self.rl_agent = rl_agent
        self.expert_agent = expert_agent
        self.train = train
        self.num_steps = num_steps
        self.rollouts = MGRolloutBuffer(num_steps, len(env.vehicles))

    def run(self, episode):
        # reset environment
        #obs = self.env.reset()
        if self.env.cur_step > 50:
            obs = self.env.reset()
        else:
            obs = self.env.current_state
        graph_obs = get_state(self.env, obs)
        self.rollouts.obs[0] = graph_obs
        episode_rewards = []
        for step in range(self.num_steps):
            # sample actions
            print("obs:", [self.env.vehicles[int(obs.vehicle_inds[0].cpu().item())].get_location() for obs in self.rollouts.obs[step]])

            if self.expert_agent and episode < 100 and False:
                actions = self.expert_agent.act(obs, 0)
                graph_actions = convert_expert_action(self.env, self.rollouts.obs[step], actions)
                mean_actions = get_mean_action(self.rollouts.obs[step], graph_actions)
                for veh_ind in range(len(self.rollouts.obs[step])):
                    self.rollouts.obs[step][veh_ind].req2req_action_index = graph_actions[veh_ind]
                    self.rollouts.obs[step][veh_ind].vehicle_action = mean_actions[veh_ind]
                values, action_log_probs, _ = self.rl_agent.actor_critic.evaluate_actions(
                                                    Batch.from_data_list(self.rollouts.obs[step],
                                                        ['requests_x', 'vehicles_x','passengers_x'])
                )
            else:
                with torch.no_grad():
                    graph_actions, action_log_probs = self.rl_agent.actor_critic.act(
                        Batch.from_data_list(self.rollouts.obs[step],['requests_x', 'vehicles_x','passengers_x'])
                    )
                    print("act: ", [obs.request_inds[a].cpu().item() for obs, a in zip(self.rollouts.obs[step], graph_actions)])
                    #print("act:", [a.cpu().item() for a in graph_actions])

                    '''
                    actions = []
                    for veh_ind in range(len(self.rollouts.obs[step])):
                        assert(veh_ind == self.rollouts.obs[step][veh_ind].vehicle_inds[0])
                        a = graph_actions[veh_ind][0]
                        veh = self.env.vehicles[veh_ind]
                        loc = veh.get_location()
                        if a == 0:
                            new_loc = (max(loc[0]-1,0), loc[1])
                        elif a == 1:
                            new_loc = (min(veh.location[0]+1, self.env.width-1), loc[1])
                        elif a == 2:
                            new_loc = (veh.location[0], min(veh.location[1]+1,self.env.height-1))
                        elif a == 3:
                            new_loc = (veh.location[0], max(veh.location[1]-1,0))
                        else:
                            #print(a)
                            assert(a == 4)
                            new_loc = (veh.location[0], veh.location[1])
                        actions.append(new_loc)
                    '''
                    actions = get_env_action(self.env, self.rollouts.obs[step], graph_actions)
                    #TODO temp
                    print("act: ", actions)
                    #mean_actions = get_mean_action(self.rollouts.obs[step], graph_actions)
                    for veh_ind in range(len(self.rollouts.obs[step])):
                        self.rollouts.obs[step][veh_ind].req2req_action_index = graph_actions[veh_ind]
                        #self.rollouts.obs[step][veh_ind].vehicle_action = mean_actions[veh_ind]
                    values = self.rl_agent.actor_critic.get_value(
                        Batch.from_data_list(self.rollouts.obs[step],
                            ['requests_x', 'vehicles_x','passengers_x'])
                    )
            rewards = get_centrality_reward(self.env, graph_obs, graph_actions)
            #rewards = get_passenger_reward(self.env, graph_obs, graph_actions)
            #rewards = get_request_reward(self.env, graph_obs, graph_actions)

            print("r: ", rewards.squeeze(), " \n")
            self.env.execute_agent_action(actions)
            infos = get_info(self.env)
            obs, done = get_done(self.env)
            graph_obs = get_state(self.env, obs)

            episode_rewards.append(rewards.mean().cpu().item())

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            self.rollouts.insert(graph_obs, graph_actions,
                            action_log_probs, values, rewards, masks, bad_masks)

        with torch.no_grad():
            graph_actions, action_log_prob = self.rl_agent.actor_critic.act(
                Batch.from_data_list(self.rollouts.obs[-1], ['requests_x', 'vehicles_x','passengers_x'])
            )
            #mean_actions = get_mean_action(self.rollouts.obs[-1], graph_actions)
            for veh_ind in range(len(self.rollouts.obs[-1])):
                self.rollouts.obs[-1][veh_ind].req2req_action_index = graph_actions[veh_ind]
                #self.rollouts.obs[-1][veh_ind].vehicle_action = mean_actions[veh_ind]
            next_value = self.rl_agent.actor_critic.get_value(
                    Batch.from_data_list(self.rollouts.obs[-1],
                        ['requests_x', 'vehicles_x','passengers_x'])
            ).detach()
        self.rollouts.compute_returns(next_value, configs['use_gae'], configs['gamma'],
                                 configs['gae_lambda'], configs['use_proper_time_limits'])

        if self.train:
            #print(self.rollouts.returns.squeeze())
            value_loss, action_loss, dist_entropy = self.rl_agent.update(self.rollouts)
        self.rollouts.after_update()
        return action_loss, value_loss, dist_entropy, episode_rewards

def evaluate(actor_critic, env):

    eval_episode_rewards = []

    obs = env.reset()
    graph_obs = get_state(env, obs)

    while len(eval_episode_rewards) < 10:
        with torch.no_grad():
            graph_actions, _ = actor_critic.act(
                Batch.from_data_list(graph_obs, ['requests_x', 'vehicles_x','passengers_x'])
            )
        # Obser reward and next obs
        actions = get_env_action(env, graph_obs, graph_actions)
        '''
        actions = []
        for veh_ind in range(len(graph_obs)):
            assert(veh_ind == graph_obs[veh_ind].vehicle_inds[0])
            a = graph_actions[veh_ind][0]
            veh = env.vehicles[veh_ind]
            loc = veh.get_location()
            if a == 0:
                new_loc = (max(loc[0]-1,0), loc[1])
            elif a == 1:
                new_loc = (min(veh.location[0]+1, env.width-1), loc[1])
            elif a == 2:
                new_loc = (veh.location[0], min(veh.location[1]+1,env.height-1))
            elif a == 3:
                new_loc = (veh.location[0], max(veh.location[1]-1,0))
            else:
                assert(a == 4)
                new_loc = (veh.location[0], veh.location[1])
            actions.append(new_loc)
        '''
        rewards = get_centrality_reward(env, graph_obs, graph_actions)
        #rewards = get_passenger_reward(env, graph_obs, graph_actions)
        #rewards = get_request_reward(env, graph_obs, graph_actions)
        #print("obs:", [obs.vehicles_x[0].cpu() for obs in graph_obs])
        #print("act:", [a.cpu().item() for a in graph_actions])
        #print("act: ", actions)
        #print("r: ", rewards.squeeze(), " \n")

        env.execute_agent_action(actions)
        obs, done = get_done(env)
        last_graph_obs = graph_obs
        graph_obs = get_state(env, obs)

        eval_episode_rewards.append(rewards.mean().cpu().item())



    #print("act: ", [obs.request_uniques[a].cpu().item() for obs, a in zip(last_graph_obs, graph_actions)])

    #print("r: ", rewards.squeeze())

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))
    return np.mean(eval_episode_rewards)

def demonstrate(expert_agent, env):

    eval_episode_rewards = []

    obs = env.reset()
    graph_obs = get_state(env, obs, "cpu")

    while len(eval_episode_rewards) < 10:
        actions = expert_agent.act(obs, 0)
        graph_actions = convert_expert_action(env, graph_obs, actions, "cpu")
        rewards = get_reward(env, graph_obs, graph_actions, "cpu")
        #print("r: ", rewards.squeeze())
        env.execute_agent_action(actions)
        obs, done = get_done(env)
        graph_obs = get_state(env, obs, "cpu")

        eval_episode_rewards.append(rewards.mean().cpu().item())


    print(" Demonstrate using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))
    return np.mean(eval_episode_rewards)

def main():
    env = GridWorldRideSharing(num_vehicles=15)
    eval_env = GridWorldRideSharing(num_vehicles=15)
    expert_env = GridWorldRideSharing(num_vehicles=15)
    actor_critic_model = GraphACModel(obs_shape, configs) #MLPACModel(obs_shape, configs)#
    actor_critic_model.to("cuda:0")
    torch.manual_seed(configs['seed'])
    torch.cuda.manual_seed_all(configs['seed'])
    rl_agent = MGACK(actor_critic_model, configs, acktr=True)
    expert_agent = Agent(env)
    expert_agent2 = Agent(expert_env)
    runner = Runner(env, rl_agent, expert_agent, configs['num_steps'])
    writer = SummaryWriter(configs['save_dir'])
    #episode_rewards = deque(maxlen=10)

    for k in range(configs['num_updates']):
        gc.collect()
        if configs['use_linear_lr_decay']:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                rl_agent.optimizer, k, configs['num_updates'],
                rl_agent.optimizer.lr if configs['algo'] == "acktr" else configs['lr'])

        action_loss, value_loss, dist_entropy, episode_rewards = runner.run(k)
        # some information logging
        if (k % configs['save_interval'] == 0
                or k == configs['num_updates'] - 1) and configs['save_dir'] != "":
            save_path = os.path.join(configs['save_dir'], "a2c")
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save(
                actor_critic_model.state_dict()
            , os.path.join(save_path, "toy_ridesharing_gn_cen2_acktr_batch_gae_tanh{}_lr-{}_l2-{}.pt".format(k, configs['lr'],configs['weight_decay'])))

        writer.add_scalar('dist_entropy',dist_entropy, k)
        writer.add_scalar('value_loss', value_loss, k)
        writer.add_scalar('action_loss', action_loss, k)
        if k % configs['log_interval'] == 0:
            total_num_steps = (k + 1) * configs['num_steps']

            print(
                "Updates {}, num timesteps {} \n Last training episodes: rewards {:.3f}, dist_entropy {:.3f}, value_loss {:.3f}, action_loss {:.3f}\n"
                .format(k, total_num_steps, np.mean(episode_rewards),
                        dist_entropy, value_loss,
                        action_loss))

        if (configs['eval_interval'] is not None and k % configs['eval_interval'] == 0):
            eval_reward = evaluate(actor_critic_model, eval_env)
            with torch.no_grad():
                writer.add_histogram("cri feat", actor_critic_model.critic.action_feat, k)
                x = actor_critic_model.critic.action_feat
                for module in actor_critic_model.critic.critic_linear:
                    x = module(x)
                    writer.add_histogram("cri "+module.__str__(), x, k)
                writer.add_histogram("ac feat", actor_critic_model.actor.action_feat)
                x = actor_critic_model.actor.action_feat
                for module in actor_critic_model.actor.actor_linear:
                    x = module(x)
                    writer.add_histogram("ac "+module.__str__(), x, k)

            #expert_reward = demonstrate(expert_agent2, expert_env)
            writer.add_scalar('eval mean reward', eval_reward, k)
            #writer.add_scalar('expert mean reward', expert_reward, k)

configs = {"veh_hidden":16,
           "req_hidden":16,
           "pas_hidden":16,
           "req2veh_hidden":32,
           "trip_hidden":32,
           "act_hidden":64,
           "num_steps":10,
           "num_updates":300,
           "batch_size":64,
           "lr": 0.25,
           "weight_decay":1e-4,
           "eps":1e-5,
           "alpha":0.99,
           "algo":"a2c",
           "save_dir":"./model",
           "log_interval":10,
           "save_interval":100,
           "eval_interval":20,
           "value_loss_coef":0.5,
           "entropy_coef":0.01,
           "max_grad_norm":0.5,
           "use_gae":True,
           "gamma":0.99,
           "gae_lambda":0.95,
           "use_proper_time_limits":False,
           "use_linear_lr_decay":False,
           'seed':36
           }
obs_shape = {"req_x":10, "veh_x":8,
             "pas_x":10, "veh_act":8, "req2veh":2, "req_e":1}

if __name__ == "__main__":
    main()
