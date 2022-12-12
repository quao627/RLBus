from typing import List, Dict, Union
from dataclasses import dataclass
import pickle
import json
import random
from itertools import chain

import gym
from gym.spaces import Box, Discrete, Tuple
import simpy
from simpy.events import AnyOf, AllOf, Event

from utils import *

num_skipping_stop = 0
num_total_stop = 0


'''
By 12/07:
Todos (Simulator):
- [✅] Shift passenger arrival time
- [✅] Buses have to alight all passengers at terminals 
- [✅] Make buses stick to the schedule at terminal
- [ ] Fill action, state, reward buffers
- [✅ ] Make sure the plots make sense (No gap between buses at the terminal)
- [✅] Test skipping & turning around actions
- [ ] Calculate bus locations

Todos (RL):

'''

data = {station_idx: [] for station_idx in range(N_STATION)}

# event_buffer = {bus_id: {'ready': False, 'events': []} for bus_id in range(N_BUS)}

class Bus:
    def __init__(self, env, simpy_env, idx, starting_time) -> None:
        self.env = env
        self.simpy_env = simpy_env
        self.idx = idx
        self.capacity = CAPACITY
        self.cur_station = self.env.stations[0]
        self.next_station = self.cur_station
        #self.next_travel_time = self.env.get_travel_time(self.cur_station)
        self.next_travel_time = 0
        self.starting_time = starting_time
        # print(f'Bus {self.idx} is successfully initialized.')
        self.proc = self.simpy_env.process(self.drive())
        self.passengers = []
        self.num_pax = 0
        self.state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.h_action = None
        self.l_action = None
        self.taking_action = False
        self.last_departure_time = starting_time
        self.action_start_time = None


    def drive(self):
        # hold at the terminal before starting to drive
        yield self.simpy_env.timeout(self.starting_time)
        self.env.departure_times.append(self.simpy_env.now)
        data[self.cur_station.idx].append(self.simpy_env.now)
        #print(f'Bus {self.idx} starts at {self.simpy_env.now}')

        # each cycle is a trip from one station to the next
        while True:
            turn_around = False

            # drive till the next station
            yield self.simpy_env.timeout(self.next_travel_time)
            if (len(self.env.departure_times) >= N_BUS) and (self.next_station.idx == 0):
                while (self.simpy_env.now < self.env.departure_times[-1] + HEADWAY):
                    yield self.simpy_env.timeout(self.env.departure_times[-1] + HEADWAY - self.simpy_env.now)
                self.env.departure_times.append(self.simpy_env.now)
            # request to enter the station
            self.env.ready = True
            self.taking_action = True
            self.env.acting_bus = self.idx
            self.action_start_time = self.simpy_env.now
            self.env.allow_skipping = (not any([pax.alight_station == self.next_station.idx for pax in self.passengers])) and (self.next_station.idx != 0)
            self.env.allow_turning_around = (self.next_station.idx != 0)

            yield self.simpy_env.timeout(0)

            # high-level and low-level actions
            h_action, l_action = self.env.action
            if h_action == 0:
                l_action = l_action[0]
            elif h_action == 2:
                l_action = l_action[1]
            else:
                l_action = 0
            self.h_action = h_action
            self.l_action = l_action
            # 0 means holding

            if h_action == 0 or self.next_station.idx in [N_STATION-1, int(N_STATION//2)-1]:
                with self.next_station.request() as req:
                    yield req
                    self.station_enter_time = self.simpy_env.now
                #print(f'Bus {self.idx} Arrives At Station {self.next_station.idx}')
                
                # if not any([pax.alight_station == self.next_station.idx for pax in self.passengers]):
                #     global num_skipping_stop
                #     num_skipping_stop = num_skipping_stop + 1
                # global num_total_stop               
                # num_total_stop += 1
                
                    data[self.next_station.idx].append(self.simpy_env.now)
                    if self.next_station.idx in [N_STATION-1, int(N_STATION//2)-1]:
                        pax_alight = self.alight_all_pax(self.next_station)
                        pax_board = 0
                    else:
                        pax_alight = self.alight_pax(self.next_station)
                        pax_board = self.board_pax(self.next_station)
                    holding_time = max(pax_board * t_b, pax_alight * t_a)
                    holding_time += l_action
                    self.station_holding_time = holding_time

                    yield self.simpy_env.timeout(holding_time)
                    self.update_state(h_action)
                    #print(f'Bus {self.idx} holds at station {self.cur_station.idx} for {holding_time} seconds')
                
            # 1 means skipping
            elif h_action == 1:
                yield self.simpy_env.timeout(0)
                self.update_state(h_action)
                #print(f'Bus {self.idx} skips station {self.cur_station.idx}')

            # 2 means turning around
            else: 
                turn_around = True

            if turn_around:
                # alight all passengers
                pax_alight = self.alight_all_pax(self.next_station) # then alight all other passengers
                holding_time = pax_alight * t_a
                # wait to alight all passengers
                yield self.simpy_env.timeout(holding_time)

                # turn around
                self.update_state(h_action)

                # wait to turn around
                yield self.simpy_env.timeout(l_action)

                #print(f'Bus {self.idx} turns around at station {self.cur_station.idx} to {self.next_station.idx} for {l_action} seconds')
            self.taking_action = False
            self.last_departure_time = self.simpy_env.now

    def alight_all_pax(self, station):
        """
        1, alight all passengers that are on the bus
        2, return the number of passengers that are leaving
        3, update the number of passengers on the bus to 0
        4, update passengers' last_start_time

        Return:
            the number of passengers that are boarding
        """
        passengers_left = []
        for pax in self.passengers:
            if pax.alight_station != station.idx:
                pax.bus = None
                passengers_left.append(pax)
                pax.new_status = 0
            else:
                pax.new_status = 2
        num_pax = len(self.passengers)
        station.passengers.extend(passengers_left)
        station.passengers.sort(key=lambda x: x.start_time)
        self.passengers = []
        self.num_pax = 0
        return num_pax


    def alight_pax(self, station):
        """
        1, alight the passengers that are supposed to get off
        2, return the number of passengers that are leaving
        3, update the number of passengers on the bus

        Return:
            the number of passengers that are boarding
        """
        pax_alight = 0
        passengers = []
        for pax in self.passengers:
            if pax.alight_station == station.idx:
                pax_alight += 1
                pax.new_status = 2     
            else:
                passengers.append(pax)
        self.passengers = passengers
        self.num_pax = len(passengers)
        return pax_alight

    def board_pax(self, station):
        """
        1, move the passengers to the bus
        2, return the number of passengers that are boarding
        3, update the arrival time of the last bus
        4, remove the passengers that are boarding from the station
        5, update passengers' on station time

        Parameters:
            station: the station that the bus is arriving at
        
        Return:
            the number of passengers that are boarding
        """
        STOP_BOARDING = False
        passengers = []
        n = 0
        for pax in station.passengers:
            if (pax.start_time < self.simpy_env.now) and (pax.bus is None) and (not STOP_BOARDING):
                pax.bus = self.idx
                self.passengers.append(pax)
                self.num_pax += 1
                n += 1
                pax.new_status = 1
                if self.num_pax == self.capacity:
                    STOP_BOARDING = True
            else:
                passengers.append(pax)
                
        station.last_arrival_time = self.simpy_env.now
        station.passengers = passengers
        return n

    def update_state(self, h_action):
        if h_action == 2:
            self.cur_station = self.next_station
            self.next_station = self.next_station.get_opposite()
            self.next_travel_time = 0
        else:
            self.cur_station = self.next_station
            self.next_station = self.next_station.get_next()
            self.next_travel_time = self.env.get_travel_time(self.cur_station)

    def get_observation(self):
        # state should include [location, headway, ego, pax, h_action, l_action]
        if self.starting_time > self.simpy_env.now:
            location = 0
        else:
            if self.taking_action:
                location = self.next_station.idx * STATION_DIST
            else:
                if self.next_travel_time == 0:
                    location = self.cur_station.idx * STATION_DIST
                else:
                    location = min(1, (self.simpy_env.now - self.last_departure_time) / self.next_travel_time) * STATION_DIST + self.cur_station.idx * STATION_DIST
        location = location % N_STATION
        action_duration = self.simpy_env.now - self.action_start_time if self.taking_action else 0
        return {'location': location,
                'pax': self.num_pax,
                'h_action': self.h_action if self.taking_action else None,
                'l_action': self.l_action if self.taking_action else None,
                'action_duration': action_duration,}

class Station:
    def __init__(self, 
                 env,
                 simpy_env, 
                 idx,
                 pax_alight,
                 pax_board) -> None:
        self.env = env
        self.simpy_env = simpy_env
        self.resource = simpy.Resource(simpy_env, capacity=1)
        self.idx = idx
        self.last_station = None
        self.next_station = None
        self.opposite_station = None
        self.pax_alight = pax_alight
        self.pax_board = pax_board
        self.passengers = self.generate_pax(self.pax_alight, self.pax_board)
        self.env.passengers.extend(self.passengers)
        self.last_arrival_time = 0

    def set_last(self, station):
        self.last_station = station

    def set_next(self, station):
        self.next_station = station

    def set_opposite(self, station):
        self.opposite_station = station

    def get_last(self):
        return self.last_station

    def get_next(self):
        return self.next_station

    def get_opposite(self):
        return self.opposite_station

    def request(self):
        return self.resource.request()

    def generate_pax(self, pax_alight, pax_board):
        return [Passenger(pax_board[i], pax_alight[i], pax_board[i]) for i in range((pax_board != np.inf).sum())] 


class Env(gym.Env):
    def __init__(self) -> None:
        self.travel_times = TABLE_TRAVEL_TIME
        self.pax_alight = PAX_ALIGHT_TABLE
        self.pax_board = PAX_ARRIVE_TABLE
        self.env = simpy.Environment()
        self.ready = False
        self.action = None
        self.departure_times = []
        self.passengers = []
        self.acting_bus = None
        self.allow_skipping = False
        self.allow_turning_around = False

        self.stations = [Station(self, self.env, i, self.pax_alight[i], self.pax_board[i]) for i in range(N_STATION)]
        self.arange_stations()
        self.buses = [Bus(self, self.env, i, BUS_SCHEDULE[i]) for i in range(N_BUS)]
        
        self.last_timestep = 0
        self.acc_waiting_time = 0
        self.acc_on_bus_time = 0

        # run simulation until the first event
        while not self.ready:
            self.env.step()
        self.ready = False

        self.observation_space = Box(low=-1e4, high=1e4, shape=(96,))
        self.h_action_space = Discrete(3)
        self.l_action_space = Box(low=0, high=THRESHOLD, shape=(2,))
        self.action_space = Tuple((self.h_action_space, self.l_action_space))


    def reset(self):
        self.env = simpy.Environment()
        self.passengers = []
        self.stations = [Station(self, self.env, i, self.pax_alight[i], self.pax_board[i]) for i in range(N_STATION)]
        self.arange_stations()
        self.buses = [Bus(self, self.env, i, BUS_SCHEDULE[i]) for i in range(N_BUS)]
        self.ready = False
        self.action = None
        self.departure_times = []

        self.acc_waiting_time = 0
        self.acc_on_bus_time = 0

        # run simulation until the first event
        while not self.ready:
            self.env.step()
        self.ready = False

        self.last_timestep = self.env.now
        obs = self.get_observation()
        return obs

    def arange_stations(self) -> None:
        for index, station in enumerate(self.stations):
            if index == len(self.stations)-1:
                station.set_next(self.stations[0])
            else:
                station.set_next(self.stations[index+1])
            
            if index == 0:
                station.set_last(self.stations[-1])
            else:
                station.set_last(self.stations[index-1])

            station.set_opposite(self.stations[len(self.stations) - index - 1])

    def step(self, action):
        self.action = action
        while not self.ready:
            self.env.step()
        # print("Environment Step")
        self.ready = False
        
        timestep = self.env.now - self.last_timestep
        self.last_timestep = self.env.now

        obs = self.get_observation()
        rewards = self.get_reward()
        done = self.env.peek() >= HORIZON - 1000
        info = {'timestep': timestep}
        
        return obs, rewards, done, info

    def action_masks(self):
        return [True, self.allow_skipping, self.allow_turning_around]

    @staticmethod
    def extract_observations(obs):
        out = np.zeros((len(obs), FEATURE_DIM))
        index = 0
        for i, ob in obs:
            if ob['h_action'] is None:
                h_action = [0, 0]
            elif ob['h_action'] == 0:
                h_action = [1, 0]
            else:
                h_action = [0, 1]
            action_duration = 0 if ob['action_duration'] is None else ob['action_duration']
            l_action = 0 if ob['l_action'] is None else ob['l_action']
            processed_ob = [ob['ego'], ob['location'], ob['headway'], ob['pax'], action_duration] + h_action + [l_action]
            out[i] = processed_ob
            if ob['ego'] == 1:
                index = i
        out = np.roll(out, -index, axis=0)
        return out


    def get_observation(self):
        def getHeadWay(self, loc1, loc2):
            t = int(self.env.now // TRAVEL_TIME_STEP)
            station1 = int(loc1 // STATION_DIST)
            station2 = int(loc2 // STATION_DIST)
            if station2 != station1:
                if station2 > station1:
                    headway = sum([self.travel_times[station1, int(self.env.now // TRAVEL_TIME_STEP)] for station1 in range(station1, station2)])
                else:
                    headway = sum([self.travel_times[station1, int(self.env.now // TRAVEL_TIME_STEP)] for station1 in chain(range(station1, N_STATION), range(0, station2))])
                headway += (loc2 - station2 * STATION_DIST) / STATION_DIST * self.travel_times[station2, int(self.env.now // TRAVEL_TIME_STEP)]
                headway += ((station1 + 1) * STATION_DIST - loc1) / STATION_DIST * self.travel_times[station1, int(self.env.now // TRAVEL_TIME_STEP)]
            else:
                headway = (loc2 - loc1) / STATION_DIST * self.travel_times[station1, int(self.env.now // TRAVEL_TIME_STEP)]
            return headway
            
        obs = {}
        for bus in self.buses:
            ob = bus.get_observation()
            if bus.idx == self.acting_bus:
                ob['ego'] = 1
            else:
                ob['ego'] = 0
            obs[bus.idx] = ob
        obs = sorted(obs.items(), key=lambda x: x[1]['location'])
        for index, (bus, ob) in enumerate(obs):
            ob['headway'] = getHeadWay(self, ob['location'], obs[(index+1)%len(obs)][1]['location'])
        
        obs = self.extract_observations(obs)
        obs = obs.reshape((-1)) # flatten the observation (96,)
        return obs


    def get_reward(self):
        alpha, beta = 3, 1
        waiting_time = 0
        on_bus_time = 0

        for pax in self.passengers:
            if pax.status == 0:
                waiting_time += max(0, self.env.now - pax.last_time)
            elif pax.status == 1:
                on_bus_time += max(0, self.env.now - pax.last_time)
            if pax.last_time < self.env.now:
                pax.last_time = self.env.now
                pax.status = pax.new_status
        # print('total num of pax: ', len([pax for pax in self.passengers if pax.start_time < self.env.now and pax.status != 2]))
        # print('total num of pax on bus: ', len([pax for pax in self.passengers if pax.status == 1]))    
        reward = alpha * waiting_time + beta * on_bus_time
        self.acc_waiting_time += waiting_time
        self.acc_on_bus_time += on_bus_time
        return -reward

    def get_travel_time(self, station1):
        return self.travel_times[station1.idx, int(self.env.now // TRAVEL_TIME_STEP)]


@dataclass
class Passenger:
    start_time: float
    alight_station: int
    last_time: float
    bus: Bus = None
    status: int = 0 # 0: waiting, 1: on bus, 2: alighted
    new_status: int = 0 # 0: waiting, 1: on bus, 2: alighted



if __name__ == '__main__':
    env = Env()
    action = env.action_space.sample() 
    total_reward = 0
    while env.env.peek() < 10700:
        obs, rew, done, info = env.step(action) 
        r = random.random()
        total_reward += rew
        # if r < 0.1 and env.allow_skipping:
        #     action = (1, 0)
        # elif r < 0.2:
        #     action = (2, 1)
        # else:
        #     action = policy(obs)
        action = env.action_space.sample()   
        # print(f'Current time: {env.env.now}')
    pickle.dump(data, open('data.pkl', 'wb'))
    print(env.departure_times)
    print('Total waiting time: ', env.acc_waiting_time)
    print('Total on bus time: ', env.acc_on_bus_time)
    print('stops allowde to skip: ', num_skipping_stop, ' ', num_total_stop)
    print('Total reward: ', total_reward)