from typing import List, Dict, Union
from dataclasses import dataclass
import pickle
import json
import random

import gym
import simpy
from simpy.events import AnyOf, AllOf, Event

from utils import *


'''
By 12/07:
Todos (Simulator):
- [✅] Shift passenger arrival time
- [✅] Buses have to alight all passengers at terminals 
- [✅] Make buses stick to the schedule at terminal
- [ ] Fill action, state, reward buffers
- [✅ ] Make sure the plots make sense (No gap between buses at the terminal)
- [ ] Test skipping & turning around actions

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
        self.next_station = self.cur_station.get_next()
        self.next_travel_time = self.env.get_travel_time(self.cur_station)
        self.starting_time = starting_time
        print(f'Bus {self.idx} is successfully initialized.')
        self.proc = self.simpy_env.process(self.drive())
        self.passengers = []
        self.num_pax = 0
        self.state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []

    def drive(self):
        # hold at the terminal before starting to drive
        yield self.simpy_env.timeout(self.starting_time)
        self.env.departure_times.append(self.simpy_env.now)
        data[self.cur_station.idx].append(self.simpy_env.now)
        print(f'Bus {self.idx} starts at {self.simpy_env.now}')

        # each cycle is a trip from one station to the next
        while True:
            turn_around = False

            # drive till the next station
            yield self.simpy_env.timeout(self.next_travel_time)
            if (self.next_station.idx == 0):
                if (self.simpy_env.now < self.env.departure_times[-1] + HEADWAY):
                    yield self.simpy_env.timeout(self.env.departure_times[-1] + HEADWAY - self.simpy_env.now)
                self.env.departure_times.append(self.simpy_env.now)

            # request to enter the station
            with self.next_station.request() as req:
                yield req
                print(f'Bus {self.idx} Arrives At Station {self.next_station.idx}')

                self.env.ready = True
                self.env.acting_bus = self.idx
                yield self.simpy_env.timeout(0)

                # high-level and low-level actions
                h_action, l_action = self.env.action
                
                # 0 means holding
                if h_action == 0 or self.next_station.idx in [N_STATION-1, int(N_STATION//2)-1]:
                    data[self.next_station.idx].append(self.simpy_env.now)
                    if self.next_station.idx in [N_STATION-1, int(N_STATION//2)-1]:
                        pax_alight = self.alight_all_pax(self.next_station)
                        pax_board = 0
                    else:
                        pax_alight = self.alight_pax(self.next_station)
                        pax_board = self.board_pax(self.next_station)
                    holding_time = max(pax_board * t_b, pax_alight * t_a)
                    holding_time += l_action
                    yield self.simpy_env.timeout(holding_time)
                    self.update_state(h_action)
                    print(f'Bus {self.idx} holds at station {self.cur_station.idx} for {holding_time} seconds')
                
                # 1 means skipping
                elif h_action == 1:
                    yield self.simpy_env.timeout(0)
                    self.update_state(h_action)
                    print(f'Bus {self.idx} skips station {self.cur_station.idx}')

                # 2 means turning around
                else: 
                    turn_around = True

            if turn_around:
                # alight all passengers
                pax_alight = self.alight_all_pax(self.next_station) # then alight all other passengers
                holding_time = pax_alight * t_a
                holding_time += l_action

                # wait to turn around
                yield self.simpy_env.timeout(holding_time)

                # turn around
                self.update_state(h_action)
                print(f'Bus {self.idx} turns around at station {self.cur_station.idx} to {self.next_station.idx} for {l_action} seconds')
        

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
        if h_action in [0, 1]:
            self.cur_station = self.next_station
            self.next_station = self.cur_station.get_next()
            self.next_travel_time = self.env.get_travel_time(self.cur_station)
        else:
            self.cur_station = self.next_station
            self.next_station = self.cur_station.get_opposite()
            self.next_travel_time = 0

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


class Env:
    def __init__(self) -> None:
        self.travel_times = TABLE_TRAVEL_TIME
        self.pax_alight = PAX_ALIGHT_TABLE
        self.pax_board = PAX_ARRIVE_TABLE
        self.env = simpy.Environment()
        self.ready = False
        self.action = None
        self.departure_times = []
        self.passengers = []

        self.stations = [Station(self, self.env, i, self.pax_alight[i], self.pax_board[i]) for i in range(N_STATION)]
        self.arange_stations()
        self.buses = [Bus(self, self.env, i, BUS_SCHEDULE[i]) for i in range(N_BUS)]
        

        self.acc_waiting_time = 0
        self.acc_on_bus_time = 0

        # run simulation until the first event
        while not self.ready:
            self.env.step()
        self.ready = False

        obs = self.get_observation()
        return obs

    def reset(self):
        self.env = simpy.Environment()
        self.stations = [Station(self.env, i, self.pax_alight[i], self.pax_board[i]) for i in range(N_STATION)]
        self.arange_stations()
        self.buses = [Bus(self, self.env, i, BUS_SCHEDULE[i]) for i in range(N_BUS)]
        self.ready = False
        self.action = None
        self.departure_times = []
        self.passengers = []

        self.acc_waiting_time = 0
        self.acc_on_bus_time = 0

        # run simulation until the first event
        while not self.ready:
            self.env.step()
        self.ready = False

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
        print("Environment Step")
        self.ready = False
        
        obs = self.get_observation()
        rewards = self.get_reward()
        done = False
        info = {}
        return obs, rewards, done, info

    def get_observation(self):
        pass

    def get_reward(self):
        alpha, beta = 1, 1
        waiting_time = 0
        on_bus_time = 0

        for pax in self.passengers:
            if pax.status == 0:
                waiting_time += self.env.now - pax.last_time
            elif pax.status == 1:
                on_bus_time += self.env.now - pax.last_time
            pax.last_time = self.env.now
            pax.status = pax.new_status
            
        reward = alpha * waiting_time + beta * on_bus_time
        self.acc_waiting_time += waiting_time
        self.acc_on_bus_time += on_bus_time
        return reward

    def get_travel_time(self, station1):
        return self.travel_times[station1.idx, int(self.env.now // TRAVEL_TIME_STEP)]

def policy(obs):
    return (0, 1)

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
    action = (0, 1)
    while env.env.peek() < 10700:
        obs = env.step(action)
        r = random.random()
        if r < 0.1:
            action = (1, 0)
        elif r < 0.2:
            action = (2, 1)
        else:
            action = policy(obs)     
        print(f'Current time: {env.env.now}')
    pickle.dump(data, open('data.pkl', 'wb'))
    print(env.departure_times)
    print('Total waiting time: ', env.acc_waiting_time)
    print('Total on bus time: ', env.acc_on_bus_time)