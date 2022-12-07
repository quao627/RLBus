from typing import List, Dict, Union
from dataclasses import dataclass
import pickle
import json

import gym
import simpy
from simpy.events import AnyOf, AllOf, Event

from utils import *


data = {station_idx: [] for station_idx in range(N_STATION)}

class Bus:
    def __init__(self, env, simpy_env, name, starting_time) -> None:
        self.env = env
        self.simpy_env = simpy_env
        self.name = name
        self.capacity = CAPACITY
        self.cur_station = self.env.stations[0]
        self.next_station = self.cur_station.get_next()
        self.next_travel_time = self.env.get_travel_time(self.cur_station)
        self.starting_time = starting_time
        print(f'Bus {name} is successfully initialized.')
        self.proc = self.simpy_env.process(self.drive())
        self.passengers = []
        self.num_pax = 0
        self.state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []

    def drive(self):
        yield self.simpy_env.timeout(self.starting_time)
        data[self.cur_station.idx].append(self.simpy_env.now)
        print(f'Bus {self.name} starts at {self.simpy_env.now}')
        while True:
            turn_around = False

            # drive till the next station
            yield self.simpy_env.timeout(self.next_travel_time)

            # request to enter the station
            with self.next_station.request() as req:
                yield req
                print(f'Bus {self.name} Arrives At Station {self.next_station.idx}')

                self.env.ready = True
                yield self.simpy_env.timeout(0)

                # high-level and low-level actions
                h_action, l_action = self.env.action
                
                # 0 means holding
                if h_action == 0:
                    data[self.next_station.idx].append(self.simpy_env.now)
                    pax_alight = self.alight_pax()
                    pax_board = self.next_station.board_pax(self)
                    holding_time = max(pax_board * t_b, pax_alight * t_a)
                    holding_time += l_action
                    yield self.simpy_env.timeout(holding_time)
                    self.update_state(h_action)
                    print(f'Bus {self.name} holds at station {self.cur_station.idx} for {l_action} seconds')
                
                # 1 means skipping
                elif h_action == 1:
                    yield self.simpy_env.timeout(0)
                    self.update_state(h_action)
                    print(f'Bus {self.name} skips station {self.cur_station}')

                # 2 means turning around
                else: 
                    turn_around = True

            if turn_around:
                # alight all passengers
                pax_alight = self.alight_all_pax(self.next_station)
                holding_time = pax_alight * t_a
                holding_time += l_action

                # wait to turn around
                yield self.simpy_env.timeout(holding_time)

                # turn around
                self.update_state(h_action)
                print(f'Bus {self.name} turns around at station {self.cur_station.idx} to {self.next_station} for {l_action} seconds')
        

    def alight_all_pax(self, station):
        for pax in self.passengers:
            pax.bus = None
        station.passengers.extend(self.passengers)
        station.passengers.sort(key=lambda x: x.arrival_time)
        self.passengers = []
        self.num_pax = 0


    def alight_pax(self):
        pax_alight = 0
        passengers = []
        for pax in self.passengers:
            if pax.alight_station == self.next_station.idx:
                pax_alight += 1
            else:
                passengers.append(pax)
        self.passengers = passengers
        self.num_pax = len(passengers)
        return pax_alight

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
                 simpy_env, 
                 idx,
                 pax_alight,
                 pax_board) -> None:
        self.simpy_env = simpy_env
        self.resource = simpy.Resource(simpy_env, capacity=1)
        self.idx = idx
        self.last_station = None
        self.next_station = None
        self.opposite_station = None
        self.pax_alight = pax_alight
        self.pax_board = pax_board
        self.passengers = self.generate_pax(self.pax_alight, self.pax_board)
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
    
    def board_pax(self, bus):
        """
        1, move the passengers to the bus
        2, return the number of passengers that are boarding
        3, update the arrival time of the last bus
        4, remove the passengers that are boarding from the station

        @param threshold: the maximum number of passengers that can board the bus
        
        @return: the number of passengers that are boarding
        """
        STOP_BOARDING = False
        passengers = []
        n = 0
        for pax in self.passengers:
            if pax.start_time < self.simpy_env.now and pax.bus is None and not STOP_BOARDING:
                pax.bus = bus.name
                bus.passengers.append(pax)
                bus.num_pax += 1
                n += 1
                if bus.num_pax == bus.capacity:
                    STOP_BOARDING = True
            else:
                passengers.append(pax)
                
        self.last_arrival_time = self.simpy_env.now
        self.passengers = passengers
        return n

    def generate_pax(self, pax_alight, pax_board):
        return [Passenger(pax_board[i], pax_alight[i]) for i in range((pax_board != np.inf).sum())] 

@dataclass
class Passenger:
    start_time: float
    alight_station: int
    bus: Bus = None

class Env:
    def __init__(self) -> None:
        self.travel_times = TABLE_TRAVEL_TIME
        self.pax_alight = PAX_ALIGHT_TABLE
        self.pax_board = PAX_ARRIVE_TABLE

        self.env = simpy.Environment()
        self.stations = [Station(self.env, i, self.pax_alight[i], self.pax_board[i]) for i in range(N_STATION)]
        self.arange_stations()
        self.buses = [Bus(self, self.env, i, BUS_SCHEDULE[i]) for i in range(8)]
        self.ready = False
        self.action = None
    
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
        return None

    def get_travel_time(self, station1):
        return self.travel_times[station1.idx, int(self.env.now // TRAVEL_TIME_STEP)]

def policy(obs):
    return (0, 0)

if __name__ == '__main__':
    env = Env()
    action = (0, 0)
    while env.env.now < 10800:
        obs = env.step(action)
        action = policy(obs)
        print(f'Current time: {env.env.now}')
    pickle.dump(data, open('data.pkl', 'wb'))