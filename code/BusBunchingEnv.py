from typing import List, Dict, Union

import gym
import simpy
from simpy.events import AnyOf, AllOf, Event

from params import *

class Bus:
    def __init__(self, env, simpy_env, name, starting_time) -> None:
        self.env = env
        self.simpy_env = simpy_env
        self.name = name
        self.capacity = CAPACITY
        self.cur_station = self.env.stations[0]
        self.next_station = self.cur_station.get_next()
        self.next_travel_time = self.env.get_travel_time(self.cur_station, self.next_station, self.simpy_env.now)
        self.starting_time = starting_time
        print(f'Bus {name} is successfully initialized.')
        self.proc = self.simpy_env.process(self.drive())

    def drive(self):
        yield self.simpy_env.timeout(self.starting_time)
        print(f'Bus {self.name} starts at {self.simpy_env.now}')
        while True:
            # drive till the next station
            yield self.simpy_env.timeout(self.next_travel_time)

            # request to enter the station
            with self.next_station.request() as req:
                yield req
                print(f'Bus {self.name} Arrives At Station {self.next_station.name}')

                self.env.ready = True
                yield self.simpy_env.timeout(0)

                stopping_time = self.take_action()
                yield self.simpy_env.timeout(stopping_time)
                print(f'Bus {self.name} holds for {stopping_time} seconds')
            
            self.update_state()

    def take_action(self):
        # decide holding time or skipping
        return self.env.action

    def update_state(self):
        self.cur_station = self.next_station
        self.next_station = self.cur_station.get_next()
        self.next_travel_time = self.env.get_travel_time(self.cur_station, self.next_station, self.simpy_env.now)

class Station:
    def __init__(self, 
                 simpy_env, 
                 name,
                 arrival_rate: float,
                 alight_ratio: float) -> None:
        self.resource = simpy.Resource(simpy_env, capacity=1)
        self.name = name
        self.last_station = None
        self.next_station = None

    def set_last(self, station):
        self.last_station = station

    def set_next(self, station):
        self.next_station = station

    def get_last(self):
        return self.last_station

    def get_next(self):
        return self.next_station

    def request(self):
        return self.resource.request()

class Env:
    def __init__(self) -> None:
        #self.stations = [Station(self, arrival_rate, alight_ratio) for (arrival_rate, alight_ratio) in zip(ARRIVAL_RATES, ALIGHT_RATIOS)]
        self.travel_times = TRAVEL_TIMES
        self.env = simpy.Environment()
        self.stations = [Station(self.env, i, 5, 5) for i in range(10)]
        self.arange_stations()
        self.buses = [Bus(self, self.env, i, i*5) for i in range(8)]
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

    def step(self, action):
        self.action = action
        while not self.ready:
            self.env.step()
        print("Environment Step")
        self.ready = False
        return None

    @staticmethod
    def get_travel_time(station1, station2, t):
        return 10

def policy(obs):
    return 1

if __name__ == '__main__':
    env = Env()
    action = 1
    while env.env.now < 100:
        obs = env.step(action)
        action = policy(obs)
        print(f'Current time: {env.env.now}')