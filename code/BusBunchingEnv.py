from typing import List, Dict, Union

import gym
import simpy
from simpy.events import AnyOf, AllOf, Event

from params import *

class Bus:
    def __init__(self, env, name, starting_time) -> None:
        self.env = env
        self.name = name
        self.capacity = CAPACITY
        self.cur_station = self.env.stations[0]
        self.next_station = self.cur_station.get_next()
        yield env.timeout(starting_time)
        print(f'Bus starts at {env.now}')

    def drive(self):
        while True:
            # drive till the next station
            yield self.env.timeout(self.next_travel_time)

            # request to enter the station
            with self.next_station.request() as req:
                yield req
                print(f'Bus Arrives At Station {self.next_station.name}')

                stopping_time = self.take_action(self)
                yield self.env.timeout(stopping_time)
            
            self.update_state(self)

    def take_action(self):
        # decide holding time or skipping
        return 1

    def update_state(self):
        self.cur_station = self.next_station
        self.next_station = self.cur_station.get_next()


class Station:
    def __init__(self, 
                 simpy_env, 
                 arrival_rate: float,
                 alight_ratio: float) -> None:
        self.resource = simpy.Resource(simpy_env, capacity=1)
        self.last_station = None
        self.next_station = None

    @classmethod
    def set_last(self, station):
        self.last_station = station

    @classmethod
    def set_next(self, station):
        self.next_station = station

    @classmethod
    def get_last(self):
        return self.last_station

    @classmethod
    def get_next(self):
        return self.next_station

    @classmethod
    def request(self):
        return self.resource.request()

class Env:
    def __init__(self, buses) -> None:
        #self.stations = [Station(self, arrival_rate, alight_ratio) for (arrival_rate, alight_ratio) in zip(ARRIVAL_RATES, ALIGHT_RATIOS)]
        self.stations = [Station(self, 5, 5) for i in range(10)]
        self.travel_times = TRAVEL_TIMES
    
    def arange_stations(self) -> None:
        for index, station in enumerate(self.stations):
            if index == len(self.stations):
                station.set_next(self.stations[0])
            else:
                station.set_next(self.stations[index+1])
            
            if index == 0:
                station.set_last(self.stations[-1])
            else:
                station.set_last(self.stations[index-1])

    def step(self, action):
        pass

if __name__ == '__main__':
    env = simpy.Environment()
