import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# number of bus
N_BUS = 6

# number of stations
from params import N_STATION, CAPACITY

# time horizon
horizon = 3 * 60 * 60 # 3 hours
travel_time_step = 5 * 60 # travel times changes every 5 minutes

# total number of passengers
n_passenger = 100

# passenger arrival and alight rate
pax_arrive_rate = n_passenger/horizon*3600
pax_avg_inveh_time = 5*60 # 5 minutes

# creating travel time table with poisson distribution
def gen_travel_time_table(n_station, horizon, travel_time_step, sigma=10, poi_lam=0.5, seed=0):
    np.random.seed(seed)
    tt_table = np.zeros((n_station, int(horizon/travel_time_step)))
    for i in range(n_station):
        for j in range(int(horizon/travel_time_step)):
            tt_table[i][j] = np.exp(-0.5*((j-int(horizon/travel_time_step)*0.5)/sigma)**2) + np.random.poisson(poi_lam)*0.5
    return tt_table

table_travel_time = gen_travel_time_table(N_STATION, horizon, travel_time_step)

plt.figure(0)
plt.xlabel("Time (5 min)")
plt.ylabel("Travel time")
plt.plot(np.mean(table_travel_time, axis=0))

plt.figure(1)
plt.xlabel("Bus stations")
plt.ylabel("Travel time")
plt.plot(np.mean(table_travel_time, axis=1))

# generating table for number of arrival passengers at each station at each time step
# i -> for each station
# j -> the number of passsengers
# value -> the time step the passenger arrives at the station i (based on Poisson distribution)
def gen_pax_arrive(n_station, horizon, n_passenger, pax_arrive_rate, seed=0):
    np.random.seed(seed)
    pax_arrive_table = np.zeros((n_station, n_passenger))
    for j in range(n_passenger):
        i = np.random.randint(n_station)
        if j == 0:
            pax_arrive_table[i][j] += np.random.poisson(pax_arrive_rate)
        else:
            pax_arrive_table[i][j] += np.sum(pax_arrive_table.transpose()[j-1]) + np.random.poisson(pax_arrive_rate)
        if pax_arrive_table[i][j]>horizon:
            pax_arrive_table[i][j] = horizon
    return pax_arrive_table
pax_arrive_table = gen_pax_arrive(N_STATION, horizon, n_passenger, pax_arrive_rate)

# generating table for number of alighting passengers at each station at each time step
# i -> for each station
# j -> the number of passsengers
# value -> the time step the passenger alighting at the station i (based on Poisson distribution)
def gen_pax_alight(n_station, horizon, pax_arrive_table, seed=0, pax_avg_inveh_time=5*60, sigma=60):
    np.random.seed(seed)
    pax_alight_table = np.zeros((n_station, n_passenger))
    for j in range(n_passenger):
        board_station = np.nonzero(pax_arrive_table.transpose()[10])[0].item()
        i = np.random.randint(board_station, n_station)
        pax_alight_table[i][j] += np.sum(pax_arrive_table.transpose()[j]) + np.random.normal(pax_avg_inveh_time, sigma)
        if pax_alight_table[i][j]>horizon:
            pax_alight_table[i][j] = horizon
    return pax_alight_table
pax_alight_table = gen_pax_alight(N_STATION, horizon, pax_arrive_table)

sns.heatmap(pax_arrive_table)
sns.heatmap(pax_alight_table)

# make the schedule of bus departing from the terminal
def gen_bus_schedule(n_bus, horizon, seed=0):
    np.random.seed(seed)
    bus_schedule = np.zeros(n_bus)
    bus_schedule[0] = 0
    for i in range(n_bus-1):
        bus_schedule[i+1] = min(horizon, bus_schedule[i]+np.random.normal(horizon/n_bus, 1))
        
    return bus_schedule

bus_schedule = gen_bus_schedule(N_BUS, horizon)