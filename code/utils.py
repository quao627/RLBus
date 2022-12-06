import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

seed=0
np.random.seed(seed)

# number of bus
N_BUS = 6

# number of stations
from params import N_STATION, CAPACITY

# time horizon
horizon = 3 * 60 * 60 # 3 hours
travel_time_step = 5 * 60 # travel times changes every 5 minutes


# passenger arrival and alight rate
avg_pax_arrive_rate = 3 # p (sec/pax)
std_pax_arrive_rate = 5

# pax_arrive_rate = [1/avg_pax_arrive_rate] * N_STATION # lambda pax/sec
pax_arrive_rate = [(np.exp(-0.5*((i-int(N_STATION)*0.5)/std_pax_arrive_rate)**2)* 1/ avg_pax_arrive_rate) for i in range(N_STATION)] # lambda pax/sec
# pax_arrive_rate = [1/(np.exp(-0.5*((i-int(N_STATION)*0.5)/std_pax_arrive_rate)**2)* avg_pax_arrive_rate) for i in range(N_STATION)] # lambda pax/sec

pax_avg_inveh_time = 5 * 60 # 5 minutes

# total number of passengers per each station
n_passenger = [horizon * pax_arrive_rate[i] for i in range(N_STATION)]

# average travel time to scale the travel time table
avg_travel_time = 4*60 # 4 minutes
min_travel_time = 2*60 # 2 minutes
max_travel_time = 10*60 # 10 minutes

# creating travel time table with poisson distribution
def gen_travel_time_table(n_station, horizon, travel_time_step, sigma=10, poi_lam=0.5, seed=0):
    np.random.seed(seed)
    tt_table = np.zeros((n_station, int(horizon/travel_time_step)))
    for i in range(n_station):
        for j in range(int(horizon/travel_time_step)):
            tt_table[i][j] = avg_travel_time * (np.exp(-0.5*((j-int(horizon/travel_time_step)*0.5)/sigma)**2) + np.random.poisson(poi_lam)*0.5)
    tt_table = np.clip(tt_table, min_travel_time, max_travel_time)
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
    pax_arrive_table = np.zeros((n_station, int(n_passenger)))
    for i in range(n_station):
        arrival_time = 0
        j = 0
        while arrival_time < horizon:
            arrival_time += np.random.exponential(1/pax_arrive_rate[i])
            pax_arrive_table[i][j] = arrival_time
            j += 1
    pax_arrive_table[pax_arrive_table == 0] = np.inf
    return pax_arrive_table
pax_arrive_table = gen_pax_arrive(N_STATION, horizon, 5000, pax_arrive_rate)

# generating table for number of alighting passengers at each station at each time step
# i -> for each station
# j -> 5 minutes time step
# value -> 1/(N_STATION-1) 
def gen_pax_alight(n_station, horizon, n_passenger, pax_arrive_table, seed=0, pax_avg_inveh_time=5*60, sigma=60):
    np.random.seed(seed)
    pax_alight_table = np.zeros((n_station, int(horizon/travel_time_step)))
    for i in range(n_station):
        for j in range(int(horizon/travel_time_step)):
            pax_alight_table[i][j] += 1/(n_station-i)
    return pax_alight_table
pax_alight_table = gen_pax_alight(N_STATION, horizon, 5000, pax_arrive_table)

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