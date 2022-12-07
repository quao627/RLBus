import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

seed=0
np.random.seed(seed)

# number of bus
N_BUS = 6

# capacity of bus
CAPACITY = 60

# planned headway
HEADWAY = 6 * 60

# number of stations
N_STATION = 10

# time HORIZON
HORIZON = 3 * 60 * 60 # 3 hours
TRAVEL_TIME_STEP = 5 * 60 # travel times changes every 5 minutes

# time for alight & board
t_a = 1.8 # time for alighting
t_b = 3   # time for boarding

# passenger arrival and alight rate
AVG_PAX_ARRIVAL_RATE = 3 # p (sec/pax)
STD_PAX_ARRIVAL_RATE = 5


PAX_ARRIVAL_RATE = [(np.exp(-0.5*((i-int(N_STATION)*0.5)/STD_PAX_ARRIVAL_RATE)**2)* 1/ AVG_PAX_ARRIVAL_RATE) for i in range(N_STATION)] # lambda pax/sec
# PAX_ARRIVAL_RATE = [1/AVG_PAX_ARRIVAL_RATE] * N_STATION # lambda pax/sec

PAX_AVG_INVEH_TIME = 5 * 60 # 5 minutes

# total number of passengers per each station
n_passenger = [HORIZON * PAX_ARRIVAL_RATE[i] for i in range(N_STATION)]

# average travel time to scale the travel time table
avg_travel_time = 4*60 # 4 minutes
min_travel_time = 2*60 # 2 minutes
max_travel_time = 10*60 # 10 minutes

# creating travel time table with poisson distribution
def gen_travel_time_table(n_station, HORIZON, TRAVEL_TIME_STEP, sigma=10, poi_lam=0.5, seed=0):
    np.random.seed(seed)
    tt_table = np.zeros((n_station, int(HORIZON/TRAVEL_TIME_STEP)))
    for i in range(n_station):
        for j in range(int(HORIZON/TRAVEL_TIME_STEP)):
            tt_table[i][j] = avg_travel_time * (np.exp(-0.5*((j-int(HORIZON/TRAVEL_TIME_STEP)*0.5)/sigma)**2) + np.random.poisson(poi_lam)*0.5)
    tt_table = np.clip(tt_table, min_travel_time, max_travel_time)
    return tt_table

# plt.figure(0)
# plt.xlabel("Time (5 min)")
# plt.ylabel("Travel time")
# plt.plot(np.mean(table_travel_time, axis=0))

# plt.figure(1)
# plt.xlabel("Bus stations")
# plt.ylabel("Travel time")
# plt.plot(np.mean(table_travel_time, axis=1))

# generating table for number of arrival passengers at each station at each time step
# i -> for each station
# j -> the number of passsengers
# value -> the time step the passenger arrives at the station i (based on Poisson distribution)
def gen_pax_arrive(n_station, HORIZON, n_passenger, PAX_ARRIVAL_RATE, seed=0):
    np.random.seed(seed)
    pax_arrive_table = np.zeros((n_station, int(n_passenger)))
    for i in range(n_station):
        arrival_time = 0
        j = 0
        while arrival_time < HORIZON:
            arrival_time += np.random.exponential(1/PAX_ARRIVAL_RATE[i])
            pax_arrive_table[i][j] = arrival_time
            j += 1
    pax_arrive_table[pax_arrive_table == 0] = np.inf
    pax_arrive_table[-1, :] = np.inf
    return pax_arrive_table

# generating table for number of alighting passengers at each station at each time step
# i -> for each station
# j -> 5 minutes time step
# value -> 1/(N_STATION-1) 
def gen_pax_alight(n_station, HORIZON, n_passenger, pax_arrive_table, seed=0, PAX_AVG_INVEH_TIME=5*60, sigma=60):
    np.random.seed(seed)
    pax_alight_table = np.zeros((n_station, int(HORIZON/TRAVEL_TIME_STEP)))
    for i in range(n_station):
        for j in range(int(HORIZON/TRAVEL_TIME_STEP)):
            pax_alight_table[i][j] += 1/(n_station-i)
    pax_alight_table[0, :] = 0
    return pax_alight_table

# sns.heatmap(pax_arrive_table)
# sns.heatmap(pax_alight_table)

# make the schedule of bus departing from the terminal
def gen_bus_schedule(n_bus, HORIZON, seed=0):
    np.random.seed(seed)
    bus_schedule = list()
    bus_schedule.append(HEADWAY)
    departure_time = HEADWAY
    while True:
        departure_time += HEADWAY
        if departure_time > HORIZON:
            break
        bus_schedule.append(departure_time)
    return np.array(bus_schedule)

BUS_SCHEDULE = gen_bus_schedule(N_BUS, HORIZON)
TABLE_TRAVEL_TIME = gen_travel_time_table(N_STATION, HORIZON, TRAVEL_TIME_STEP)
PAX_ARRIVE_TABLE = gen_pax_arrive(N_STATION, HORIZON, 5000, PAX_ARRIVAL_RATE)
# PAX_ALIGHT_TABLE = gen_pax_alight(N_STATION, HORIZON, 5000, PAX_ARRIVE_TABLE)
def gen_pax_alight(pax_arrive_table):
    np.random.seed(seed)
    pax_alight_table = np.zeros_like(pax_arrive_table)
    n_stations = pax_arrive_table.shape[0]
    for i in range(n_stations - 1):
        alight_stations = np.random.randint(low=i+2, high=n_stations+1, size=(pax_arrive_table[i] < np.inf).sum())
        pax_alight_table[i, pax_arrive_table[i] < np.inf] = alight_stations
    return pax_alight_table

PAX_ALIGHT_TABLE = gen_pax_alight(PAX_ARRIVE_TABLE)