import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

seed=0
np.random.seed(seed)

# number of bus
N_BUS = 7

# feature dimension
FEATURE_DIM = 2

# capacity of bus
CAPACITY = 60

# planned headway
HEADWAY = 2 * 60

# Threshold for low-level action
THRESHOLD = 3 * 60

# distance between stations
STATION_DIST = 1 # km

# maximum time a passenger will wait in the system
MAX_WAITING_TIME = 30 * 60

# number of stations
N_STATION = 10

# time HORIZON
HORIZON = 6 * 60 * 60 # 3 hours
TRAVEL_TIME_STEP = 5 * 60 # travel times changes every 5 minutes

# time for alight & board
t_a = 1.8 # time for alighting
t_b = 3   # time for boarding

# passenger arrival and alight rate
AVG_PAX_ARRIVAL_RATE = 30 # p (sec/pax)
STD_PAX_ARRIVAL_RATE = 10


PAX_ARRIVAL_RATE = [(np.exp(-0.5*((i-int(N_STATION // 2)*0.5)/STD_PAX_ARRIVAL_RATE)**2)* 1/ AVG_PAX_ARRIVAL_RATE) for i in range(int(N_STATION // 2))] # lambda pax/sec
# PAX_ARRIVAL_RATE = [1/AVG_PAX_ARRIVAL_RATE] * N_STATION # lambda pax/sec

PAX_AVG_INVEH_TIME = 5 * 60 # 5 minutes

# total number of passengers per each station
n_passenger = [HORIZON * PAX_ARRIVAL_RATE[i] for i in range(int(N_STATION // 2))]

# average travel time to scale the travel time table
avg_travel_time = 4*60 # 4 minutes
min_travel_time = 2*60 # 2 minutes
max_travel_time = 10*60 # 10 minutes

# creating travel time table with poisson distribution
def gen_travel_time_table(n_station, HORIZON, TRAVEL_TIME_STEP, sigma=20, seed=0):
    n_station = int(n_station // 2)
    def gen_table(seed):
        np.random.seed(seed)
        tt_table = np.zeros((n_station, int(HORIZON/TRAVEL_TIME_STEP)))
        for i in range(n_station):
            for j in range(int(HORIZON/TRAVEL_TIME_STEP)):
                tt_table[i][j] = avg_travel_time * (np.exp(-0.5*((j-int(HORIZON/TRAVEL_TIME_STEP)*0.5)/sigma)**2) + np.random.normal(0, 0.15))
        tt_table = np.clip(tt_table, min_travel_time, max_travel_time)
        return tt_table
    tt_table_1 = gen_table(123)
    tt_table_2 = gen_table(42)[::-1, :]
    tt_table = np.concatenate([tt_table_1, tt_table_2], axis=0)
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
    n_station = int(n_station // 2)
    def gen_table(seed):
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
    pax_arrive_table_1 = gen_table(123)
    pax_arrive_table_2 = gen_table(42)[::-1, :][list(range(1, n_station)) + [0]]
    pax_arrive_table = np.concatenate([pax_arrive_table_1, pax_arrive_table_2], axis=0)
    for i in range(pax_arrive_table.shape[0]):
        pax_arrive_table[i, :] = pax_arrive_table[i, :] + i * HEADWAY
    return pax_arrive_table

# generating table for number of alighting passengers at each station at each time step
# i -> for each station
# j -> 5 minutes time step
# value -> 1/(N_STATION-1) 
# def gen_pax_alight(n_station, HORIZON, n_passenger, pax_arrive_table, seed=0, PAX_AVG_INVEH_TIME=5*60, sigma=60):
#     np.random.seed(seed)
#     pax_alight_table = np.zeros((n_station, int(HORIZON/TRAVEL_TIME_STEP)))
#     for i in range(n_station):
#         for j in range(int(HORIZON/TRAVEL_TIME_STEP)):
#             pax_alight_table[i][j] += 1/(n_station-i)
#     pax_alight_table[0, :] = 0
#     return pax_alight_table

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
# BUS_SCHEDULE = {i: [i*HEADWAY + N_BUS * HEADWAY * k for k in range(10)] for i in range(N_BUS)}
TABLE_TRAVEL_TIME = gen_travel_time_table(N_STATION, HORIZON, TRAVEL_TIME_STEP)
TABLE_TRAVEL_TIME = np.ones_like(TABLE_TRAVEL_TIME) * avg_travel_time
# PAX_ARRIVE_TABLE = gen_pax_arrive(N_STATION, HORIZON, 5000, PAX_ARRIVAL_RATE)
PAX_ARRIVE_TABLE = np.zeros((int(N_STATION // 2), 5000))
PAX_ARRIVE_TABLE[:, :] = np.inf
for i, arrival_rate in enumerate(AVG_PAX_ARRIVAL_RATE * np.ones(shape=(int(N_STATION // 2)-1))):
    data = np.array(np.arange(0, HORIZON, arrival_rate))
    PAX_ARRIVE_TABLE[i, :len(data)] = data
PAX_ARRIVE_TABLE = np.tile(PAX_ARRIVE_TABLE, (2, 1))
PAX_ARRIVE_TABLE[-1, :] = np.inf
PAX_ARRIVE_TABLE[int(N_STATION // 2)-1, :] = np.inf
for i in range(PAX_ARRIVE_TABLE.shape[0]):
    PAX_ARRIVE_TABLE[i, :] = PAX_ARRIVE_TABLE[i, :] + i * HEADWAY

# PAX_ALIGHT_TABLE = gen_pax_alight(N_STATION, HORIZON, 5000, PAX_ARRIVE_TABLE)
def gen_pax_alight(pax_arrive_table):
    np.random.seed(seed)
    pax_alight_table = np.zeros_like(pax_arrive_table)
    n_stations = pax_arrive_table.shape[0]
    for i in range(int(n_stations // 2) - 1):
        prob = np.arange(int(n_stations // 2)+n_stations//2, i+1+n_stations//2, -1)
        prob = prob / prob.sum()
        if prob.sum() < 1:
            prob[-1] += 1 - prob.sum()
        alight_stations = np.random.choice(np.arange(i+1, int(n_stations // 2)), size=(pax_arrive_table[i] < np.inf).sum(), p=prob)
        pax_alight_table[i, pax_arrive_table[i] < np.inf] = alight_stations
    for i in range(int(n_stations // 2), n_stations-1):
        prob = np.arange(n_stations, i+1, -1)
        prob = prob / prob.sum()
        if prob.sum() < 1:
            prob[-1] += 1 - prob.sum()
        alight_stations = np.random.choice(np.arange(i+1, n_stations), size=(pax_arrive_table[i] < np.inf).sum(), p=prob)
        pax_alight_table[i, pax_arrive_table[i] < np.inf] = alight_stations
    pax_alight_table = np.where(pax_alight_table == 0, np.inf, pax_alight_table)
    return pax_alight_table

# def gen_pax_alight(pax_arrive_table):
#     all_index = np.where(pax_arrive_table < np.inf)
#     n_pax = int(len(all_index[0]) // (N_STATION - 2))
#     all_index = np.stack(all_index, axis=1)
#     PAX_ALIGHT_TABLE = np.zeros_like(pax_arrive_table)
#     for i in range(1, int(N_STATION // 2)):
#         idx = np.random.randint((all_index[:, 0] < i).sum(), size=n_pax)
#         PAX_ALIGHT_TABLE[all_index[idx][:, 0], all_index[idx][:, 1]] = i
#         mask = np.ones(len(all_index), dtype=bool) 
#         mask[idx] = False
#         all_index = all_index[mask]
#     mask = all_index[:, 0] < int(N_STATION // 2)
#     # PAX_ALIGHT_TABLE[all_index[mask][:, 0], all_index[mask][:, 1]] = int(N_STATION // 2) - 1
#     # all_index = all_index[~mask]
#     for i in range(int(N_STATION // 2) + 1, N_STATION):
#         idx = np.random.randint((all_index[:, 0] < i).sum(), size=n_pax)
#         PAX_ALIGHT_TABLE[all_index[idx][:, 0], all_index[idx][:, 1]] = i
#         mask = np.ones(len(all_index), dtype=bool) 
#         mask[idx] = False
#         all_index = all_index[mask]
#     # PAX_ALIGHT_TABLE[all_index[:, 0], all_index[:, 1]] = N_STATION - 1
#     return PAX_ALIGHT_TABLE

PAX_ALIGHT_TABLE = gen_pax_alight(PAX_ARRIVE_TABLE)
