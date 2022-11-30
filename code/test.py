import simpy
from simpy.events import AnyOf, AllOf, Event

class car:
    def __init__(self, env, name, bcs, driving_time, charge_duration):
        self.env = env
        self.name = name
        self.bcs = bcs
        self.driving_time = driving_time
        self.charge_duration = charge_duration
        self.proc = env.process(self.drive())
        self.driving_times = 0

    # def run(self):
    #     while True:
    #         self.drive()

    def drive(self):
        name, bcs, driving_time, charge_duration = self.name, self.bcs, self.driving_time, self.charge_duration
        while True:
            # Simulate driving to the BCS
            yield env.timeout(driving_time)

            # Request one of its charging spots
            print('%s arriving at %d' % (name, env.now))
            with bcs.request() as req:
                yield req

                # Charge the battery
                print('%s starting to charge at %s' % (name, env.now))
                yield env.timeout(charge_duration)
                print('%s leaving the bcs at %s' % (name, env.now))
            self.driving_times += 1
        
env = simpy.Environment()
bcs = simpy.Resource(env, capacity=2)
cars = [car(env, 'Car %d' % i, bcs, i*2, 5) for i in range(4)]
# procs = [env.process(car) for car in cars]
# def run():
#     while True:
#         ret = AnyOf(env, procs)
#         print(ret)

# env.process(run(cars))

while env.peek() < 100:
    env.step()
    print(env.now)