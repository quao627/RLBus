import argparse
import time
import warnings
import pickle
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument("--env", default="Busbunch-v0", help="environment ID : Moving-v0, Sliding-v0, Busbunch-v0")
parser.add_argument("--alg", default="HybridPPO", help="algorithm to use: HybridPPO | HybridTransformerPPO (not yet)")
parser.add_argument("--num-timesteps", type=int, default=1000000)
parser.add_argument("--lr", type=float, default=0.00025, help="learning rate for optimizer")
parser.add_argument("--batch-size", type=int, default=64, help="number of timesteps to optimize at the same time")
args = parser.parse_args()

if args.alg=="HybridTransformerPPO":
    from HybridTransformerPPO.hybridppo import *
    from HybridTransformerPPO.hybridBuffer import *
    from HybridTransformerPPO.policies import *
else:
    from HybridPPO.hybridBuffer import *
    from HybridPPO.hybridppo import *
    from HybridPPO.policies import *

from bus_bunch.environments import Env

if __name__ == '__main__':

    env = Env()
    # if recording
    # env = gym.wrappers.Monitor(env, "./video", force=True)
    # env.metadata["render.modes"] = ["human", "rgb_array"]
    
    env.reset()
    
    print(":::::::::::Start training!!:::::::::::")

    if args.alg == "HybridTransformerPPO":
        model = HybridTransformerPPO("HybridPolicy", 
                        env, 
                        verbose=1, 
                        batch_size=args.batch_size, 
                        tensorboard_log="./results/"+args.env+"/"+args.alg+"/",
                        learning_rate=args.lr,)
    else:
        model = HybridPPO("HybridPolicy", 
                        env, 
                        verbose=1, 
                        batch_size=args.batch_size, 
                        tensorboard_log="./results/"+args.env+"/"+args.alg+"/",
                        learning_rate=args.lr,)

    model.learn(total_timesteps=args.num_timesteps)
    model.save("./results/"+args.env+"/"+args.alg+"/model")

    del model # remove to demonstrate saving and loading

    # print(":::::::::::Start evaluating!!:::::::::::")
    # if args.alg == "HybridTransformerPPO":
    #     model = HybridTransformerPPO.load("./results/"+args.env+"/"+args.alg+"/model")
    # else:
    #     model = HybridPPO.load("./results/"+args.env+"/"+args.alg+"/model")

    # obs = env.reset()
    # while env.env.peek() < 10700:
    #     action, _ = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    #     # if rendering
    #     # env.render()
    #     time.sleep(0.1)
    
    # pickle.dump(env.data, open("./results/"+args.env+"/"+args.alg+"/data.pkl", 'wb'))
    # print(env.departure_times)
    # print('Total waiting time: ', env.acc_waiting_time)
    # print('Total on bus time: ', env.acc_on_bus_time)
    # print('stops allowde to skip: ', env.num_skipping_stop, ' ', env.num_total_stop)

    # time.sleep(1)
    # env.close()
    
    # import numpy as np
    # import matplotlib.pyplot as plt
    # import seaborn as sns

    # import pickle
    # data = pickle.load(open("./results/"+args.env+"/"+args.alg+"/data.pkl", 'rb'))
    # data_plot = np.array([[k, i] for k, v in data.items() for i in v])
    # sns.scatterplot(x=data_plot[:, 0], y=data_plot[:, 1])
