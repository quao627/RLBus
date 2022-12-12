import argparse
import time
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument("--env", default="Moving-v0", help="environment ID : Moving-v0, Sliding-v0, Busbunch-v0")
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
import gym_hybrid
import bus_bunch
from bus_bunch.environments import Env

if __name__ == '__main__':

    env = Env()
    print(env.action_space)
    # if recording
    # env = gym.wrappers.Monitor(env, "./video", force=True)
    # env.metadata["render.modes"] = ["human", "rgb_array"]
    
    env.reset()

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

    if args.alg == "HybridTransformerPPO":
        model = HybridTransformerPPO.load("./results/"+args.env+"/"+args.alg+"/model")
    else:
        model = HybridPPO.load("./results/"+args.env+"/"+args.alg+"/model")

    obs = env.reset()
    while True:
        action, _ = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        # if rendering
        env.render()
        time.sleep(0.1)

    time.sleep(1)
    env.close()