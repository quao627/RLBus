import warnings
warnings.filterwarnings("ignore")

from hybridppo import *

if __name__ == '__main__':

    env = gym.make('Moving-v0')
    # if recording
    # env = gym.wrappers.Monitor(env, "./video", force=True)
    # env.metadata["render.modes"] = ["human", "rgb_array"]
    
    env.reset()

    ACTION_SPACE = env.action_space[0].n
    PARAMETERS_SPACE = env.action_space[1].shape[0]
    OBSERVATION_SPACE = env.observation_space.shape[0]

    model = HybridPPO("HybridPolicy", 
                    env, 
                    verbose=1, 
                    batch_size=64, 
                    tensorboard_log="./moving_env_tensorboard/",
                    learning_rate=0.00025,)

    model.learn(total_timesteps=1000000)
    model.save("moving_env")

    del model # remove to demonstrate saving and loading

    model = HybridPPO.load("moving_env")

    obs = env.reset()
    while True:
        action, _ = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        # if rendering
        env.render()
        time.sleep(0.1)

    time.sleep(1)
    env.close()