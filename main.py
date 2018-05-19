import gym
from src.rl import RL
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i',
                        default='200',
                        dest='ITERATION',
                        help='input the iteration of training')

    parser.add_argument('-m',
                        default='2000',
                        dest='MEMORYSIZE',
                        help='input the size of memory')

    args = parser.parse_args()
    
    try:
        iteration = int(args.ITERATION)
        mem_size  = int(args.MEMORYSIZE)
    except ValueError:
        print('error: iteration or memory size must be an integer')
        sys.exit()

    game = gym.make('CartPole-v0')
    game = game.unwrapped
    rl   = RL(game.observation_space.shape[0] , range(game.action_space.n), Memory_size=mem_size)

    step = 0
    episode = 0
    while episode < iteration:
        s = game.reset()
        done = False
        while not done:
            game.render()
            a = rl.actor(s)
            ns, r, done, _ = game.step(a)
            x, x_dot, theta, theta_dot = ns
            r1 = (game.x_threshold - abs(x))/game.x_threshold - 0.8
            r2 = (game.theta_threshold_radians - abs(theta))/game.theta_threshold_radians - 0.5
            r = r1 + r2
            rl.store_observation(s, a, r, ns)
            if step > mem_size and step % 5 == 0:
                rl.learn()
            s = ns
            step += 1
        episode += 1

    plt.plot(np.arange(len(rl.history)), rl.history)
    plt.show()
