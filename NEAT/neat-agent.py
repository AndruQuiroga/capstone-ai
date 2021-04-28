import multiprocessing

import numpy as np
import neat
import gym


class Dude(object):
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config

    def do(self):
        env = gym.make("Assault-ram-v0")
        env.reset()

        state, _, _, *_ = env.step(env.action_space.sample())
        net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)

        fitness = 0
        counter = 0
        remaining_lives = 4

        while True:
            # env.render()

            state = state / 255.0
            actions = net.activate(state)

            state, reward, done, info = env.step(np.argmax(actions))

            fitness += reward
            fitness += .25
            counter += 1

            if info['ale.lives'] != remaining_lives:
                remaining_lives -= 1
                fitness -= 50

            if done:
                fitness -= 50
                break

            if counter > 1500:
                break

        return fitness


def eval_genomes(genome, config):
    return Dude(genome, config).do()


if __name__ == '__main__':

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config-feedforward.txt')

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genomes)
    winner = p.run(pe.evaluate, 300)

