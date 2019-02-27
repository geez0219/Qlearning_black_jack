"""
train the Qlearning model for playing blackJack.
At the ena, it will save the learning agent, RL_brain instance, as 'trained_RL.pkl'

"""
from blackjack_game import BlackJackGame
from RL_brain import QLearningTable
from plot_strategy import plotStrategy
import pickle


def autoPlayTrain(env, RL, rounds, detail):
    for episode in range(rounds):
        if episode % int(rounds/100) == 0:
            print('autoPlayTrain process: {:.0f}%'.format(episode/rounds*100))

        # initial observation
        observation, _, _ = env.newRound()
        if detail == 1:
            print(observation)
            while True:
                # RL choose action based on observation
                action = RL.choose_action(observation)
                if action == 0:
                    print('player stand')
                elif action == 1:
                    print('player hit')
                elif action == 2:
                    print('player double down')
                elif action == 3:
                    print('player split')
                else:
                    raise ValueError('action should be in [0,1,2,3]')
                # RL take action and get next observation and reward
                observation_, reward, done = env.step(action)
                print(observation_)

                # RL learn from this transition
                RL.learn(observation, action, reward, observation_, done)

                # swap observation
                observation = observation_

                # break while loop when end of this episode
                if done:
                    if reward > 0:
                        print('player won {}!!'.format(reward))
                    elif reward == 0:
                        print('push')
                    else:
                        print('dealer won {}!!'.format(-1 * reward))
                    print('-----------------------------')
                    break
        else:
            while True:
                # RL choose action based on observation
                action = RL.choose_action(observation)

                # RL take action and get next observation and reward
                observation_, reward, done = env.step(action)

                # RL learn from this transition
                RL.learn(observation, action, reward, observation_, done)

                # swap observation
                observation = observation_

                # break while loop when end of this episode
                if done:
                    break
    # end of game

if __name__ == "__main__":
    env = BlackJackGame()
    RL = QLearningTable(actions=list(range(env.n_actions)))
    autoPlayTrain(env, RL, 100000000, 0)
    file = open('trained_RL.pkl', 'wb')
    pickle.dump(RL, file)
    plotStrategy(RL.q_table)
