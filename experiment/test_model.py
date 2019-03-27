"""
test the trained RL_brain by automatically playing blackJack until either lose all money or hit the rounds limit
"""

from blackjack_game import BlackJackGame
import numpy as np
import matplotlib.pyplot as plt
import pickle


def autoPlayWithMoney(env, strategyTable, startMoney, rounds, detail):
    money = startMoney
    env.reset()
    if detail:
        for round in range(rounds):
            print('================ {}th round ================='.format(round))
            print('your recent money: {}$'.format(money))
            print('--------------------------------------------')
            if money <= 0:
                print('you lose all money QQ')
                break

            obs, _, _ = env.newRound()
            print(obs)
            while True:
                action = strategyTable.choose_action(obs)
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
                obs, reward, done = env.step(action)
                print(obs)
                if done:
                    if reward > 0:
                        print('player won {}$'.format(reward))
                    elif reward == 0:
                        print('push')
                    else:
                        print('player lose {}$'.format(-reward))
                    money += reward
                    break

        print('================ game over ==================')
        print('your start money: {}$'.format(startMoney))
        print('your final money: {}$'.format(money))

    else:
        for round in range(rounds):
            if money <= 0:
                break

            obs, _, _ = env.newRound()
            while True:
                action = strategyTable.choose_action(obs)
                obs, reward, done = env.step(action)
                if done:
                    money += reward
                    break

    return money, round

if __name__ == "__main__":
    env = BlackJackGame()
    file = open('trained_RL2.pkl', 'rb')
    RL = pickle.load(file)
    exp_times = 100
    start_money = 100
    max_round = 10000

    turn_hist = np.zeros(100)
    for i in range(exp_times):
        money, turn_hist[i] = autoPlayWithMoney(env, RL, start_money, max_round, detail=0)

    plt.hist(turn_hist)
    plt.show()
