"""
test the trained RL_brain by automatically playing blackJack until either lose all money or hit the rounds limit
"""

from blackjack_game import BlackJackGame
import pickle


def autoPlayWithMoney(env, strategyTable, startMoney, rounds):
    money = startMoney
    env.reset()
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

if __name__ == "__main__":
    env = BlackJackGame()
    file = open('trained_RL.pkl', 'rb')
    RL = pickle.load(file)
    autoPlayWithMoney(env, RL, 100, 10000)
