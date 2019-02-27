import numpy as np
from blackjack_game import BlackJackGame


def printHand(hand):
    for card in hand:
        print('{},'.format(card), end='')


if __name__ == '__main__':
    env = BlackJackGame()

    while True:
        observation, reward, done = env.newRound()
        print(observation)
        print('dealer\'s hand: ')
        printHand(env.dealerHand)
        print('')
        print('player\'s hand: ')
        printHand(env.playerHand)
        print('')
        while not done:
            action = int(input('enter the action (0: stand, 1: hit, 2: double down, 3: split, 4: check remain deck)'))
            if action == 4:
                env.printTheRemainDeck()
            else:
                observation, reward, done = env.step(action)
                print(observation)
                print('dealer\'s hand: ')
                printHand(env.dealerHand)
                print('')
                print('player\'s hand: ')
                printHand(env.playerHand)
                print('')

        if reward > 0:
            print('player win {}$ !!!'.format(reward))
        elif reward < 0:
            print('player lose {}$'.format(-1*reward))
        else:
            print('push')
        print('----------------------------------------')

























