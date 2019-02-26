"""
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].
This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""
import numpy as np
from blackJackGame2 import blackJackGame
from RL_brain import QLearningTable
import matplotlib.pyplot as plt

def drawBlackJackStrategy(Qdict):

    # change the table format
    Qtable = [[[1,2],[3,4]],[[5,6],[7,8]]]
    for playerIsSoft in range(2):
        for canDouble in range(2):
            for canSplit in range(2):
                if (canDouble, canSplit) == (1, 1):
                    actionNum = 4
                elif (canDouble, canSplit) == (1, 0):
                    actionNum = 3
                else:
                    actionNum = 2
                Qtable[playerIsSoft][canDouble][canSplit] = np.zeros((22, 22, actionNum))
                for dealerPoints in range(22):
                    for playerPoints in range(22):
                        state = str([dealerPoints, playerPoints, [playerIsSoft, canDouble, canSplit]])
                        if state in Qdict:
                            Qtable[playerIsSoft][canDouble][canSplit][dealerPoints, playerPoints, :] = Qdict[state]

                        else:
                            Qtable[playerIsSoft][canDouble][canSplit][dealerPoints, playerPoints, :] = np.zeros(actionNum)
                print('{},{},{}'.format(playerIsSoft, canDouble, canSplit))
                print(Qtable[0][0][0].shape)
    stratMapAction = np.zeros((36, 10, 2))  # rows, cols, canDouble
    stratMapValue = np.zeros((36, 10, 2))
    actions = ['S', 'H', 'D', 'P']
    # rowData[i] = [playerPoint, isSoft, canSplit]
    rowData = [[x, 0, 0] for x in range(5, 22)]  # for playerPoint 5~21 (hard)
    rowData += [[x, 1, 0] for x in range(13, 22)]  # for playerPoint 13~21 (soft)
    rowData += [[12, 1, 1]]  # for player got A,A
    rowData += [[x*2, 0, 1] for x in range(2, 11)] #for player got 2,2 ~ 10,10

    for row in range(36):
        playerPoints, isSoft, canSplit = rowData[row]
        for col in range(10):
            dealerPoints = col + 2
            for canDouble in range(2):
                state = Qtable[isSoft][canDouble][canSplit][dealerPoints, playerPoints]
                stratMapValue[row, col, canDouble] = state.max()
                stratMapAction[row, col, canDouble] = np.random.choice(np.where(state == state.max())[0])

    dealerPointTick = [str(x) for x in range(2,11)] + ['A']
    playerPointTick = ['{}(H)'.format(x) for x in range(5,22)] + \
                      ['{}(S)'.format(x) for x in range(13,22)] + \
                      ['A,A'] + \
                      ['{},{}'.format(x, x) for x in range(2,11)]
    #

    fig, ax = plt.subplots(1,2)
    for canDouble in range(2):
        im = ax[canDouble].imshow(stratMapAction[:,:, canDouble])
        ax[canDouble].grid(which='major', linestyle='--')
        fig.colorbar(im, ax=ax[canDouble], shrink=0.6)
        # We want to show all ticks...
        ax[canDouble].set_xticks(np.arange(len(dealerPointTick)))
        ax[canDouble].set_yticks(np.arange(len(playerPointTick)))
        # ... and label them with the respective list entries
        ax[canDouble].set_xticklabels(dealerPointTick)
        ax[canDouble].set_yticklabels(playerPointTick)
        # Rotate the tick labels and set their alignment.
        plt.setp(ax[canDouble].get_xticklabels(), rotation=45, ha="right",
                        rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(playerPointTick)):
            for j in range(len(dealerPointTick)):
                text = ax[canDouble].text(j, i, '{:.2f}'.format(stratMapValue[i, j, canDouble]),
                                        ha="center", va="center", color="w" if stratMapValue[i, j, canDouble] > 0 else "r")
        ax[canDouble].set_title("when player {} double".format('can' if canDouble else 'cannot'))
        fig.tight_layout()
    plt.show()

def update(env, RL, detail):

    for episode in range(100000000):
        if episode % 10000 == 0:
            print(episode/10000)

        # initial observation
        observation, _, _ = env.reset()
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
    drawBlackJackStrategy(RL.q_table)
    print('game over')

if __name__ == "__main__":
    env = blackJackGame()
    RL = QLearningTable(actions=list(range(env.n_actions)))
    update(env, RL, 0)


