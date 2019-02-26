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
from blackJackGame import blackJackGame
from RL_brain import QLearningTable
import matplotlib.pyplot as plt


def drawBlackJackStrategy(pandasDataFrame):
    hitValue = np.zeros((18,10,2))
    standValue = np.zeros((18,10,2))
    hitStandValueDiff = np.zeros((18,10,2))


    for playerPoints in range(4, 22):
        for dealerPoints in range(2, 12):
            for playerIsSoft in range(0,2):
                state = str([dealerPoints, playerPoints, playerIsSoft])
                if state not in pandasDataFrame.index:
                    hitValue[playerPoints-4, dealerPoints-2, playerIsSoft] = 0
                    standValue[playerPoints-4, dealerPoints-2, playerIsSoft] = 0
                    hitStandValueDiff[playerPoints-4, dealerPoints-2, playerIsSoft] = 0
                else:
                    hitValue[playerPoints-4, dealerPoints-2, playerIsSoft] = pandasDataFrame.loc[state,1]
                    standValue[playerPoints-4, dealerPoints-2, playerIsSoft] = pandasDataFrame.loc[state,0]
                    hitStandValueDiff[playerPoints-4, dealerPoints-2, playerIsSoft] = pandasDataFrame.loc[state,1] - pandasDataFrame.loc[state,0]

    dealerPointTick = [str(x) for x in range(2,12)]
    playerPointTick = [str(x) for x in range(4,22)]

    for table, tableName in zip([hitValue, standValue, hitStandValueDiff], ['hit value', 'stand value', 'hit - stand']):
        fig, ax = plt.subplots(1, 2)
        for isSoft in [0,1]:
            im = ax[isSoft].imshow(table[:,:,isSoft])
            im.set_clim(-1,1)
            ax[isSoft].grid(which='major', linestyle='--')
            fig.colorbar(im, ax=ax[isSoft], shrink=0.6)
            # We want to show all ticks...
            ax[isSoft].set_xticks(np.arange(len(dealerPointTick)))
            ax[isSoft].set_yticks(np.arange(len(playerPointTick)))
            # ... and label them with the respective list entries
            ax[isSoft].set_xticklabels(dealerPointTick)
            ax[isSoft].set_yticklabels(playerPointTick)

            # Rotate the tick labels and set their alignment.
            plt.setp(ax[isSoft].get_xticklabels(), rotation=45, ha="right",
                    rotation_mode="anchor")

            # Loop over data dimensions and create text annotations.
            for i in range(len(playerPointTick)):
                for j in range(len(dealerPointTick)):
                    text = ax[isSoft].text(j, i, '{:.2f}'.format(table[i, j, isSoft]),
                                    ha="center", va="center", color="w" if table[i, j, isSoft] > 0 else "r")
            ax[isSoft].set_title("{} (when player is {})".format(tableName, 'soft' if isSoft == 1 else 'hard'))
            fig.tight_layout()
    plt.show()


def update(env, RL):
    for episode in range(10000):
        if episode % 10000 == 0:
            print(episode/10000)
        # initial observation
        observation, _, _ = env.reset()
        # print(observation)
        while True:
            # RL choose action based on observation
            action = RL.choose_action(str(observation))
            # print('player {}'.format('hit' if action else 'stand'))
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            # print(observation_)

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_), done)

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                # if reward > 0:
                #     print('player won !!')
                # elif reward == 0:
                #     print('push')
                # else:
                #     print('dealer won !!')
                # print('-----------------------------')
                break

    # end of game
    drawBlackJackStrategy(RL.q_table)
    print('game over')

if __name__ == "__main__":
    env = blackJackGame()
    RL = QLearningTable(actions=list(range(env.n_actions)))
    update(env, RL)
