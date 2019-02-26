import numpy as np
from blackJackGame import blackJackGame
from RL_brain import QLearningTable
import matplotlib.pyplot as plt
import pickle

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

    # fig, ax = plt.subplots(1,2)
    # for canDouble in range(2):
    #     im = ax[canDouble].imshow(stratMapAction[:,:, canDouble])
    #     ax[canDouble].grid(which='major', linestyle='--')
    #     fig.colorbar(im, ax=ax[canDouble], shrink=0.6)
    #     # We want to show all ticks...
    #     ax[canDouble].set_xticks(np.arange(len(dealerPointTick)))
    #     ax[canDouble].set_yticks(np.arange(len(playerPointTick)))
    #     # ... and label them with the respective list entries
    #     ax[canDouble].set_xticklabels(dealerPointTick)
    #     ax[canDouble].set_yticklabels(playerPointTick)
    #     # Rotate the tick labels and set their alignment.
    #     plt.setp(ax[canDouble].get_xticklabels(), rotation=45, ha="right",
    #                     rotation_mode="anchor")
    #
    #     # Loop over data dimensions and create text annotations.
    #     for i in range(len(playerPointTick)):
    #         for j in range(len(dealerPointTick)):
    #             text = ax[canDouble].text(j, i, '{:.2f}'.format(stratMapValue[i, j, canDouble]),
    #                                     ha="center", va="center", color="w" if stratMapValue[i, j, canDouble] > 0 else "r")
    #     ax[canDouble].set_title("when player {} double".format('can' if canDouble else 'cannot'))
    #     fig.tight_layout()

    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(stratMapAction[:,:, 0])
    ax.grid(which='major', linestyle='--')
    fig.colorbar(im, ax=ax, shrink=0.6)
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(dealerPointTick)))
    ax.set_yticks(np.arange(len(playerPointTick)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(dealerPointTick)
    ax.set_yticklabels(playerPointTick)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                    rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(playerPointTick)):
        for j in range(len(dealerPointTick)):
            text = ax.text(j, i, '{:.2f}'.format(stratMapValue[i, j, 0]),
                                    ha="center", va="center", color="w" if stratMapValue[i, j, 0] > 0 else "r")
    ax.set_title("when player {} double".format('can' if 0 else 'cannot'))
    # fig.tight_layout()

    plt.draw()
    plt.show()
    # plt.pause(0.01)
    # input('press enter to continue')
    # plt.savefig('test.png')

if __name__ == '__main__':
    file = open('trained_RL.pkl', 'rb')
    RL = pickle.load(file)
    drawBlackJackStrategy(RL.q_table)


