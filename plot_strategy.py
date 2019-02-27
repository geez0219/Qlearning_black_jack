"""
plot the interactive strategy table
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox, BboxTransform
import pickle


def plotStrategy(Qdict):
    def get_courser_index(img, event):
        xmin, xmax, ymin, ymax = img.get_extent()
        if img.origin == 'upper':
            ymin, ymax = ymax, ymin
        arr = img.get_array()
        data_extent = Bbox([[ymin, xmin], [ymax, xmax]])
        array_extent = Bbox([[0, 0], arr.shape[:2]])
        trans = BboxTransform(boxin=data_extent, boxout=array_extent)
        y, x = event.ydata, event.xdata
        point = trans.transform_point([y, x])
        if any(np.isnan(point)):
            return None
        row, col = point.astype(int)
        # Clip the coordinates at array bounds
        return row, col

    def hover(event):
        # if the mouse is over the scatter points
        if im.contains(event)[0]:
            if last_text_object:
                last_text_object[0].set_color('w')
                last_text_object.pop()

            row, col = get_courser_index(im, event)
            text_object[row, col].set_color('r')
            last_text_object.append(text_object[row, col])

            #
            playerPoints, isSoft, canSplit = rowData[row]
            dealerPoints = col + 2
            state = str([dealerPoints, playerPoints, [isSoft, 1, canSplit]])

            ax2.clear()
            if state in Qdict:
                value_list = Qdict[state]
                ax2.bar(np.arange(len(value_list)), value_list, width=0.5)
                ax2.set_xticks(np.arange(len(value_list)))
                ax2.set_xticklabels(actions_long[:len(value_list)])
                ax2.axhline(linestyle='--', color='k')
                for i in range(len(value_list)):
                    ax2.text(i, value_list[i], '{:.3f}'.format(value_list[i]), ha="center",
                             va="bottom" if value_list[i]>0 else "top", color="k")
            else:
                ax2.text(0.5, 0.5, 'Not Exist', ha="center", va="center", color="k", fontsize=10, fontstretch=100)
            state = str([dealerPoints, playerPoints, [isSoft, 0, canSplit]])
            ax2.set_title("If can double")
            ax2.set_ylabel('expected wining money (assume play 1$)')

            ax3.clear()
            if state in Qdict:
                value_list = Qdict[state]
                ax3.bar(np.arange(len(value_list)), value_list, width=0.5)
                ax3.set_xticks(np.arange(len(value_list)))
                ax3.set_xticklabels(actions_long[:len(value_list)])
                ax3.axhline(linestyle='--', color='k')
                for i in range(len(value_list)):
                    ax3.text(i, value_list[i], '{:.3f}'.format(value_list[i]), ha="center",
                             va="bottom" if value_list[i] > 0 else "top", color="k")
            else:
                ax3.text(0.5, 0.5, 'Not Exist', ha="center", va="center", color="k", fontsize=10, fontstretch=100)
            ax3.set_title("If cannot double")
            ax3.set_ylabel('expected wining money (assume play 1$)')
        fig.canvas.draw_idle()

    # change the table format
    action_img = np.zeros((36, 10), dtype=np.int32)  # rows, cols, canDouble
    actions_short = ['S', 'H', 'D', 'Sp']
    actions_long = ['Stand', 'Hit', 'Double', 'Split']

    # rowData[i] = [playerPoint, isSoft, canSplit]
    rowData = [[x, 0, 0] for x in range(5, 22)]  # for playerPoint 5~21 (hard)
    rowData += [[x, 1, 0] for x in range(13, 22)]  # for playerPoint 13~21 (soft)
    rowData += [[12, 1, 1]]  # for player got A,A
    rowData += [[x*2, 0, 1] for x in range(2, 11)]  # for player got 2,2 ~ 10,10

    for row in range(36):
        playerPoints, isSoft, canSplit = rowData[row]
        for col in range(10):
            dealerPoints = col + 2
            state = str([dealerPoints, playerPoints, [isSoft, 1, canSplit]])
            if state not in Qdict:
                state = str([dealerPoints, playerPoints, [isSoft, 0, canSplit]])

            action_img[row, col] = np.argmax(Qdict[state])

    dealerPointTick = [str(x) for x in range(2,11)] + ['A']
    playerPointTick = ['{}(H)'.format(x) for x in range(5,22)] + \
                      ['{}(S)'.format(x) for x in range(13,22)] + \
                      ['A,A'] + \
                      ['{},{}'.format(x, x) for x in range(2,11)]

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(224)

    ax1.set_title("Clickable Strategy Table")
    ax2.set_title("If can double")
    ax2.text(0.5, 0.5, 'Shown by clicking the Strategy Table', ha="center", va="center", color="k")

    ax3.set_title("If cannot double")
    ax3.text(0.5, 0.5, 'Shown by clicking the Strategy Table', ha="center", va="center", color="k")

    im = ax1.imshow(action_img, aspect='auto')

    # fig.colorbar(im, ax=ax1, shrink=0.6)
    # We want to show all ticks...
    ax1.set_xlabel('dealer hands')
    ax1.set_ylabel('player hands')
    ax1.set_xticks(np.arange(len(dealerPointTick)))
    ax1.set_yticks(np.arange(len(playerPointTick)))
    ax1.set_xticks(np.arange(-0.5, len(dealerPointTick)-1, 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, len(playerPointTick)-1, 1), minor=True)

    ax1.grid(which='minor', linestyle='--')

    # ... and label them with the respective list entries
    ax1.set_xticklabels(dealerPointTick)
    ax1.set_yticklabels(playerPointTick)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax1.get_xticklabels(), ha="right")

    # Loop over data dimensions and create text annotations.
    text_object = np.empty((36,10), dtype=np.object)
    for i in range(len(playerPointTick)):
        for j in range(len(dealerPointTick)):
            text_object[i, j] = ax1.text(j, i, actions_short[action_img[i, j]], ha="center", va="center", color="w")

    last_text_object = []
    fig.canvas.mpl_connect('button_press_event', hover)
    plt.show()


if __name__ == '__main__':
    file = open('trained_RL.pkl', 'rb')
    RL = pickle.load(file)
    plotStrategy(RL.q_table)


