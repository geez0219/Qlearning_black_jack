'''
blackJack game environment with action: stand, hit, split, double down

* when split action is conducted, the player card will return to single and double the bet, but the game thread will
still remain to one.

* the dealer deck can be arbitrary, the card will not be replaced

'''

import numpy as np


class BlackJackGame(object):
    def __init__(self, startDeck=None):
        self.n_actions = 4  # 0: stand, 1:hit, 2:double down, 3:split
        self.bet = 1
        if startDeck is None:
            self.deckNumber = 8
            self.startDeck = ['A', 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * self.deckNumber
        else:
            self.startDeck = startDeck
        self.shuffleTime = 0.5
        self.cardPool = self.startDeck.copy()
        np.random.shuffle(self.cardPool)
        self.dealerHand = [self.cardPool.pop(0)]
        self.playerHand = [self.cardPool.pop(0), self.cardPool.pop(0)]
        self.playerPoints, self.playerIsSoft = self.getPoint(0, 0, self.playerHand[0])
        self.playerPoints, self.playerIsSoft = self.getPoint(self.playerPoints, self.playerIsSoft, self.playerHand[1])
        self.dealerPoints, self.dealerIsSoft = self. getPoint(0, 0, self.dealerHand[0])
        self.canSplit = 1 if self.playerHand[0] == self.playerHand[1] else 0
        self.canDouble = 1

    def getPoint(self, points, isSoft, newCard):
        if newCard == 'A':
            if points <= 10:
                newPoints = points + 11
                isSoft = 1
            else:
                newPoints = points + 1
        else:
            newCard = int(newCard)
            newPoints = points + newCard
        if newPoints > 21 and isSoft:
            newPoints += -10
            newIsSoft = 0
        else:
            newIsSoft = isSoft

        return newPoints, newIsSoft

    def step(self, action):
        if action == 0:  # stand
            done = 1
            result = self.dealerPlayAndJudge()

        elif action == 1:  # hit
            self.canDouble = 0
            self.canSplit = 0
            done = self.playerHit()
            result = -1 if done == 1 else 0

        elif action == 2:  # double down
            if self.canDouble == 0:
                raise ValueError('cannot double down !!')
            done = 1
            self.bet *= 2
            playerLose = self.playerHit()
            if playerLose:
                result = -1
            else:
                result = self.dealerPlayAndJudge()

        elif action == 3:  # split
            if self.canSplit == 0:
                raise ValueError('cannot split !!')
            done = 0
            result = 0
            self.bet *= 2
            self.playerHand[1] = self.cardPool.pop(0)
            self.playerPoints, self.playerIsSoft = self.getPoint(0, 0, self.playerHand[0])
            self.playerPoints, self.playerIsSoft = self.getPoint(self.playerPoints, self.playerIsSoft, self.playerHand[1])
            self.canSplit = 1 if self.playerHand[0] == self.playerHand[1] else 0

        else:
            ValueError('there is no action {}'.format(action))
        observation = [self.dealerPoints, self.playerPoints, [self.playerIsSoft, self.canDouble, self.canSplit]]
        return observation, result * self.bet, done

    def playerHit(self):
        newCard = self.cardPool.pop(0)
        self.playerPoints, self.playerIsSoft = self.getPoint(self.playerPoints, self.playerIsSoft, newCard)
        self.playerHand = np.concatenate((self.playerHand, [newCard]))

        if self.playerPoints > 21 and not self.playerIsSoft:
            playerLose = 1
        else:
            playerLose = 0

        return playerLose

    def dealerPlayAndJudge(self):
        while self.dealerPoints < 17 or (self.dealerPoints > 21 and self.dealerIsSoft):
            newCard = self.cardPool.pop(0)
            self.dealerPoints, self.dealerIsSoft = self.getPoint(self.dealerPoints, self.dealerIsSoft, newCard)
            self.dealerHand = np.concatenate((self.dealerHand, [newCard]))

        if self.dealerPoints > 21:
            result = 1
        else:
            if self.dealerPoints > self.playerPoints:
                result = -1
            elif self.dealerPoints == self.playerPoints:
                result = 0
            else:
                result = 1

        return result

    def reset(self):
        self.cardPool = self.startDeck.copy()
        np.random.shuffle(self.cardPool)
        self.bet = 1
        self.dealerHand = [self.cardPool.pop(0)]
        self.playerHand = [self.cardPool.pop(0), self.cardPool.pop(0)]
        self.playerPoints, self.playerIsSoft = self.getPoint(0, 0, self.playerHand[0])
        self.playerPoints, self.playerIsSoft = self.getPoint(self.playerPoints, self.playerIsSoft, self.playerHand[1])
        self.dealerPoints, self.dealerIsSoft = self. getPoint(0, 0, self.dealerHand[0])
        self.canSplit = 1 if self.playerHand[0] == self.playerHand[1] else 0
        self.canDouble = 1

        observation = [self.dealerPoints, self.playerPoints, [self.playerIsSoft, self.canDouble, self.canSplit]]
        return observation, 0, 0

    def newRound(self):
        if len(self.cardPool) < len(self.startDeck) * self.shuffleTime:
            self.cardPool = self.startDeck.copy()
            np.random.shuffle(self.cardPool)
        self.bet = 1
        self.dealerHand = [self.cardPool.pop(0)]
        self.playerHand = [self.cardPool.pop(0), self.cardPool.pop(0)]
        self.playerPoints, self.playerIsSoft = self.getPoint(0, 0, self.playerHand[0])
        self.playerPoints, self.playerIsSoft = self.getPoint(self.playerPoints, self.playerIsSoft, self.playerHand[1])
        self.dealerPoints, self.dealerIsSoft = self. getPoint(0, 0, self.dealerHand[0])
        self.canSplit = 1 if self.playerHand[0] == self.playerHand[1] else 0
        self.canDouble = 1

        observation = [self.dealerPoints, self.playerPoints, [self.playerIsSoft, self.canDouble, self.canSplit]]
        return observation, 0, 0

    def printTheRemainDeck(self):
        cardStat = {}
        print('the remain deck:')
        for card in self.cardPool:
            if card in cardStat:
                cardStat[card] += 1
            else:
                cardStat[card] = 1

        for card in ['A', 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            if card in cardStat:
                print('card {}: {} '.format(card, cardStat[card]))
            else:
                print('card {}: 0 '.format(card))

