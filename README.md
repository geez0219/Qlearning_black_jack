# Qlearning_black_jack
## Background
Black jack is the most popular casino game all over the world because of its highest winning expectation (almost 50%) if players play it "correctly". Refer to playing "correctly", it means you follow the black jack strategy table that already built by consicely calculation. But those strategy are made under the assumption that the deck is totally balanced (the decks is composed by mutiple poker deck). If casinos use imballanced deck, the so-called "best strategy" may not be the best strategy anymore.  

## What this package can do
1. This repository allows users to train a black jack strategy (black jack AI) by using Q-learning algorithm. 
2. Because the strategy is learned by playing millions of games from the specific game environment we create, this package can adapt different environment setting and find the best strategy. (e.g: different cards distribution, different rules)


# How to use it?
## train model
```
Qlearning_black_jack#  python train_model.py
```

This command will train the RL_agent to play black jack by playing 100000000 games, output the trained model to "trained_RL.pkl" file, and plot the interactive strategy table.

![ezgif com-crop](https://user-images.githubusercontent.com/27904418/53593159-d2b2d780-3b4c-11e9-8cc0-b8ede73f2542.gif)

## test model
```
Qlearning_black_jack# python test_model.py
```
This command will load the trained model call "trained_RL.pkl" and test it by playing 10000 black jack until it lose all its money (starting money: 100$, basic bet: 1$)

![test_model_demo](https://user-images.githubusercontent.com/27904418/54009499-eb5c5800-411f-11e9-809b-8234ab77eb03.png)

## plot strategy table
```
Qlearning_black_jack# python plot_strategy.py
```
This command will plot the interactive strategy table from model file called "trained_RL.pkl".

## play yourself
```
Qlearning_black_jack# python play_myself.py
```
This command will that you play black jack and test the gaming environment by yourself
![play_myself_demo](https://user-images.githubusercontent.com/27904418/54009916-9588af80-4121-11e9-8766-fc26ce7d0b37.png)

