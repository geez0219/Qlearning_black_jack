# Qlearning_black_jack
This repository allows user to train a black jack strategy table (black jack AI) by using Q-learning algorithm 

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
This command will load the trained model call "trained_RL.pkl" and test it by playing 10000 black jack until it lose all its money (starting money: 100, basic bet: 1)

![test_model_demo](https://user-images.githubusercontent.com/27904418/54009499-eb5c5800-411f-11e9-809b-8234ab77eb03.png)
