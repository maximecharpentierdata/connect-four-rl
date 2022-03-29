# connect-four-rl
Reinforcement Learning the for Connect Four game.

The code related to training the agents and generating episodes is situated in `src/game`. The environment is coded in `src/connect_four_env` and the agents are in `src/agents`.

## Environment setup

To install the necessary packages you can simply run the following command. 

```
pip install -r requirements.txt
```

## Training agents

To train an agent you can use the `Train an agent.ipynb` notebook we provided, or simply use it as a usage example.

## Playing against an agent

If you feel like playing an against an agent you trained, feel free to do so using the `Play with an agent.ipynb` notebook. Please note that it does work in the VS Code notebook renderer.

We provided an agent already trained in `models/best_agent` if you want to face this one.