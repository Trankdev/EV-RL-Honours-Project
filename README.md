# EV-RL-Honours-Project
Repository for EV RL Honours project work
----------------------------------------------

Configs - Saves/stores hyperparameters

Evaluations - Where past evaluations are saved. 

Experiments - Where past trainings are saved. Subfolders should be of the form Agentname_Kparameterval_Zparameterval_seedval_date_miscidentifier and should contain used configs, logs, models folders and files

Scenarios - All relevant SUMO files for a scenario. Each Scenario should be it's own folder. (Could have subfolder(s) for differing demands/rou.xml files for a given network/net.xml file??)

Scripts - contains Evaluation, Training and Utils sub folders
  - Training - used to train the agent
  - Evaluation - used after Agent is trained and is to test the agent
  - Utils - misc. file for plotting results and other utilities

src – Reinforcement Learning environment and agents
  - agents – Contains RL agent implementations
      - DQN – Deep Q-Network (value-based, single-agent)
      - MAPPO – Multi-Agent Proximal Policy Optimisation (actor-critic, multi-agent)

  - core – Environment definition and supporting components
      - env.py – Core environment. Defines the World class which maintains the full SUMO simulation state
      - parlenv.py – Parallel multi-agent environment wrapper. Treats each traffic light as an independent agent
      - Observations.py – Defines the observation/state representation provided to agents
      - Rewards.py – Defines reward functions used to train agents
