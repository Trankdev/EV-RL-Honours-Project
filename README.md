# EV-RL-Honours-Project
Repository for EV RL Honours project work

Configs - Saves/stores hyperparameters

Evaluations - Where past evaluations are saved. 

Experiments - Where past trainings are saved. Subfolders should be of the form Agentname_Kparameterval_Zparameterval_seedval_date_miscidentifier and should contain used configs, logs, models folders and files

Scenarios - All relevant SUMO files for a scenario. Each Scenario should be it's own folder. (Could have subfolder(s) for differing demands/rou.xml files for a given network/net.xml file??)

Scripts - contains Evaluation, Training and Utils sub folders
  - Training - used to train the agent
  - Evaluation - used after Agent is trained and is to test the agent
  - Utils - misc. file for plotting results and other utilities

src - RL environment, contains core and agents sub folders
  - agents - contains RL agent algorithm code
  - core - contains environment/state file (env.py), environment assisting file (parlenv.py), environment observation file (Observations.py) and rewards file (Rewards.py)
