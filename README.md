# EV-RL-Honours-Project
Repository for EV RL Honours project work
----------------------------------------------

Configs - Stores hyperparameters and experiment configuration files (.yaml). 
        - Defines model, algorithm, observation, reward, and training settings. Used as input to training and evaluation scripts

Evaluations - Where past evaluations are saved. 

Experiments - Stores results from training runs. 
            - Subfolders follow the format: AgentName_Kvalue_Zvalue_seed_date_misc. 
            - Each experiment typically contains: 
              - models/ (saved checkpoints)
              - logs/ (training metrics)
              - configs/ (used configuration files)

Scenarios - Contains all SUMO scenario files. Each scenario is stored in its own folder
          - Includes: 
            - .net.xml (road network)
            - .rou.xml (traffic demand) - May include multiple demand files for the same network

Scripts - contains Evaluation, Training and Utils sub folders
  - Training - Used to train the agent (e.g. MAPPO or DQN). Produces trained models, logs, and configuration files.
  - Evaluation - Used after the agent is trained to test its performance. Includes scripts for different algorithms (DQN and MAPPO) and reports metrics such as reward, ambulance travel time, and civilian waiting time.
  - Utils - Contains utility scripts for plotting and analysing results (e.g. comparing baseline vs trained agent performance).

src – Reinforcement Learning environment and agents
  - agents – Contains RL agent implementations
      - DQN – Deep Q-Network (value-based, single-agent)
      - MAPPO – Multi-Agent Proximal Policy Optimisation (actor-critic, multi-agent)

  - core – Environment definition and supporting components
      - env.py – Core environment. Defines the World class which maintains the full SUMO simulation state
      - parlenv.py – Parallel multi-agent environment wrapper. Treats each traffic light as an independent agent
      - Observations.py – Defines the observation/state representation provided to agents
      - Rewards.py – Defines reward functions used to train agents
