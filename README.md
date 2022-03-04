# RL-project-2022
## What is this project about?
In this project we realized an RL environment that simulates
distance and traffic flow in a city in order to find a path
through the city streets that causes the least amount of 
CO2 emissions. For this we created a grid-like graph
environment fitted with several constraints like one-way
streets, construction sites or traffic lights to simulate
a more realistic environment.\
\
In order to solve this environment we tested several RL 
agents including SARSA and A2C that attempt to navigate
the city environment to deliver packages to several nodes
in the environment while emitting the least amount of
emissions. We compared the performance of these RL agents
against several baselines that move either randomly or 
follow simple rules like always choosing the edge with the
lowest weight.
## Installation
- pull project
- run `pip install -r requirements.txt`
- if you wish to run the a2c or ppo agents on the GPU
  the GPU version of torch and the corresponding CUDA
  version need to be installed manually
## Reproducibility
To reproduce the results from the experiments several steps
are necessary. First, to train  with SARSA on a specific 
seed and at one of the preset sizes (small: 3x3, 
medium: 5x5, large: 10x10) you need to make sure there 
are no saved Q tables. Saved Q tables can be found in 
the project folder where the main.py is also stored. 
They are recognizable by always  being saved as 
q_sarsa_{small/medium/large}_{seed}.csv. Once you made
sure there are no Q tables for the chosen seed and size
training and evaluation can be started by running the main
using `python main.py` in the command line of a terminal
that is opened in the project folder.

To modify the training several parameters can be used. 
For the sake of reproducibility the only parameters 
explained here are the ones necessary to reproduce the 
results. The other parameters can be ignored and are 
explained in the code comments of main.py. While the 
results for the SARSA agent can be reproduced simply by 
running main.py the other agents and baselines require 
additional parameters.

To run the A2C agent `python main.py -m a2c` needs to be
called and to run the PPO agent `python main.py -m ppo`
needs to be called. All internal parameters like grid-size,
number of time steps, starting position, constraints and
position of packages are preset and do not need to be set 
manually. Also of note here is that A2C and PPO only 
produce sensible results for the small 3x3 grid, therefore
nonsensical results for 5x5 and 10x10 do not mean that
you have followed the instructions wrongly.

To reproduce the baseline results additional parameters 
have to be added similarly to the other agents. To recreate
the random baseline `python main.py -m random` has to be 
run, to recreate the minimum weight baseline `python main.py
-m min_weight` has to be run and to recreate the maximum
weight baseline `python main.py -m max_weight` has to be
run.

To show a graphic representation of the created 
environment `-g` can be called as an additional parameter
that shows a color-coded version of the environment as a
grid. An example graph can be seen below. The nodes are
color-coded as follows: green (starting point), 
red (package), yellow (traffic light), blue (normal node).
Additionally missing edges represent construction sites
and directed edges represent one-way streets. This 
graph representation is not perfect and can not display
several color-codes on the same node. For example a node
that has a package and a traffic light will only be shown
in red. 

Furthermore, the files "training.py" and "test.py" can be
called individually to train the (SARSA) agents and
evaluate the agents respectively. But be aware that you have
to run the "training.py" or "main.py" once to save Q tables which are
necessary for evaluating the SARSA agents. But this does not
affect the reproducibility and using "main.py" instead is highly
recommended.