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
- Pull project
- run `pip install -r requirements.txt`