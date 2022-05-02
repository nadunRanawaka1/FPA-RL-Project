# setup
The provided FPA-RL.yml file should help with installing the required 
dependencies to run the code. Please contact if there are issues. 

# Which files to run

learner.py runs (double) deep Q-learning. In the main function, set the number of
episodes you want to run the DQL for and which map to run it for (specified by
num_obs). A stopping threshold can also be set and the algorithm will halt 
execution if a path that is shorter than the threshold is found.


fpa.py runs the flower pollination algorithm followed by DQL . In the main 
function, you can set the hyperparameters for FPA and parameters for DQL as
mentioned above. 