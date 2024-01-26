# Communicationa-aware formation control algorithm
# Author: Sang Xing
# Date: 01/26/2024

import numpy as np
import matplotlib.pyplot as plt
import time
from utils import *

#---------------------------#
# Initialize all parameters #
#---------------------------#

# Initialize the maximum iteration
max_iter = 300

# Initialize agents' positions
swarm_position = np.array([
    [-5, 14],
    [-5, -19],
    [0, 0],
    [35, -4],
    [68, 0],
    [72, 13],
    [72, -18]
])

# Initialize the swarm size
swarm_size = swarm_position.shape[0]

# Initialize the swarm velocity
swarm_speed = np.zeros((swarm_size, 2))

# Initialize the list for swarm trajectory plot
swarm_paths = []

# Define agent color
node_colors = np.array([
    [108, 155, 207],  # Light Blue
    [247, 147, 39],   # Orange
    [242, 102, 171],  # Light Pink
    [255, 217, 90],   # Light Gold
    [122, 168, 116],  # Green
    [147, 132, 209],  # Purple
    [245, 80, 80]     # Red
]) / 255  # Divide by 255 to scale the RGB values to the [0, 1] range

# Define the color of the agents' communication links aka edges
line_colors = np.random.rand(swarm_position.shape[0], swarm_position.shape[0], 3)

# System parameter about antenna characteristics
alpha = 10**(-5)  

# Required application data rate
delta = 2  
beta = alpha*(2**delta-1)

# Path loss exponent
v = 3  

# Reference distance 
r0 = 5  

# Reception probability threshold
PT = 0.94  

# Performance indicators
Jn = 0
rn = 0

# Initialize the communication qualities matrix to record the communication qualities between agents
communication_qualities = np.zeros((swarm_size, swarm_size))

# Initialize the distances matrix to record the distances between agents
distances = np.zeros((swarm_size, swarm_size))

# Initialize the rho matrix to record the rho_ij value (combined near and far-field quality) to indicate neighboring agents
neighbor_agent = np.zeros((swarm_size, swarm_size))



#---------------------#
# Gradient Controller #
#---------------------#

Jn = []
rn = []

start_time = time.time()
t_elapsed = []

# Initialize the figure
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

for iter in range(max_iter):
    print('Iteration: ', iter)
    for i in range(swarm_size):
        print('Agent: ', i)
        for j in [x for x in range(swarm_size) if x != i]:
            print('Neighbor: ', j)
            rij = calculate_distance(swarm_position[i], swarm_position[j])
            aij = calculate_aij(alpha, delta, rij, r0, v)
            gij = calculate_gij(rij, r0)
            if aij >= PT:
                rho_ij = calculate_rho_ij(beta, v, rij, r0)
            else:
                rho_ij = 0
            
            qi = swarm_position[i, :]
            qj = swarm_position[j, :]
            normalized_distance = (qi - qj) / np.sqrt(np.linalg.norm(qi - qj))

            swarm_speed[i, 0] += rho_ij * normalized_distance[0]
            swarm_speed[i, 1] += rho_ij * normalized_distance[1]
            
            # Record the communication qualities, distances, and neighbor_agent matrix for Jn and rn performance plots
            phi_rij = gij * aij
            communication_qualities[i, j] = phi_rij
            communication_qualities[j, i] = phi_rij
            print('communication_qualities: ', communication_qualities)
            
            distances[i, j] = rij
            distances[j, i] = rij
            print('distances: ', distances)
            
            neighbor_agent[i, j] = aij
            neighbor_agent[j, i] = aij
            print('neighbor_agent: ', neighbor_agent)

        swarm_position[i, 0] += swarm_speed[i, 0]
        swarm_position[i, 1] += swarm_speed[i, 1]
        swarm_speed[i, 0] = 0
        swarm_speed[i, 1] = 0
        
    Jn.append(calculate_Jn(communication_qualities, neighbor_agent, PT))
    rn.append(calculate_rn(distances, neighbor_agent, PT))     
    t_elapsed.append(time.time() - start_time)
        
    #-----------------#
    # Starts plotting #
    #-----------------#
    
    plot_figures(axs, t_elapsed, Jn, rn, swarm_position, PT, communication_qualities, swarm_size, swarm_paths, node_colors, line_colors)
plt.show()