# Communicationa-aware Formation Control Algorithm
# Author: Sang Xing
# Date: 01/26/2024

import numpy as np
import matplotlib.pyplot as plt
import time
import utils 

#---------------------------#
# Initialize all parameters #
#---------------------------#

# Initialize the maximum iteration
max_iter = 500

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

# Initialize the swarm control
swarm_control_ui = np.zeros((swarm_size, 2))

# Initialize system parameter about antenna characteristics
alpha = 10**(-5)  

# Initialize required application data rate
delta = 2  
beta = alpha*(2**delta-1)

# Initialize path loss exponent
v = 3  

# Initialize reference distance 
r0 = 5  

# Initialize reception probability threshold
PT = 0.94

# Initialize performance indicators
Jn = []
rn = []

# Initialize timer
start_time = time.time()
t_elapsed = []

# Initialize the communication qualities matrix to record the communication qualities between agents
communication_qualities_matrix = np.zeros((swarm_size, swarm_size))

# Initialize the distances matrix to record the distances between agents
distances_matrix = np.zeros((swarm_size, swarm_size))

# Initialize the rho matrix to record the rho_ij value (combined near and far-field quality) to indicate neighboring agents
neighbor_agent_matrix = np.zeros((swarm_size, swarm_size))

# Initialize the list for swarm trajectory plot
swarm_paths = []

# Assign node (aka agent) color
node_colors = np.random.rand(swarm_position.shape[0], 3)

# Assign edge (aka communication links between agents) color
line_colors = np.random.rand(swarm_position.shape[0], swarm_position.shape[0], 3)


#----------------------#
# Formation Controller #
#----------------------#

# Initialize the figure
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

for iter in range(max_iter):
    print('Iteration: ', iter)
    for i in range(swarm_size):
        print('Agent: ', i)
        for j in [x for x in range(swarm_size) if x != i]:
            print('Neighbor: ', j)
            rij = utils.calculate_distance(swarm_position[i], swarm_position[j])
            aij = utils.calculate_aij(alpha, delta, rij, r0, v)
            gij = utils.calculate_gij(rij, r0)
            if aij >= PT:
                rho_ij = utils.calculate_rho_ij(beta, v, rij, r0)
            else:
                rho_ij = 0
            
            qi = swarm_position[i, :]
            qj = swarm_position[j, :]
            eij = (qi - qj) / np.sqrt(rij)
            
            # Calculate the control input
            swarm_control_ui[i, 0] += rho_ij * eij[0]
            swarm_control_ui[i, 1] += rho_ij * eij[1]
            
            # Record the communication qualities, distances, and neighbor_agent matrices for Jn and rn performance plots
            phi_rij = gij * aij
            communication_qualities_matrix[i, j] = phi_rij
            communication_qualities_matrix[j, i] = phi_rij
            # print('communication_qualities matrix: ', communication_qualities_matrix)
            
            distances_matrix[i, j] = rij
            distances_matrix[j, i] = rij
            # print('distances matrix: ', distances_matrix)
            
            neighbor_agent_matrix[i, j] = aij
            neighbor_agent_matrix[j, i] = aij
            # print('neighbor_agent matrix: ', neighbor_agent_matrix)

        swarm_position[i, 0] += swarm_control_ui[i, 0]
        swarm_position[i, 1] += swarm_control_ui[i, 1]
        swarm_control_ui[i, 0] = 0
        swarm_control_ui[i, 1] = 0
        
    Jn.append(utils.calculate_Jn(communication_qualities_matrix, neighbor_agent_matrix, PT))
    rn.append(utils.calculate_rn(distances_matrix, neighbor_agent_matrix, PT))     
    t_elapsed.append(time.time() - start_time)
        
    #-----------------#
    # Starts plotting #
    #-----------------#
    
    utils.plot_figures(axs, t_elapsed, Jn, rn, swarm_position, PT, communication_qualities_matrix, swarm_size, swarm_paths, node_colors, line_colors)

    # Check if the last 50 values in Jn are the same
    if len(Jn) > 49 and len(set(Jn[-50:])) == 1:
        print("Simulation stopped early: Jn values has converged.")
        break
plt.show()