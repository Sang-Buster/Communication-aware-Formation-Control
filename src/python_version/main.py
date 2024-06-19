# Communication-aware Formation Control Algorithm
# Reproduce Author: Sang Xing
# Date: 01/26/2024

import numpy as np
import matplotlib.pyplot as plt
import time
import utils 


#---------------------------#
# Initialize all parameters #
#---------------------------#
max_iter = 500     # Maximum number of iterations
alpha = 10**(-5)   # Initialize system parameter about antenna characteristics
delta = 2          # Initialize required application data rate
beta = alpha*(2**delta-1)
v = 3              # Initialize path loss exponent
r0 = 5             # Initialize reference distance 
PT = 0.94          # Initialize the threshold value for communication quality


# Initialize agents' positions
swarm_position = np.array([
    [-5, 14],
    [-5, -19],
    [0, 0],
    [35, -4],
    [68, 0],
    [72, 13],
    [72, -18]
], dtype=float)

# Initialize the swarm destination
swarm_destination = np.array([35, 100], dtype=float)

# Initialize the swarm size
swarm_size = swarm_position.shape[0]

# Initialize the swarm control
swarm_control_ui = np.zeros((swarm_size, 2))

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

# Initialize the matrix to record the aij (near-field communication quality) value to indicate neighboring agents
neighbor_agent_matrix = np.zeros((swarm_size, swarm_size))

# Initialize the list for swarm trajectory plot
swarm_paths = []

# Assign node (aka agent) color
node_colors = [
    [108/255, 155/255, 207/255],  # Light Blue
    [247/255, 147/255, 39/255],   # Orange
    [242/255, 102/255, 171/255],  # Light Pink
    [255/255, 217/255, 90/255],   # Light Gold
    [122/255, 168/255, 116/255],  # Green
    [147/255, 132/255, 209/255],  # Purple
    [245/255, 80/255, 80/255]     # Red
]

# Assign edge (aka communication links between agents) color
line_colors = np.random.rand(swarm_position.shape[0], swarm_position.shape[0], 3)

# Initialize a flag for Jn convergence
Jn_converged = False

# Initialize the figure
fig, axs = plt.subplots(2, 2, figsize=(10, 10))


#----------------------#
# Formation Controller #
#----------------------#
for iter in range(max_iter):
    print('Iteration: ', iter)
    for i in range(swarm_size):
        # print('Agent: ', i)
        for j in [x for x in range(swarm_size) if x != i]:
            # print('Neighbor: ', j)
            rij = utils.calculate_distance(swarm_position[i], swarm_position[j])
            aij = utils.calculate_aij(alpha, delta, rij, r0, v)
            gij = utils.calculate_gij(rij, r0)
            if aij >= PT:
                # rho_ij is the derivative of upper case phi_ij (which is the same as the lower case phi_ij in powerpoint)
                rho_ij = utils.calculate_rho_ij(beta, v, rij, r0)
            else:
                rho_ij = 0
            
            qi = swarm_position[i, :]
            qj = swarm_position[j, :]
            eij = (qi - qj) / np.sqrt(rij)
            
            ###########################
            # Formation control input #
            ###########################
            swarm_control_ui[i, 0] += rho_ij * eij[0]
            swarm_control_ui[i, 1] += rho_ij * eij[1]
            
            # Record the communication qualities, distances, and neighbor_agent matrices for Jn and rn performance plots
            phi_rij = gij * aij
            communication_qualities_matrix[i, j] = phi_rij
            communication_qualities_matrix[j, i] = phi_rij
            
            distances_matrix[i, j] = rij
            distances_matrix[j, i] = rij
            
            neighbor_agent_matrix[i, j] = aij
            neighbor_agent_matrix[j, i] = aij

        swarm_position[i, 0] += swarm_control_ui[i, 0]
        swarm_position[i, 1] += swarm_control_ui[i, 1]
        swarm_control_ui[i, 0] = 0
        swarm_control_ui[i, 1] = 0
        
        Jn_new = utils.calculate_Jn(communication_qualities_matrix, neighbor_agent_matrix, PT)
        rn_new = utils.calculate_rn(distances_matrix, neighbor_agent_matrix, PT)
    
    # Record the performance indicators
    Jn.append(round(Jn_new, 4))
    rn.append(round(rn_new, 4))  

    # Check if the last 20 values in Jn are the same, if so, the formation is completed, and the swarm starts to reach the destination
    if len(Jn) > 19 and len(set(Jn[-20:])) == 1:
        if not Jn_converged:
            print(f"Formation completed: Jn values has converged in {round(t_elapsed[-1], 2)} seconds {iter-20} iterations.")
            Jn_converged = True
            break

    # Record the elapsed time
    t_elapsed.append(time.time() - start_time)
    
    # Starts plotting
    utils.plot_figures_task1(axs, t_elapsed, Jn, rn, swarm_position, PT, communication_qualities_matrix, swarm_size, swarm_paths, node_colors, line_colors)
plt.show()
