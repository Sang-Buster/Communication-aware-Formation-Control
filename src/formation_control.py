import numpy as np
import matplotlib.pyplot as plt
import time

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

# Initialize the swarm velocity
swarm_speed = np.zeros((swarm_size, 2))

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

# Initialize the communication qualities matrix
communication_qualities = np.zeros((swarm_size, swarm_size))

# Initialize timer
t_elapsed = 0

# Performance indicators
Jn = 0
rn = 0




#---------------------#
# Gradient Controller #
#---------------------#
def calculate_distance(agent_i, agent_j):
    '''
        Calculate the distance between two agents
        
        Parameters:
            agent_i (list): The position of agent i
            agent_j (list): The position of agent j
        
        Returns:
            float: The distance between agent i and agent j
    '''
    return np.sqrt((agent_i[0] - agent_j[0])**2 + (agent_i[1] - agent_j[1])**2)


def calculate_aij(alpha, delta, rij, r0, v):
    '''
        Calculate the aij value
        
        Parameters:
            alpha (float): A system parameter about antenna characteristics
            delta (float): The required application data rate
            rij (float): The distance between two agents
            r0 (float): Reference distance value
            v (float): Path loss exponent
        
        Returns:
            float: The calculated aij (communication quality in antenna near-field) value
    '''
    return np.exp(-alpha*(2**delta-1)*(rij/r0)**v)


def calculate_gij(rij, r0):
    '''
        Calculate the gij value
        
        Parameters:
            rij (float): The distance between two agents
            r0 (float): Reference distance value
        
        Returns:
            float: The calculated gij (communication quality in antenna far-field) value
    '''
    return rij / np.sqrt(rij**2 + r0**2)


def calculate_rho_ij(beta, v, rij, r0):
    '''
        Calculate the rho_ij (the derivative of phi_ij) value
        
        Parameters:
            beta (float): alpha * (2**delta - 1)
            v (float): Path loss exponent
            rij (float): The distance between two agents
            r0 (float): Reference distance value
        
        Returns:
            float: The calculated rho_ij value
    '''
    return (-beta*v*rij**(v+2) - beta*v*(r0**2)*(rij**v) + r0**(v+2))*np.exp(-beta*(rij/r0)**v)/np.sqrt((rij**2 + r0**2)**3)

Jn = []
t_elapsed = []

for iter in range(max_iter):
    print('Iteration: ', iter)
    for i in range(swarm_size):
        print('Agent: ', i)
        for j in [x for x in range(swarm_size) if x != i]:
            rij = calculate_distance(swarm_position[i], swarm_position[j])
            aij = calculate_aij(alpha, delta, rij, r0, v)
            gij = calculate_gij(rij, r0)
            if aij >= PT:
                rho_ij = calculate_rho_ij(beta, v, rij, r0)
            else:
                rho_ij = 0

            phi_rij = gij * aij
            communication_qualities[i, j] = phi_rij
            qi = swarm_position[i, :]
            qj = swarm_position[j, :]
            normalized_distance = (qi - qj) / np.sqrt(1 + np.linalg.norm(qi - qj))

            swarm_speed[i, 0] += rho_ij * normalized_distance[0]
            swarm_speed[i, 1] += rho_ij * normalized_distance[1]

        swarm_position[i, 0] += swarm_speed[i, 0]
        swarm_position[i, 1] += swarm_speed[i, 1]
        swarm_position[i, 0] = 0
        swarm_position[i, 1] = 0

    Jn.append(calculate_Jn(communication_qualities))
    rn.append(calculate_rn())     
    t_elapsed = np.append(t_elapsed, time.time())
        
    #-----------------#
    # Starts plotting #
    #-----------------#
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    ########################
    # Plot formation scene #
    ########################
    axs[0, 0].set_title('Formation Scene')
    axs[0, 0].set_xlabel('x')
    axs[0, 0].set_ylabel('y', rotation=0)

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

    # Plot the nodes
    for i in range(swarm_position.shape[0]):
        axs[0, 0].scatter(*swarm_position[i], color=node_colors[i])

    # Assign a different color to each edge
    line_colors = np.random.rand(swarm_position.shape[0], swarm_position.shape[0], 3)

    # Plot the edges
    for i in range(swarm_position.shape[0]):
        for j in range(i+1, swarm_position.shape[0]):
            if communication_qualities[i, j] > PT:
                axs[0, 0].plot(*zip(swarm_position[i], swarm_position[j]), color=line_colors[i, j], linestyle='--')

    ###########################
    # Plot swarm trajectories #
    ###########################
    axs[0, 1].set_title('Swarm Trajectories')
    axs[0, 1].set_xlabel('x')
    axs[0, 1].set_ylabel('y', rotation=0)

    # Initialize the list of positions
    positions = []

    # Store the current positions
    for _ in range(max_iter):
        positions.append(swarm_position)

    # Convert the list of positions to a numpy array
    positions = np.array(positions)

    # Plot the trajectories
    for i in range(swarm_position.shape[0]):
        axs[0, 1].plot(positions[:, i, 0], positions[:, i, 1], color=node_colors[i])

    # Plot the initial positions
    axs[0, 1].scatter(positions[0, :, 0], positions[0, :, 1], color=node_colors)

    #######################
    # Plot Jn performance #
    #######################
    axs[1, 0].set_title('Average Communication Performance Indicator')
    axs[1, 0].plot(t_elapsed, Jn)
    axs[1, 0].set_xlabel('t(s)')
    axs[1, 0].set_ylabel('Jn', rotation=0)

    #######################
    # Plot rn performance #
    #######################
    axs[1, 1].set_title('Average Distance Performance Indicator')
    axs[1, 1].plot(t_elapsed, rn)
    axs[1, 1].set_xlabel('t(s)')
    axs[1, 1].set_ylabel('rn', rotation=0)

    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)
    plt.clf()
    
    time.sleep(0.01)

time.sleep(0.01)