% Communication-aware Formation Control Algorithm
% Author: Sang Xing
% Date: 1/26/2024

clear all;
close all;
clc;

% Import the helper functions from utils.m
helper = utils;


%---------------------------%
% Initialize all parameters %
%---------------------------%
max_iter    = 500;                     % maximum number of iterations
alpha       = 10^(-5);                 % system parameter about antenna characteristics
delta       = 2;                       % required application data rate
beta        = alpha*(2^delta-1);
v           = 3;                       % path loss exponent
r0          = 5;                       % reference distance between antenna near and far-field
PT          = 0.94;                    % reception probability threshold
rho_ij      = 0;

% Initialize agents' positions
swarm = [
    -5,  14;
    -5, -19;
     0,   0;
    35,  -4;
    68,   0;
    72,  13;
    72, -18;
    ];

% Initialize the swarm size
swarm_size = size(swarm, 1);

% Initialize the velocity of each agent
swarm_control_ui = zeros(swarm_size, swarm_size);

% Initialize the communication qualities
communication_qualities_matrix = zeros(swarm_size, swarm_size);

% Initialize the distances matrix to record the distances between agents
distances_matrix = zeros(swarm_size, swarm_size);

% Initialize the matrix to record the aij (near-field communication quality) value to indicate neighboring agents
neighbor_agent_matrix = zeros(swarm_size, swarm_size);

% Initialize performance indicators
Jn_list = [];                         % Store all average communication performance values
rn_list = [];                         % Store all average neighboring distance values

% Initialize timer
start_time = tic;
t_elapsed = [];

% Assign a different color to each edge-label pair
edge_colors = rand(swarm_size, swarm_size, 3);

% Assign a different color to each agent node
node_colors = [
    108 155 207;  % Light Blue
    247 147 39;   % Orange
    242 102 171;  % Light Pink
    255 217 90;   % Light Gold
    122 168 116;  % Green
    147 132 209;  % Purple
    245 80 80     % Red
    ] / 255;  % Divide by 255 to scale the RGB values to the [0, 1] range

% Define the figure positions
figure_positions = [
    %Left Bottom Right Width Height
    200, 480, 500, 400;   % Position for Figure 1
    750, 480, 500, 400    % Position for Figure 2
    200, 10, 500, 400;    % Position for Figure 3
    750, 10, 500, 400;    % Position for Figure 4
    ];

fig1 = figure('Position', figure_positions(1, :)); % Formation scene plot
fig2 = figure('Position', figure_positions(2, :)); % Node trace plot
fig3 = figure('Position', figure_positions(3, :)); % Jn plot
fig4 = figure('Position', figure_positions(4, :)); % rn plot


%----------------------%
% Formation Controller %
%----------------------%
for iter=1:max_iter
    fprintf("Iteration %d\n", iter);
    for i=1:swarm_size
        fprintf("Agent %d\n", i);
        for j=setdiff(1:swarm_size, i)
            fprintf("Neighbor %d\n", j);
            rij = helper.calculate_rij(swarm(i, :), swarm(j, :));
            aij = helper.calculate_aij(rij, alpha, delta, r0, v);
            gij = helper.calculate_gij(rij, r0);
            if aij>=PT
                rho_ij = helper.calculate_rho_ij(rij, r0, v, beta);
            else
                rho_ij=0;
            end

            qi=[swarm(i,1), swarm(i,2)];
            qj=[swarm(j,1), swarm(j,2)];
            eij=(qi-qj)/sqrt(1+norm(qi-qj));

            % Calculate the control input
            swarm_control_ui(i,1)=swarm_control_ui(i,1)+rho_ij*eij(1);
            swarm_control_ui(i,2)=swarm_control_ui(i,2)+rho_ij*eij(2);

            % Record the communication qualities, distances, and neighbor_agent matrix for Jn and rn performance plots
            phi_rij=gij*aij;
            communication_qualities_matrix(i,j) = phi_rij;
            communication_qualities_matrix(j, i) = phi_rij;
            % fprintf("communication_qualities_matrix(%d, %d) = %f\n", i, j, communication_qualities_matrix(i, j));

            distances_matrix(i, j) = rij;
            distances_matrix(j, i) = rij;
            % fprintf("distances_matrix(%d, %d) = %f\n", i, j, distances_matrix(i, j));

            neighbor_agent_matrix(i, j) = aij;
            neighbor_agent_matrix(j, i) = aij;
            % fprintf("neighbor_agent_matrix(%d, %d) = %f\n", i, j, neighbor_agent_matrix(i, j));
        end
        
        swarm(i,1)=swarm(i,1)+swarm_control_ui(i,1);
        swarm(i,2)=swarm(i,2)+swarm_control_ui(i,2);
        swarm_control_ui(i,1)=0;
        swarm_control_ui(i,2)=0;
        
        % Store the node trace
        swarm_trace(iter, i, :) = swarm(i, :);
    end

    Jn = helper.calculate_Jn(communication_qualities_matrix, neighbor_agent_matrix, PT);
    rn = helper.calculate_rn(distances_matrix, neighbor_agent_matrix, PT);    
    t_elapsed = [t_elapsed, toc(start_time)];

    % Append Jn and rn to their respective lists
    Jn_list = [Jn_list, Jn];
    rn_list = [rn_list, rn];

    % Plot the formation scene, node trace, Jn, and rn plots
    helper.plot_formation(swarm, swarm_size, neighbor_agent_matrix, PT, iter, node_colors, edge_colors, fig1, fig2, fig3, fig4, Jn_list, rn_list, t_elapsed, swarm_trace)

    % Check if the last 50 values in Jn are the same
    if length(Jn_list) > 49 && length(unique(round(Jn_list(end-49:end), 4))) == 1
        fprintf('Simulation stopped early: Jn values has converged.\n');
        break;
    end
end