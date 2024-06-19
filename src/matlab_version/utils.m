classdef utils
    methods (Static)
        function rij = calculate_rij(swarm_i, swarm_j)
            % Calculate the rij (distance) value
            % 
            % Parameters:
            %   swarm_i (matrix): The swarm_i matrix
            %   swarm_j (matrix): The swarm_j matrix
            % % 
            % Returns:
            %   float: The calculated rij value
            
            rij = sqrt((swarm_i(1) - swarm_j(1))^2 + (swarm_i(2) - swarm_j(2))^2);
        end


        function aij = calculate_aij(rij, alpha, delta, r0, v)
            % Calculate the aij (communication far-field quality) value
            %
            % Parameters:
            %   rij (float): Distance between agent i and j
            %   alpha (float): System parameter about antenna characteristics
            %   delta (float): Required application data rate
            %   r0 (float): Reference distance between antenna near and far-field
            %   v (float): Path loss exponent
            %
            % Returns:
            %   float: The calculated aij value

            aij = exp(-alpha * (2^delta - 1) * (rij / r0)^v);
        end


        function gij = calculate_gij(rij, r0)
            % Calculate the gij (communication near-field quality) value
            %
            % Parameters:
            %   rij (float): Distance between agent i and j
            %   r0 (float): Reference distance between antenna near and far-field
            %
            % Returns:
            %   float: The calculated gij value

            gij = rij / sqrt(rij^2 + r0^2);
        end


        function rho_ij = calculate_rho_ij(rij, r0, v, beta)
            % Calculate the rho_ij (derivative of phi_ij, where phi_ij is the interaction model that outputs combined near and far-field communication quality) value
            %
            % Parameters:
            %   rij (float): Distance between agent i and j
            %   r0 (float): Reference distance between antenna near and far-field
            %   v (float): Path loss exponent
            %   beta (float): alpha * (2**delta - 1)
            %
            % Returns:
            %   float: The calculated rho_ij value

            rho_ij = (-beta * v * rij^(v+2) - beta * v * (r0^2) * (rij^v) + r0^(v+2)) * exp(-beta * (rij / r0)^v) / sqrt((rij^2 + r0^2)^3);
        end


        function Jn = calculate_Jn(communication_qualities, neighbor_agent, PT)
            % Calculate the Jn (average communication quality performance indicator) value
            %
            % Parameters:
            %   communication_qualities (matrix): The communication qualities matrix among agents
            %   neighbor_agent (matrix): The neighbor_agent matrix which is adjacency matrix of aij value
            %   PT (float): The reception probability threshold
            %
            % Returns:
            %   float: The calculated Jn value

            total_communication_quality = 0;
            total_neighbors = 0;
            swarm_size = size(communication_qualities, 1);
            for i = 1:swarm_size
                for j=setdiff(1:swarm_size, i)
                    if neighbor_agent(i, j) > PT
                        total_communication_quality = total_communication_quality + communication_qualities(i, j);
                        total_neighbors = total_neighbors + 1;
                    end
                end
            end
            Jn = total_communication_quality / total_neighbors;
        end


        function rn = calculate_rn(distances, neighbor_agent, PT)
            % Calculate the rn (average neighboring distance performance indicator) value
            %
            % Parameters:
            %   distances (matrix): The distances matrix among agents
            %   neighbor_agent (matrix): The neighbor_agent matrix which is adjacency matrix of aij value
            %   PT (float): The reception probability threshold
            %
            % Returns:
            %   float: The calculated rn value

            total_distance = 0;
            total_neighbors = 0;
            swarm_size = size(distances, 1);
            for i = 1:swarm_size
                for j=setdiff(1:swarm_size, i)
                    if neighbor_agent(i, j) > PT
                        total_distance = total_distance + distances(i, j);
                        total_neighbors = total_neighbors + 1;
                    end
                end
            end
            rn = total_distance / total_neighbors;
        end


        function closest_agent_index = find_closest_agent(swarm_position, swarm_centroid)
            % Find the index of the agent with the minimum distance to the destination
            %
            % Parameters:
            %   swarm_position (double matrix): The positions of the swarm
            %   swarm_centroid (double matrix): The centroid of the swarm
            %
            % Returns:
            %   int: The index of the agent with the minimum distance to the destination

            % Calculate the Euclidean distance from each agent to the destination
            distances_matrix = sqrt(sum((swarm_position - swarm_centroid).^2, 2));
            
            % Find the index of the agent with the minimum distance
            [~, closest_agent_index] = min(distances_matrix);
        end
        

        function plot_formation_task1(swarm, swarm_size, neighbor_agent_matrix, PT, iter, node_colors, edge_colors, fig1, fig2, fig3, fig4, Jn_list, rn_list, t_elapsed, swarm_trace)
            % Plot the formation scene, node trace, Jn, and rn\
            %
            % Parameters:
            %   swarm (matrix): The posiotions of swarm
            %   swarm_size (int): The number of agents in the swarm
            %   neighbor_agent_matrix (matrix): The neighbor_agent matrix which is adjacency matrix of swarm in aij value
            %   PT (float): The reception probability threshold
            %   iter (int): The current iteration
            %   node_colors (matrix): The node_colors 
            %   edge_colors (matrix): The edge_colors 
            %   fig1 (figure): The figure for formation scene
            %   fig2 (figure): The figure for node trace
            %   fig3 (figure): The figure for Jn plot
            %   fig4 (figure): The figure for rn plot
            %   Jn_list (matrix): All of Jn values over time
            %   rn_list (matrix): All of rn values over time 
            %   t_elapsed (matrix): The elapsed time
            %   swarm_trace (matrix): The trace of swarm (in iteration, swarm_swize's x-coordinates, swarm_size y-coordinates)
            %
            % Returns:
            %   None
            
            %-----------------------%
            %  Formation scene plot %
            %-----------------------%
            figure(fig1);
            clf; 
            xlabel('$x$', 'Interpreter','latex', 'FontSize', 12, 'Rotation', 0)
            ylabel('$y$', 'Interpreter','latex', 'FontSize', 12, 'Rotation', 0)
            title('Formation Scene');
            hold on;
            
            for i = 1:swarm_size
                plot(swarm(i, 1), swarm(i, 2), '.', 'Color', node_colors(i, :), 'MarkerSize', 20);
                hold on;
            end

            % Draw edges between agents if they're above the PT value
            for i = 1:swarm_size
                for j = i+1:swarm_size
                    if neighbor_agent_matrix(i, j) >= PT
                        line([swarm(i, 1), swarm(j, 1)], [swarm(i, 2), swarm(j, 2)], 'Color', edge_colors(i, j, :), 'LineStyle', '--');
                    end
                end
            end
            
            xlim([min(swarm(:, 1)) - 10, max(swarm(:, 1)) + 10]);
            ylim([min(swarm(:, 2)) - 10, max(swarm(:, 2)) + 10]);            
            axis equal;
            drawnow;
            
            %------------------%
            %  Node trace plot %
            %------------------%
            figure(fig2);
            title('Node Trace');
            xlabel('$x$', 'Interpreter','latex', 'FontSize', 12, 'Rotation', 0)
            ylabel('$y$', 'Interpreter','latex', 'FontSize', 12, 'Rotation', 0)
            hold on;

            % Plot the trace of each agent
            for i = 1:swarm_size
                trace_x = squeeze(swarm_trace(:, i, 1));
                trace_y = squeeze(swarm_trace(:, i, 2));

                % Plot the initial position as a dot
                if iter == 1
                    plot(trace_x(1), trace_y(1), '.', 'Color', node_colors(i, :), 'MarkerSize', 20);
                end

                % Plot the trace as a line with arrows
                if iter > 1  % Only plot the trace if there is more than one point
                    quiver(trace_x(1:end-1), trace_y(1:end-1), diff(trace_x), diff(trace_y), 0, 'Color', node_colors(i, :), 'LineWidth', 1.5, 'MaxHeadSize', 2);
                end
            end

            % Plot added/removed agent
            % Your code here...
            
            xlim([-10, 80]);  
            ylim([-30, 30]);        
            axis equal;
            drawnow;
            
            %---------%
            % Jn plot %
            %---------%
            figure(fig3);
            plot(t_elapsed, Jn_list); 
            xlim([0, t_elapsed(end) + 0.2 * t_elapsed(end)]);  % Add 20% padding to the right
            if ~isempty(Jn_list)  % Check if Jn_list is not empty
                if min(Jn_list) ~= max(Jn_list)  % Check if min and max are different
                    ylim([min(Jn_list) - 0.001, max(Jn_list) + 0.001]);  
                end
            end
            text(t_elapsed(end), Jn_list(end), sprintf(' %.4f', Jn_list(end)), 'VerticalAlignment', 'top')
            xlabel('$t(s)$', 'Interpreter','latex', 'FontSize', 12, 'Rotation', 0)
            ylabel('$J_n$', 'Interpreter','latex', 'FontSize', 12, 'Rotation', 0)
            title('Average Communication Performance Indicator');
            drawnow;

            %---------%
            % rn plot %
            %---------%
            figure(fig4);
            plot(t_elapsed, rn_list);  
            xlim([0, t_elapsed(end) + 0.2 * t_elapsed(end)]);  % Add 20% padding to the right
            if ~isempty(rn_list)  % Check if rn_list is not empty
                if min(rn_list) ~= max(rn_list)  % Check if min and max are different
                    ylim([min(rn_list) - 1, max(rn_list) + 1]);  
                end
            end
            text(t_elapsed(end), rn_list(end), sprintf(' %.4f', rn_list(end)), 'VerticalAlignment', 'top')
            xlabel('$t(s)$', 'Interpreter','latex', 'FontSize', 12, 'Rotation', 0)
            ylabel('$r_n$', 'Interpreter','latex', 'FontSize', 12, 'Rotation', 0)
            title('Average Distance Indicator');
            drawnow;
        end

        
        function plot_formation_task2(swarm, swarm_destination, swarm_size, neighbor_agent_matrix, PT, iter, node_colors, edge_colors, fig1, fig2, fig3, fig4, Jn_list, rn_list, t_elapsed, swarm_trace)
            % Plot the formation scene, node trace, Jn, and rn\
            %
            % Parameters:
            %   swarm (matrix): The posiotions of swarm
            %   swwarm_destination (matrix): The destination of swarm
            %   swarm_size (int): The number of agents in the swarm
            %   neighbor_agent_matrix (matrix): The neighbor_agent matrix which is adjacency matrix of swarm in aij value
            %   PT (float): The reception probability threshold
            %   iter (int): The current iteration
            %   node_colors (matrix): The node_colors 
            %   edge_colors (matrix): The edge_colors 
            %   fig1 (figure): The figure for formation scene
            %   fig2 (figure): The figure for node trace
            %   fig3 (figure): The figure for Jn plot
            %   fig4 (figure): The figure for rn plot
            %   Jn_list (matrix): All of Jn values over time
            %   rn_list (matrix): All of rn values over time 
            %   t_elapsed (matrix): The elapsed time
            %   swarm_trace (matrix): The trace of swarm (in iteration, swarm_swize's x-coordinates, swarm_size y-coordinates)
            %
            % Returns:
            %   None
            
            %-----------------------%
            %  Formation scene plot %
            %-----------------------%
            figure(fig1);
            clf; 
            xlabel('$x$', 'Interpreter','latex', 'FontSize', 12, 'Rotation', 0)
            ylabel('$y$', 'Interpreter','latex', 'FontSize', 12, 'Rotation', 0)
            title('Formation Scene');
            hold on;
            
            for i = 1:swarm_size
                plot(swarm(i, 1), swarm(i, 2), '.', 'Color', node_colors(i, :), 'MarkerSize', 20);
                hold on;
            end

            % Draw edges between agents if they're above the PT value
            for i = 1:swarm_size
                for j = i+1:swarm_size
                    if neighbor_agent_matrix(i, j) >= PT
                        line([swarm(i, 1), swarm(j, 1)], [swarm(i, 2), swarm(j, 2)], 'Color', edge_colors(i, j, :), 'LineStyle', '--');
                    end
                end
            end
            
            % Plot the destination as a black square
            square_size = 3;
            square_x = swarm_destination(1) - square_size / 2;
            square_y = swarm_destination(2) - square_size / 2;
            rectangle('Position', [square_x, square_y, square_size, square_size], 'EdgeColor', 'k', 'LineWidth', 2);
            text(swarm_destination(1), swarm_destination(2) + 3, 'Destination', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');

            % Plot obstacles
            % Your code here...

            xlim([min([swarm(:, 1); swarm_destination(1)]) - 10, max([swarm(:, 1); swarm_destination(1)]) + 10]);
            ylim([min([swarm(:, 2); swarm_destination(2)]) - 10, max([swarm(:, 2); swarm_destination(2)]) + 10]);            
            axis equal;
            drawnow;
            
            %------------------%
            %  Node trace plot %
            %------------------%
            figure(fig2);
            title('Node Trace');
            xlabel('$x$', 'Interpreter','latex', 'FontSize', 12, 'Rotation', 0)
            ylabel('$y$', 'Interpreter','latex', 'FontSize', 12, 'Rotation', 0)
            hold on;

            % Plot the trace of each agent
            for i = 1:swarm_size
                trace_x = squeeze(swarm_trace(:, i, 1));
                trace_y = squeeze(swarm_trace(:, i, 2));

                % Plot the initial position as a dot
                if iter == 1
                    plot(trace_x(1), trace_y(1), '.', 'Color', node_colors(i, :), 'MarkerSize', 20);
                end

                % Plot the trace as a line with arrows
                if iter > 1  % Only plot the trace if there is more than one point
                    quiver(trace_x(1:end-1), trace_y(1:end-1), diff(trace_x), diff(trace_y), 0, 'Color', node_colors(i, :), 'LineWidth', 1.5, 'MaxHeadSize', 2);
                end
            end
            
            % Plot the destination as a black square
            square_size = 3;
            square_x = swarm_destination(1) - square_size / 2;
            square_y = swarm_destination(2) - square_size / 2;
            rectangle('Position', [square_x, square_y, square_size, square_size], 'EdgeColor', 'k', 'LineWidth', 2);
            text(swarm_destination(1), swarm_destination(2) + 3, 'Destination', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');

            % Plot obstacles
            % Your code here...

            % Adjust the x and y limits to include the destination
            xlim([-20, 80]);  
            ylim([-30, 130]);            
            axis equal;
            drawnow;
            
            %---------%
            % Jn plot %
            %---------%
            figure(fig3);
            plot(t_elapsed, Jn_list); 
            xlim([0, t_elapsed(end) + 0.2 * t_elapsed(end)]);  % Add 20% padding to the right
            if ~isempty(Jn_list)  % Check if Jn_list is not empty
                if min(Jn_list) ~= max(Jn_list)  % Check if min and max are different
                    ylim([min(Jn_list) - 0.001, max(Jn_list) + 0.001]);  
                end
            end
            text(t_elapsed(end), Jn_list(end), sprintf(' %.4f', Jn_list(end)), 'VerticalAlignment', 'top')
            xlabel('$t(s)$', 'Interpreter','latex', 'FontSize', 12, 'Rotation', 0)
            ylabel('$J_n$', 'Interpreter','latex', 'FontSize', 12, 'Rotation', 0)
            title('Average Communication Performance Indicator');
            drawnow;

            %---------%
            % rn plot %
            %---------%
            figure(fig4);
            plot(t_elapsed, rn_list);  
            xlim([0, t_elapsed(end) + 0.2 * t_elapsed(end)]);  % Add 20% padding to the right
            if ~isempty(rn_list)  % Check if rn_list is not empty
                if min(rn_list) ~= max(rn_list)  % Check if min and max are different
                    ylim([min(rn_list) - 1, max(rn_list) + 1]);  
                end
            end
            text(t_elapsed(end), rn_list(end), sprintf(' %.4f', rn_list(end)), 'VerticalAlignment', 'top')
            xlabel('$t(s)$', 'Interpreter','latex', 'FontSize', 12, 'Rotation', 0)
            ylabel('$r_n$', 'Interpreter','latex', 'FontSize', 12, 'Rotation', 0)
            title('Average Distance Indicator');
            drawnow;
        end
    end
end
