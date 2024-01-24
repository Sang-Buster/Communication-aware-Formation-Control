%% Communication-aware Formation Control
clear all;
close all;
clc;

% Initialize all parameters
max_iter    = 500;
h           = 1;
swarm_size  = 7;
alpha       = 10^(-5);                 % system parameter about antenna characteristics
delta       = 2;                       % required application data rate
beta        = alpha*(2^delta-1);
v           = 3;                       % path loss exponent
r0          = 5;                       % reference antenna near-field
PT          = 0.94;                    % reception probability threshold
rho_ij      = 0;
sim_speed   = 1;
communication_qualities = zeros(swarm_size, swarm_size);


%% ---Initialize Agents' Positions---
swarm = [
    -5,  14;
    -5, -19;
     0,   0;
    35,  -4;
    68,   0;
    72,  13;
    72, -18;
    ];

% % Define the range of coordinates
% x_min = 0;
% x_max = 50;
% y_min = 0;
% y_max = 50;
%
% % Generate random positions for the swarm
% x_coords = x_min + (x_max - x_min) * rand(swarm_size, 1);
% y_coords = y_min + (y_max - y_min) * rand(swarm_size, 1);
%
% % Combine the x and y coordinates into a single matrix
% swarm = [x_coords, y_coords];


% Print agents' initial positions
% for i = 1:swarm_size
%     for j = 1:2
%         fprintf("%d", swarm(i, j))
%     end
%     fprintf("\n")
% end


%% ---Initialize the velocity---
for j = 1:swarm_size
    speed(j,1) = 0;
    speed(j,2) = 0;
end


%% ---Performance Indicators---
t_Elapsed = 0;
Jn        = 0;
rn        = 0;

% Define the figure positions
figure_positions = [
    %Left Bottom Right Width Height
    200, 480, 500, 400;   % Position for Figure 1
    750, 480, 500, 400    % Position for Figure 2
    200, 10, 500, 400;    % Position for Figure 3
    750, 10, 500, 400;    % Position for Figure 4
    ];

% Plot Jn
figure(1)
Jn_Plot = plot(t_Elapsed, Jn);
set(gcf, 'Position', figure_positions(1, :));
xlabel('$t(s)$', 'Interpreter','latex', 'FontSize', 12, 'Rotation', 0)
ylabel('$J_n$', 'Interpreter','latex', 'FontSize', 12, 'Rotation', 0)
title('Average Communication Performance Indicator');
hold on
Jn_Text = text(t_Elapsed(end), Jn(end), sprintf('Jn: %.4f', Jn(end)), 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');

% Plot rn
figure(2)
rn_Plot = plot(t_Elapsed, rn);
set(gcf, 'Position', figure_positions(2, :));
xlabel('$t(s)$', 'Interpreter','latex', 'FontSize', 12, 'Rotation', 0)
ylabel('$r_n$', 'Interpreter','latex', 'FontSize', 12, 'Rotation', 0)
title('Average Distance Indicator');
hold on
rn_Text = text(t_Elapsed(end), rn(end), sprintf('rn: %.4f', rn(end)), 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');

drawnow                                         % Force a graphics refresh so that isn't counted in our elapsed time

tic

% Assign a different color to each edge-label pair
line_colors = rand(swarm_size, swarm_size, 3);
label_colors = line_colors;

% Define the color matrix with predefined colors
node_colors = [
    108 155 207;  % Light Blue
    247 147 39;   % Orange
    242 102 171;  % Light Pink
    255 217 90;   % Light Gold
    122 168 116;  % Green
    147 132 209;  % Purple
    245 80 80     % Red
    ] / 255;  % Divide by 255 to scale the RGB values to the [0, 1] range


%% ---Simulation---
for k=1:max_iter

    % Plot the node trace inside the loop
    figure(4)
    title('Node Trace');
    xlabel('$x$', 'Interpreter','latex', 'FontSize', 12, 'Rotation', 0)
    ylabel('$y$', 'Interpreter','latex', 'FontSize', 12, 'Rotation', 0)
    set(gcf, 'Position', figure_positions(4, :));
    hold on;

    % Plot all nodes as markers
    scatter(swarm(:, 1), swarm(:, 2), [], node_colors, 'filled');

    %--- Formation Scene + Node Trace---
    for i = 1:swarm_size
        % Plot the node trace
        swarm_trace(k, i, :) = swarm(i, :);

        trace_x = squeeze(swarm_trace(:, i, 1));
        trace_y = squeeze(swarm_trace(:, i, 2));
        plot(trace_x, trace_y, 'Color', node_colors(i, :));

        % Add arrows to indicate node movement
        arrow_x = trace_x(1:end-1);
        arrow_y = trace_y(1:end-1);
        arrow_dx = diff(trace_x);
        arrow_dy = diff(trace_y);

        % Normalize the arrow displacements
        arrow_magnitudes = sqrt(arrow_dx.^2 + arrow_dy.^2);
        max_arrow_magnitude = max(arrow_magnitudes);
        scaling_factor = 2;  % Adjust the scaling factor for arrow size
        normalized_arrow_dx = arrow_dx * scaling_factor / max_arrow_magnitude;
        normalized_arrow_dy = arrow_dy * scaling_factor / max_arrow_magnitude;

        quiver(arrow_x, arrow_y, normalized_arrow_dx, normalized_arrow_dy, 0, 'Color', node_colors(i, :), 'LineWidth', 1.5, 'MaxHeadSize', 2);
    end

    figure(3);
    clf; % Clear the figure
    set(gcf, 'Position', figure_positions(3, :));
    [img, map, alphachannel] = imread('drone','png');
    markersize = [3, 3];
    xlabel('$x$', 'Interpreter','latex', 'FontSize', 12, 'Rotation', 0)
    ylabel('$y$', 'Interpreter','latex', 'FontSize', 12, 'Rotation', 0)
    title('Formation Scene');
    axis equal;
    hold on;

    for k = 1:swarm_size
        x_low = swarm(k, 1) - markersize(1)/2;
        x_high = swarm(k, 1) + markersize(1)/2;
        y_low = swarm(k, 2) - markersize(2)/2;
        y_high = swarm(k, 2) + markersize(2)/2;

        imagesc([x_low x_high], [y_low y_high], img, 'AlphaData', alphachannel, 'CData', repmat(reshape(node_colors(k, :), [1 1 3]), [size(img, 1), size(img, 2), 1]));
    end


    %--- Formation Scene Edge+Label ---
    for i = 1:swarm_size
        for j= 1:swarm_size
            if i ~= j && communication_qualities(i, j) > 0

                hold on;

                % Get the line color and label color for the current pair
                line_color = line_colors(i, j, :);
                label_color = label_colors(i, j, :);

                % line_color = node_colors(i, :);
                % label_color = line_color;

                a1 = swarm(i,:);
                a2 = swarm(j,:);
                plot([a1(1), a2(1)], [a1(2), a2(2)], '--', 'Color', line_color);


                % Calculate the position for the label
                scaling_factor = 0.2;  % Adjust the scaling factor for the label positioning
                midpoint_x = (a1(1) + a2(1)) / 2;
                midpoint_y = (a1(2) + a2(2)) / 2;
                displacement_x = (a2(1) - a1(1)) * scaling_factor;
                displacement_y = (a2(2) - a1(2)) * scaling_factor;
                label_x = midpoint_x + displacement_x;
                label_y = midpoint_y + displacement_y + 2;

                % Add the number label at the midpoint of the line
                label = communication_qualities(i, j);
                label_str = sprintf('%.4f', label);

                text(label_x, label_y, label_str, 'HorizontalAlignment', 'center', 'Color', label_color);
                hold off;

                % Remove quality value for refresh
                communication_qualities(i, j) = 0;
                communication_qualities(j, i) = 0;
            end
        end
    end

    hold off;

    axis equal;

    %--- Controller ---
    for i=1:swarm_size
        rho_ij=0;
        for j=[1:(i-1),(i+1):swarm_size]
            rij=sqrt((swarm(i,1)-swarm(j,1))^2+(swarm(i,2)-swarm(j,2))^2);
            aij=exp(-alpha*(2^delta-1)*(rij/r0)^v);
            gij=rij/sqrt(rij^2+r0^2);
            if aij>=PT
                rho_ij=(-beta*v*rij^(v+2)-beta*v*(r0^2)*(rij^v)+r0^(v+2))*exp(-beta*(rij/r0)^v)/sqrt((rij^2+r0^2)^3);
            else
                rho_ij=0;
            end

            phi_rij=gij*aij;
            communication_qualities(i,j) = phi_rij;
            qi=[swarm(i,1),swarm(i,2)];
            qj=[swarm(j,1),swarm(j,2)];
            nd=(qi-qj)/sqrt(1+norm(qi-qj)*sim_speed);

            speed(i,1)=speed(i,1)+rho_ij*nd(1);
            speed(i,2)=speed(i,2)+rho_ij*nd(2);

        end
        swarm(i,1)=swarm(i,1)+speed(i,1)*h;
        swarm(i,2)=swarm(i,2)+speed(i,2)*h;
        speed(i,1)=0;
        speed(i,2)=0;

        t_Elapsed=cat(1, t_Elapsed, toc);

        %---Average Communication Performance Indicator---%
        Jn=cat(1, Jn, phi_rij);
        Jn=smooth(Jn);
        set(Jn_Plot, 'xdata', t_Elapsed, 'ydata', Jn);      % Plot Jn
        set(Jn_Text, 'Position', [t_Elapsed(end), Jn(end)], 'String', sprintf('Jn: %.4f', Jn(end)));
        pause(0);

        %---Average Neighboring Distance Indicator---%
        rn=cat(1, rn, rij);
        rn=smooth(rn);
        set(rn_Plot, 'xdata', t_Elapsed, 'ydata', rn);      % Plot rn
        set(rn_Text, 'Position', [t_Elapsed(end), rn(end)], 'String', sprintf('rn: %.4f', rn(end)));

        pause(0);

    end

    pause(0)

end

figure(1)
axis([0 max_iter+10  min(Jn) max(Jn)]);  % Update the limits for Figure 1

figure(2)
axis([0 max_iter+10 min(rn) max(rn)]);  % Update the limits for Figure 2

% Plot the final node trace outside the loop
figure(4)
hold on;
for i = 1:swarm_size
    trace_x = squeeze(swarm_trace(:, i, 1));
    trace_y = squeeze(swarm_trace(:, i, 2));
    plot(trace_x, trace_y);
end
hold off;
axis([x_min x_max y_min y_max]);
hold off;