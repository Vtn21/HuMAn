clear variables;
close all;
clc;

% Combinations of skeleton_input and skeleton_prediction
skeleton = ["full_body", "full_body", "full_body", "full_body", "legs_arms", "legs_arms", "legs_arms", "legs", "arms";
            "full_body", "legs_arms",      "legs",      "arms", "legs_arms",      "legs",      "arms", "legs", "arms"];
            
for item = skeleton
    % Load data from MAT file
    load(strcat('skeleton_input=', item(1), ' skeleton_prediction=', item(2), '.mat'));
    % Convert single to double
    sampling_time = double(sampling_time);
    horizon_time = double(horizon_time);
    % Create a linearly spaced mesh
    xv = linspace(0.01, max(sampling_time), 100);
    yv = linspace(min(horizon_time), max(horizon_time), 100);
    [X, Y] = meshgrid(xv, yv);
    % Interpolate mean values over the grid
    MEAN = griddata(sampling_time, horizon_time, mean, X, Y, "natural");
    STDEV = griddata(sampling_time, horizon_time, stdev, X, Y, "natural");
    % Remove some NaN values
    keep = ~isnan(MEAN(:, end));
    X = X(keep, :);
    Y = Y(keep, :);
    MEAN = MEAN(keep, :);
    STDEV = STDEV(keep, :);
    % Figure
    f = figure("Name", strcat(item(1), " ", item(2)));
    % 2D mean
    subplot(1, 2, 1);
    surf(X, Y, MEAN, "EdgeColor", "none");
    shading interp;
    colormap jet;
    colorbar;
    view(2);
    xlabel("Sampling time [s]");
    ylabel("Prediction horizon [s]");
    title("Mean absolute error [rad]");
    axis tight;
    % 2D standard deviation
    subplot(1, 2, 2);
    surf(X, Y, STDEV, "EdgeColor", "none");
    shading interp;
    colormap jet;
    colorbar;
    view(2);
    xlabel("Sampling time [s]");
    ylabel("Prediction horizon [s]");
    title("Standard deviation [rad]");
    axis tight;
    % Window size
    f.Position(3:4) = [840 360];
end

% Combinations of skeleton_prediction and skeleton_input
skeleton = {"legs_arms", "legs", "arms"; 
            ["legs_arms", "full_body"; "Legs and arms", "Full body"; "#D95319", "#0072BD"], ...
            ["legs", "legs_arms", "full_body"; "Legs only", "Legs and arms", "Full body"; "#EDB120", "#D95319", "#0072BD"], ...
            ["arms", "legs_arms", "full_body"; "Arms only", "Legs and arms", "Full body"; "#7E2F8E", "#D95319", "#0072BD"]};

for item = skeleton
    f = figure("Name", item{1});
    axis;
    hold on;
    for input = item{2}
        % Load data from MAT file
        load(strcat('skeleton_input=', input(1), ' skeleton_prediction=', item{1}, '.mat'));
        % Convert single to double
        sampling_time = double(sampling_time);
        horizon_time = double(horizon_time);
        % Create a linearly spaced mesh
        xv = linspace(0.01, max(sampling_time), 100);
        yv = linspace(min(horizon_time), max(horizon_time), 100);
        [X, Y] = meshgrid(xv, yv);
        % Interpolate mean values over the grid
        MEAN = griddata(sampling_time, horizon_time, mean, X, Y, "natural");
        % Remove some NaN values
        keep = ~isnan(MEAN(:, end));
        X = X(keep, :);
        Y = Y(keep, :);
        MEAN = MEAN(keep, :);
        % Plot
        surf(X, Y, MEAN, "EdgeColor", "none", "FaceColor", input(3), "DisplayName", input(2), "FaceAlpha", 0.9, "FaceLighting", "gouraud");
        light("Position", [0 -1 0]);
        xlabel("Sampling time [s]");
        ylabel("Prediction horizon [s]");
        zlabel("Mean absolute error [rad]");
    end
    view(68, 5);
    legend("Location", "northwest");
    grid;
    % Window size
    f.Position(3:4) = [600 350];
end








