clear variables;
close all;
clc;

load("percentages.mat");

errors = [top90; top95; average; worst10; worst5]';
horizon = bin_limits(2:end);

x_labels = cell(1, 3);
for i = 1:length(bin_limits) - 1
    x_labels{i} = "[" + bin_limits(i) + " - " + bin_limits(i + 1) + "]";
end

figure;
bar(horizon, errors);
xticks(horizon);
xticklabels(x_labels);
xlabel("Prediction horizon [s]");
ylabel("Mean absolute error [rad]");
legend("90 % smallest errors", "95 % smallest errors", "Average error", ...
       "10 % largest errors", "5 % largest errors", "Location", "northwest");
grid on;