clear variables;
close all;
clc;

load("percentages.mat");

errors = [top90; top95; average; worst10; worst5]';
horizon = bin_limits(2:end);

figure;
bar(horizon, errors);
xticks(horizon);
xlabel("Prediction horizon [s]");
ylabel("Mean absolute error [rad]");
legend("90 % smallest errors", "95 % smallest errors", "Average error", ...
       "10 % largest errors", "5 % largest errors", "Location", "northwest");
grid on;