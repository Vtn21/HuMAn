clear variables;
close all;
clc;

load("percentages.mat");

errors = [top90; top95; average; worst10; worst5]';
stdevs = [top90_std; top95_std; average_std; worst10_std; worst5_std]';
horizon = bin_limits(2:end);

x_labels = cell(1, 3);
for i = 1:length(bin_limits) - 1
    x_labels{i} = "[" + bin_limits(i) + " - " + bin_limits(i + 1) + "]";
end

figure;
b = bar(horizon, errors);
for k = 1:size(errors, 2)
    ctr(k, :) = bsxfun(@plus, b(1).XData, b(k).XOffset');
    ydt(k, :) = b(k).YData;
end
hold on;
errorbar(ctr, ydt, 0*stdevs', stdevs', "Color", [0 0 0], ...
         "LineStyle", "none", "LineWidth", 1);
xticks(horizon);
xticklabels(x_labels);
xlabel("Prediction horizon [s]");
ylabel("Mean absolute error [rad]");
legend("90 % smallest errors", "95 % smallest errors", "Average error", ...
       "10 % largest errors", "5 % largest errors", "Standard deviation", "Location", "northwest");
grid on;