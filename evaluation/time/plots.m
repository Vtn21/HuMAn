clear variables;
close all;
clc;

load("time.mat");

width = 2.0;

figure;
plot(mean, "LineWidth", width);
hold on;
plot(mean + stdev, "LineWidth", width);
grid on;
axis tight;
ylim([0.01 0.09]);
legend("\mu", "\mu + \sigma");
xlabel("Frames");
ylabel("Absolute error [rad]");