clear variables;
close all;
clc;

load("joints.mat");

for i = 1:length(joints)
    joints_str(i) = strtrim(string(joints(i, :)));
end

[mean_s, idx] = sort(mean);
stdev_s = stdev(idx);
joints_s = joints_str(idx);

clear mean;

x = categorical(joints_s);
x = reordercats(x, joints_s);

figure;
b = barh(x, mean_s, "FaceColor", "flat");
hold on;
yticklabels(joints_s);
xl = xline(mean(mean_s), "--r", "LineWidth", 2);
er = errorbar(mean_s, x, 0*stdev_s, stdev_s, "horizontal", "Color", [0 0 0], "LineStyle", "none", "LineWidth", 1);
grid;
xlabel("Absolute error [rad]");
legend("Mean absolute error", "Average MAE", "Standard deviation", "Location", "southeast");
