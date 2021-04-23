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
xl = xline(mean(mean_s), "--r", "Average", "LineWidth", 2);
xl.LabelVerticalAlignment = "bottom";
grid;
xlabel("Mean absolute error [rad]");