function f0_frames = convert_f0_to_frames(f0,tt,window)
% Takes the output of rack_f0.m and converts it into frames
% This uses a simple criteria - Look for F0 values in each window and take
% the average

max_length = tt(end);

n_windows = max_length/window;

frame_idx = floor(tt/window);

for i = 1:n_windows
    idx = find(frame_idx==i);
    current_f0 = f0(idx);
    f0_frames(i) = mean(current_f0);
end
