clc; clear; close all;

%% ---------------- 1. User Parameters ----------------
videoFile = 'good_river.mp4';
realWidth  = 20;      % meters (x-direction, across river)
realLength = 10;      % meters (y-direction, along flow)
meanDepth  = 2.5;     % meters
surfaceToMeanFactor = 0.85;

% PIV Parameters
winSizes   = [128 64];  % Increased first pass to handle noise better
searchExp  = [2   1.5]; 
frameStep     = 2;      
maxSpeed_mps  = 3;      
res_px_m = 60; % Increasing this from 30 or 40 creates a denser rectified image
overlaps = [0.5 0.75]; % Increasing the second pass overlap to 0.75 creates more vectors
fprintf('--- Starting PIV Analysis (Manual Scaling Mode) ---\n');

%% ---------------- 2. Read Video ----------------
v = VideoReader(videoFile);
dt = 1 / v.FrameRate;
rawFrames = {};
k = 1;
while hasFrame(v) && k <= 100 % Processing 100 frames for testing
    frame = readFrame(v);
    if size(frame,3) == 3, I = rgb2gray(frame); else, I = frame; end
    rawFrames{k} = im2double(I);
    k = k + 1;
end
nFrames = numel(rawFrames);
fprintf('   > Read %d frames.\n', nFrames);

%% ---------------- 3. Rectification ROI ----------------
I0 = rawFrames{1};
figure('Name', 'ROI Selection'); imshow(I0, []);
title('Click 4 points CLOCKWISE around the water surface');
[xi, yi] = ginput(4); close;

% Physical rectangle in meters
xr = [0 realWidth realWidth 0];
yr = [0 0         realLength realLength];
tform = fitgeotrans([xi yi], [xr' yr'], 'projective');

% Define output reference
nRows = round(realLength * res_px_m);
nCols = round(realWidth  * res_px_m);
R = imref2d([nRows nCols], [0 realWidth], [0 realLength]);

%% ---------------- 4. Preprocessing (BG Sub & HPF) ----------------
fprintf('   > Preprocessing frames...\n');
avgFrame = zeros(R.ImageSize);
for k = 1:min(5, nFrames):nFrames
    avgFrame = avgFrame + imwarp(rawFrames{k}, tform, 'OutputView', R);
end
avgFrame = avgFrame / numel(1:min(5, nFrames):nFrames);

processedFrames = cell(1, nFrames);
for k = 1:nFrames
    I = imwarp(rawFrames{k}, tform, 'OutputView', R);
    I_sub = I - avgFrame; 
    % HPF: Highlight ripples, suppress low-freq noise
    I_hp = I_sub - imgaussfilt(I_sub, 5); 
    % Contrast Normalize
    processedFrames{k} = single((I_hp - min(I_hp(:))) / (max(I_hp(:)) - min(I_hp(:)) + 1e-6));
end
clear rawFrames;

%% ---------------- 5. PIV Analysis & Animation Setup ----------------
fprintf('   > Initializing Video Writer...\n');
outputVideo = VideoWriter('PIV_Result.mp4', 'MPEG-4');
outputVideo.FrameRate = (1/dt) / frameStep; % Adjust for frame separation
open(outputVideo);

[H, W] = size(processedFrames{1});
minWin = winSizes(end);
dx_min = minWin * (1 - overlaps(end));
[xg_base, yg_base] = meshgrid((minWin/2+1):dx_min:(W-minWin/2-1), ...
                              (minWin/2+1):dx_min:(H-minWin/2-1));

all_mean_speeds = [];
figAnim = figure('Name', 'PIV Animation Generator', 'Color', 'w'); 

for idx1 = 1:frameStep:(nFrames - frameStep)
    idx2 = idx1 + frameStep;
    I1 = processedFrames{idx1}; 
    I2 = processedFrames{idx2};
    
    % Initialize accumulated displacement for this pair
    U_tot = zeros(size(xg_base)); V_tot = zeros(size(yg_base));
    xg = xg_base; yg = yg_base;

    % --- Multi-pass Loop ---
    for p = 1:numel(winSizes)
        winSize = winSizes(p); halfWin = winSize / 2;
        searchW = winSize * searchExp(p); halfSearch = searchW / 2;
        
        if p > 1
            dx = winSize * (1-overlaps(p));
            [xg, yg] = meshgrid((halfWin+1):dx:(W-halfWin-1), (halfWin+1):dx:(H-halfWin-1));
            U_tot = imresize(U_tot, size(xg)); V_tot = imresize(V_tot, size(yg));
        end
        
        U = NaN(size(xg)); V = NaN(size(yg));
        for i = 1:numel(xg)
            xp = round(xg(i) + U_tot(i)); yp = round(yg(i) + V_tot(i));
            
            % Boundary Check
            r1 = round(yg(i)-halfWin+1):round(yg(i)+halfWin);
            c1 = round(xg(i)-halfWin+1):round(xg(i)+halfWin);
            r2 = round(yp-halfSearch+1):round(yp+halfSearch);
            c2 = round(xp-halfSearch+1):round(xp+halfSearch);
            
            if r1(1)<1 || r1(end)>H || c1(1)<1 || c1(end)>W || ...
               r2(1)<1 || r2(end)>H || c2(1)<1 || c2(end)>W
                continue; 
            end
            
            % FFT Cross-Correlation
            C = real(ifft2(fft2(I1(r1,c1), searchW, searchW) .* conj(fft2(I2(r2,c2), searchW, searchW))));
            [maxVal, idxM] = max(C(:)); [py, px] = ind2sub(size(C), idxM);
            
            % 3-Point Gaussian Sub-pixel Fit
            if py > 1 && py < searchW && px > 1 && px < searchW
                cp = log(maxVal+1e-9); 
                cxm = log(C(py,px-1)+1e-9); cxp = log(C(py,px+1)+1e-9);
                cym = log(C(py-1,px)+1e-9); cyp = log(C(py+1,px)+1e-9);
                U(i) = (px + (cxm-cxp)/(2*(cxm-2*cp+cxp))) - (halfSearch+0.5);
                V(i) = (py + (cym-cyp)/(2*(cym-2*cp+cyp))) - (halfSearch+0.5);
            else
                U(i) = px - (halfSearch+0.5); V(i) = py - (halfSearch+0.5);
            end
        end
        U_tot = U_tot + fillmissing(U,'constant',0); 
        V_tot = V_tot + fillmissing(V,'constant',0);
        
    end
    
    % --- CONVERT TO VELOCITY (m/s) BEFORE VALIDATION ---
    dt_pair = dt * (idx2 - idx1);
    u_ms = U_tot / res_px_m / dt_pair;
    v_ms = V_tot / res_px_m / dt_pair;
    speed = sqrt(u_ms.^2 + v_ms.^2); % <--- DEFINED HERE TO FIX ERROR
    
    % --- Validation & Filtering ---
    valid = (speed < maxSpeed_mps) & (speed > 0.01);
    u_plot = u_ms; v_plot = v_ms;
    u_plot(~valid) = NaN; v_plot(~valid) = NaN; 
    
    if any(valid(:))
        all_mean_speeds(end+1) = mean(speed(valid));
        fprintf('   > Frame Pair %d-%d: Mean Speed = %.3f m/s\n', idx1, idx2, all_mean_speeds(end));
    end

    % --- ANIMATION OVERLAY ---
    cla;
    imshow(processedFrames{idx1}, R, []); hold on;
    % Overlay valid vectors in red
    %quiver(xg/res_px_m, yg/res_px_m, u_plot, v_plot, 2, 'r', 'LineWidth', 1);
    % Temporary debug change in the animation section:
    quiver(xg/res_px_m, yg/res_px_m, u_ms, v_ms, 2, 'r'); % Plot ALL vectors, not just u_plot
    title(sprintf('River Velocity Overlay | Frame %d', idx1));
    drawnow;
    
    % Capture and write to video
    F = getframe(figAnim);
    writeVideo(outputVideo, F);
end

close(outputVideo);
fprintf('   > Animation saved as PIV_Result.mp4\n');

%% ---------------- 6. Final Results ----------------
if isempty(all_mean_speeds)
    error('No valid vectors detected. Check your ROI or frameStep.');
end

avg_V_surface = mean(all_mean_speeds);
A = realWidth * meanDepth;
Q = avg_V_surface * surfaceToMeanFactor * A;

fprintf('\n--- FINAL DISCHARGE CALCULATION ---\n');
fprintf('Overall Mean Surface Speed: %.4f m/s\n', avg_V_surface);
fprintf('Cross-Section Area:        %.2f m^2\n', A);
fprintf('Total Discharge Q:         %.3f m^3/s\n', Q);
%% ---------------- 6. Final Results ----------------
% (Keep your final Q calculations here)

%% ---------------- 6. Final Results ----------------
avg_V_surface = mean(all_mean_speeds);
Q = avg_V_surface * surfaceToMeanFactor * (realWidth * meanDepth);

fprintf('\n--- FINAL RESULTS ---\n');
fprintf('Mean Surface Speed: %.3f m/s\n', avg_V_surface);
fprintf('Estimated Discharge Q: %.2f m^3/s\n', Q);

figure; imshow(processedFrames{end}, R, []); hold on;
quiver(xg/res_px_m, yg/res_px_m, u_ms, v_ms, 2, 'r');
title('Final Velocity Field (Manual Scaling)');
xlabel('Width (m)'); ylabel('Flow Length (m)');