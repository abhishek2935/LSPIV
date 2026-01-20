clc; clear; close all;

%% ---------------- 1. User Parameters ----------------
videoFile = 'good_river.mp4';
realWidth  = 20;      
realLength = 10;      
meanDepth  = 2.5;     
surfaceToMeanFactor = 0.85;

% PIV Parameters
winSizes   = [128 64];  
searchExp  = [2.5 2.0]; 
frameStep  = 5;         % Optimization: Larger step reduces noise sensitivity
maxSpeed_mps = 3.5;     
res_px_m   = 60;        
overlaps   = [0.5 0.75];

fprintf('--- Starting Optimized PIV Analysis ---\n');

%% ---------------- 2. Read Video ----------------
v = VideoReader(videoFile);
dt = 1 / v.FrameRate;
rawFrames = {}; k = 1;
while hasFrame(v) && k <= 100 
    frame = readFrame(v);
    if size(frame,3) == 3, I = rgb2gray(frame); else, I = frame; end
    rawFrames{k} = im2double(I);
    k = k + 1;
end
nFrames = numel(rawFrames);

%% ---------------- 3. Rectification ----------------
I0 = rawFrames{1};
tform_fig = figure('Name', 'ROI Selection'); imshow(I0, []);
title('Click 4 points CLOCKWISE');
[xi, yi] = ginput(4); close(tform_fig);

xr = [0 realWidth realWidth 0];
yr = [0 0         realLength realLength];
tform = fitgeotrans([xi yi], [xr' yr'], 'projective');
nRows = round(realLength * res_px_m);
nCols = round(realWidth  * res_px_m);
R = imref2d([nRows nCols], [0 realWidth], [0 realLength]);

%% ---------------- 4. Preprocessing ----------------
fprintf('   > Background Subtraction and Normalizing...\n');
avgFrame = zeros(R.ImageSize);
for k = 1:min(5, nFrames):nFrames
    avgFrame = avgFrame + imwarp(rawFrames{k}, tform, 'OutputView', R);
end
avgFrame = avgFrame / numel(1:min(5, nFrames):nFrames);

processedFrames = cell(1, nFrames);
for k = 1:nFrames
    I = imwarp(rawFrames{k}, tform, 'OutputView', R);
    I_hp = (I - avgFrame) - imgaussfilt(I - avgFrame, 3); 
    processedFrames{k} = single(I_hp); 
end

%% ---------------- 5. PIV Analysis (Optimized Loop) ----------------
% Removed Animation Figure for Speed
outputVideo = VideoWriter('PIV_Optimized_Result.mp4', 'MPEG-4');
outputVideo.FrameRate = (1/dt) / frameStep; 
open(outputVideo);

[H, W] = size(processedFrames{1});
minWin = winSizes(end);
dx_min = minWin * (1 - overlaps(end));
[xg_base, yg_base] = meshgrid((minWin/2+1):dx_min:(W-minWin/2-1), ...
                              (minWin/2+1):dx_min:(H-minWin/2-1));

all_mean_speeds = [];
fprintf('   > Processing %d frame pairs...\n', floor(nFrames/frameStep));

% Hidden figure for Video Capture ONLY (No real-time rendering)
figHidden = figure('visible', 'off'); 

for idx1 = 1:frameStep:(nFrames - frameStep)
    idx2 = idx1 + frameStep;
    I1 = processedFrames{idx1}; I2 = processedFrames{idx2};
    U_tot = zeros(size(xg_base)); V_tot = zeros(size(yg_base));
    xg = xg_base; yg = yg_base;

    for p = 1:numel(winSizes)
        winSize = winSizes(p); halfWin = winSize / 2;
        searchW = round(winSize * searchExp(p)); halfSearch = searchW / 2;
        if p > 1
            dx = winSize * (1-overlaps(p));
            [xg, yg] = meshgrid((halfWin+1):dx:(W-halfWin-1), (halfWin+1):dx:(H-halfWin-1));
            U_tot = imresize(U_tot, size(xg)); V_tot = imresize(V_tot, size(yg));
        end
        U = NaN(size(xg)); V = NaN(size(yg));
        
        for i = 1:numel(xg)
            xp = round(xg(i) + U_tot(i)); yp = round(yg(i) + V_tot(i));
            r1 = round(yg(i)-halfWin+1):round(yg(i)+halfWin);
            c1 = round(xg(i)-halfWin+1):round(xg(i)+halfWin);
            r2 = round(yp-halfSearch+1):round(yp+halfSearch);
            c2 = round(xp-halfSearch+1):round(xp+halfSearch);
            
            if r1(1)<1 || r1(end)>H || c1(1)<1 || c1(end)>W || r2(1)<1 || r2(end)>H || c2(1)<1 || c2(end)>W
                continue; 
            end
            
            % --- OPTIMIZATION: Zero-Mean Normalization ---
            sub1 = I1(r1,c1); sub1 = sub1 - mean(sub1(:));
            sub2 = I2(r2,c2); sub2 = sub2 - mean(sub2(:));
            
            sz = size(sub1) + size(sub2) - 1;
            C = real(ifft2(fft2(sub1, sz(1), sz(2)) .* conj(fft2(sub2, sz(1), sz(2)))));
            [maxVal, idxM] = max(C(:)); [py, px] = ind2sub(size(C), idxM);
            
            % --- OPTIMIZATION: Peak Quality Check (PSR) ---
            psr = (maxVal - mean(C(:))) / (std(C(:)) + 1e-6);
            if psr < 3.5, continue; end 
            
            if py > 1 && py < sz(1) && px > 1 && px < sz(2)
                cp = log(maxVal+1e-9); 
                cxm = log(C(py,px-1)+1e-9); cxp = log(C(py,px+1)+1e-9);
                cym = log(C(py-1,px)+1e-9); cyp = log(C(py+1,px)+1e-9);
                U(i) = (px + (cxm-cxp)/(2*(cxm-2*cp+cxp))) - (halfSearch + halfWin);
                V(i) = (py + (cym-cyp)/(2*(cym-2*cp+cyp))) - (halfSearch + halfWin);
            end
        end
        U_tot = U_tot + fillmissing(U,'constant',0); V_tot = V_tot + fillmissing(V,'constant',0);
    end
    
    u_ms = U_tot / res_px_m / (dt * (idx2 - idx1));
    v_ms = V_tot / res_px_m / (dt * (idx2 - idx1));
    speed = sqrt(u_ms.^2 + v_ms.^2);
    valid = (v_ms > 0) & (speed < maxSpeed_mps) & (speed > 0.05);

    if any(valid(:))
        all_mean_speeds(end+1) = mean(speed(valid), 'omitnan');
        % Burn Frame to Video (Without visible plot update)
        u_p = u_ms; v_p = v_ms; u_p(~valid)=NaN; v_p(~valid)=NaN;
        imshow(I1, R, []); hold on;
        quiver(xg/res_px_m, yg/res_px_m, u_p, v_p, 2.5, 'r');
        writeVideo(outputVideo, getframe(figHidden));
        hold off;
    end
end
close(outputVideo);

%% ---------------- 6. Final Results ----------------
avg_V_surface = mean(all_mean_speeds);
Q = avg_V_surface * surfaceToMeanFactor * (realWidth * meanDepth);

fprintf('\n--- FINAL DISCHARGE RESULTS ---\n');
fprintf('Calculated Q: %.3f m^3/s\n', Q);
fprintf('Mean Surface Velocity: %.3f m/s\n', avg_V_surface);

% Single final plot for review
figure('Name', 'Final Flow Map');
imshow(processedFrames{end}, R, []); hold on;
quiver(xg/res_px_m, yg/res_px_m, u_ms, v_ms, 2, 'r');
title('Integrated Surface Velocity Field');