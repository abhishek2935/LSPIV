clc; clear; close all;

%% ---------------- User parameters ----------------
videoFile = 'ngwerere_20191103.mp4';
E_raw = [642735.8076; 642732.6705; 642732.7864; 642737.5823];
N_raw = [8304292.1190; 8304296.8580; 8304298.4250; 8304295.593];

%% ---------------- INTERACTIVE ROI SELECTION ----------------
v = VideoReader(videoFile);
frame1 = readFrame(v, 1);
if size(frame1,3) == 3, I0 = rgb2gray(frame1); else I0 = frame1; end

figure(1); imshow(I0,[]); hold on; title('CLICK 4 RIVER ROI CORNERS (clockwise)');
[xi, yi] = ginput(4);
plot(xi,yi,'ro','MarkerSize',15,'LineWidth',4);
legend('Your ROI'); pause(2); close(1);

% Center UTM coordinates (meters)
E = E_raw - mean(E_raw); N = N_raw - mean(N_raw);
tform = fitgeotrans([xi yi], [E N], 'projective');

fprintf('ROI span: %.0f x %.0f px\n', range(xi), range(yi));

%% ---------------- FIXED: Proper UTM grid ----------------
dx_world = 0.15;  % 0.15m/px
Emin = min(E_raw)-0.3; Emax = max(E_raw)+0.3;
Nmin = min(N_raw)-0.3; Nmax = max(N_raw)+0.3;

nCols = round((Emax-Emin)/dx_world);  % ~35px
nRows = round((Nmax-Nmin)/dx_world);  % ~45px

% FIXED: Correct imref2d syntax
R = imref2d([nRows nCols], [Emin Emax], [Nmin Nmax]);
fprintf('UTM grid: %dx%d px (%.2f m/px)\n', nRows, nCols, dx_world);

%% ---------------- Read & Warp frames ----------------
dt = 1/v.FrameRate; nFrames = v.NumFrames;
frames = cell(1, nFrames);

fprintf('Warping %d frames...\n', nFrames);
for k = 1:nFrames
    frame = readFrame(v, k);
    if size(frame,3) == 3, I = rgb2gray(frame); else I = frame; end
    I = im2double(I);
    
    % FIXED: No OutputFormat parameter needed
    I_warped = imwarp(I, tform, 'OutputView', R);
    frames{k} = imgaussfilt(I_warped, 0.8);  % Gentle Gaussian only
    
    if mod(k,20)==0, fprintf('%d ', k); end
end
fprintf('\nWarping complete!\n');

%% ---------------- Robust PIV grid ----------------
[H,W] = size(frames{1});
dx_grid = 5;
[xg_base, yg_base] = meshgrid(8:dx_grid:W-8, 8:dx_grid:H-8);
nWindows = numel(xg_base);
fprintf('PIV grid: %dx%d = %d vectors\n', size(xg_base,2), size(xg_base,1), nWindows);

%% ---------------- FIXED PIV - No flat patches ----------------
nPairs = min(15, nFrames-2);
u_all = NaN(nWindows, nPairs);
v_all = NaN(nWindows, nPairs);

for pair = 1:nPairs
    fprintf('Pair %d/%d\r', pair, nPairs);
    I1 = frames{pair}; I2 = frames{pair+2};
    
    halfWin = 6;
    for i = 1:nWindows
        x = xg_base(i); y = yg_base(i);
        x1 = max(1,round(x-halfWin)):min(W,round(x+halfWin));
        y1 = max(1,round(y-halfWin)):min(H,round(y+halfWin));
        
        if numel(x1)<10 || numel(y1)<10, continue; end
        
        p1 = I1(y1,x1); p2 = I2(y1,x1);
        
        % Skip uniform patches
        if std(p1(:)) < 0.015 || std(p2(:)) < 0.015, continue; end
        
        % Safe cross-correlation
        try
            c = normxcorr2(p2-mean(p2(:)), p1-mean(p1(:)));
            [~, maxIdx] = max(c(:));
            [dy_sub, dx_sub] = ind2sub(size(c), maxIdx);
            
            u_all(i,pair) = (dx_sub - size(c,2)/2) * dx_world / dt;
            v_all(i,pair) = (dy_sub - size(c,1)/2) * dx_world / dt;
        catch
            continue;
        end
    end
end
fprintf('\nPIV complete!\n');

%% ---------------- Results ----------------
meanU = mean(u_all, 2, 'omitnan');
meanV = mean(v_all, 2, 'omitnan');
mean_speed = sqrt(meanU.^2 + meanV.^2);
mean_speed(isnan(mean_speed)) = 0;

fprintf('\n=== NGWERERE RIVER RESULTS ===\n');
fprintf('Mean speed: %.3f ± %.3f m/s\n', mean(mean_speed), std(mean_speed));
fprintf('Valid vectors: %d/%d (%.0f%%)\n', sum(mean_speed>0.02), nWindows, ...
    100*sum(mean_speed>0.02)/nWindows);

%% ---------------- Final plots ----------------
figure('Position', [100 100 1600 500]);

subplot(1,4,1);
imshow(frames{1}, []); hold on;
plot(xg_base, yg_base, 'g.', 'MarkerSize', 10);
title('PIV grid on warped frame');

subplot(1,4,2);
quiver(xg_base, yg_base, meanU, meanV, 2, 'LineWidth', 2, 'Color', 'b');
colorbar; caxis([-1.5 1.5]); title('Velocity field'); axis equal;

subplot(1,4,3);
imagesc(mean_speed); colorbar; caxis([0 1.5]);
title(sprintf('Speed map (%.2f m/s)', mean(mean_speed)));
colormap jet; axis equal;

subplot(1,4,4);
histogram(mean_speed(mean_speed>0), 15, 'FaceColor', 'g', 'EdgeColor', 'k');
title('Speed distribution'); xlabel('m/s'); grid on;

sgtitle(sprintf('SUCCESS! Ngwerere River PIV | %.2f m/s | UTM 35S', mean(mean_speed)));
saveas(gcf, 'ngwerere_piv_success.png');

save('ngwerere_piv_success.mat', 'u_all', 'v_all', 'xg_base', 'yg_base', 'mean_speed', 'tform');
fprintf('✓ SAVED: ngwerere_piv_success.png & .mat\n');
