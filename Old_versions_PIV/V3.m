clc; clear; close all;

%% ---------------- User parameters ----------------
videoFile = 'ngwerere_20191103.mp4';

realWidth  = 20;      % meters (x-direction, across river)
realLength = 10;      % meters (y-direction, along flow)
meanDepth  = 2.5;     % meters
surfaceToMeanFactor = 0.85;

% multi-pass interrogation parameters
winSizes   = [96 64];   % window size for each pass (pixels)
overlaps   = [0.5 0.5]; % overlap for each pass (0–1)
searchExp  = [2   1.5]; % search-window expansion factors

frameStep     = 2;      % frame separation between pair members
maxSpeed_mps  = 3;      % physical speed limit for filtering (m/s)

%% ---------------- Read video ----------------
v  = VideoReader(videoFile);
dt = 1 / v.FrameRate;

frames = {};
k = 1;
while hasFrame(v)
    frame = readFrame(v);
    if size(frame,3) == 3
        I = rgb2gray(frame);
    else
        I = frame;
    end
    frames{k} = im2double(I);
    k = k + 1;
end
nFrames = numel(frames);
fprintf('Read %d frames at %.2f fps (dt = %.4f s)\n', ...
        nFrames, v.FrameRate, dt);

if nFrames < (1 + frameStep)
    error('Not enough frames for the chosen frameStep.');
end

%% ---------------- Select rectification points ----------------
I0 = frames{1};
[H0, W0] = size(I0);

figure; imshow(I0, []);
title('Click 4 points CLOCKWISE around water surface ROI');
[xi, yi] = ginput(4);
close;

% Physical rectangle in meters
xr = [0 realWidth realWidth 0];
yr = [0 0         realLength realLength];

tform = fitgeotrans([xi yi], [xr' yr'], 'projective');

%% ---------------- Define output reference in meters ----------------
px_per_meter = W0 / realWidth;   % estimate pixel density

imgWidth_m  = realWidth;
imgHeight_m = realLength;

nRows = round(imgHeight_m * px_per_meter);
nCols = round(imgWidth_m  * px_per_meter);

xWorld = [0 imgWidth_m];
yWorld = [0 imgHeight_m];

R = imref2d([nRows nCols], xWorld, yWorld);

fprintf('Rectified image size: %d x %d px (%.2f px/m)\n', ...
        nRows, nCols, px_per_meter);

%% ---------------- Warp and preprocess all frames ----------------
for k = 1:nFrames
    I = imwarp(frames{k}, tform, 'OutputView', R);
    I = adapthisteq(I, 'ClipLimit', 0.02);
    I = medfilt2(I, [3 3]);
    frames{k} = I;
end

%% ---------------- Base grid for smallest window ----------------
minWin     = winSizes(end);
halfWinMin = minWin / 2;

[H, W] = size(frames{1});

dx_min = minWin * (1 - overlaps(end));

xStart = halfWinMin + 1;
xEnd   = W - halfWinMin - 1;
yStart = halfWinMin + 1;
yEnd   = H - halfWinMin - 1;

[xg_base, yg_base] = meshgrid(xStart:dx_min:xEnd, ...
                              yStart:dx_min:yEnd);

nWindows = numel(xg_base);
fprintf('Base PIV grid: %d x %d = %d windows\n', ...
        size(xg_base,2), size(xg_base,1), nWindows);

%% ---------------- Loop over frame pairs with multi-pass PIV ----------------
all_mean_speeds = [];
pairCount       = 0;

for idx1 = 1:frameStep:(nFrames - frameStep)
    idx2 = idx1 + frameStep;
    I1 = frames{idx1};
    I2 = frames{idx2};

    pairCount = pairCount + 1;

    % start from base grid each pair
    xg = xg_base;
    yg = yg_base;

    % initial guess: zero displacement
    U_tot = zeros(size(xg));
    V_tot = zeros(size(yg));

    % ===== multi-pass loop =====
    for p = 1:numel(winSizes)
        winSize    = winSizes(p);
        overlap    = overlaps(p);
        halfWin    = winSize / 2;
        halfSearch = searchExp(p) * winSize / 2;

        dx = winSize * (1 - overlap);

        % refine grid from previous pass (except first pass)
        if p == 1
            % first pass: reuse base grid
        else
            xStart_p = halfWin + 1;
            xEnd_p   = W - halfWin - 1;
            yStart_p = halfWin + 1;
            yEnd_p   = H - halfWin - 1;
            [xg, yg] = meshgrid(xStart_p:dx:xEnd_p, ...
                                yStart_p:dx:yEnd_p);

            % interpolate accumulated displacement to new grid
            U_tot = imresize(U_tot, size(xg), 'bilinear');
            V_tot = imresize(V_tot, size(yg), 'bilinear');
        end

        U = NaN(size(xg));
        V = NaN(size(yg));

        searchH = 2 * halfSearch;
        searchW = 2 * halfSearch;

        for i = 1:numel(xg)
            % previous guess position in frame 2
            x0 = xg(i);
            y0 = yg(i);

            if p == 1
                xp = x0;
                yp = y0;
            else
                xp = x0 + U_tot(i);
                yp = y0 + V_tot(i);
            end

            % clamp to valid range to avoid negative / out-of-bounds indices
            xp = max(halfSearch+1, min(W-halfSearch-1, round(xp)));
            yp = max(halfSearch+1, min(H-halfSearch-1, round(yp)));

            r1 = (yp-halfWin+1)   : (yp+halfWin);
            c1 = (xp-halfWin+1)   : (xp+halfWin);
            r2 = (yp-halfSearch+1): (yp+halfSearch);
            c2 = (xp-halfSearch+1): (xp+halfSearch);

            if r1(1) < 1 || r1(end) > H || c1(1) < 1 || c1(end) > W ...
               || r2(1) < 1 || r2(end) > H || c2(1) < 1 || c2(end) > W
                continue;
            end

            W1 = I1(r1, c1);
            W2 = I2(r2, c2);

            C = real(ifft2( fft2(W1, searchH, searchW) .* ...
                            conj(fft2(W2, searchH, searchW)) ));

            [~, idxMax] = max(C(:));
            [py, px] = ind2sub(size(C), idxMax);

            du = px - (halfSearch + 0.5);
            dv = py - (halfSearch + 0.5);

            U(i) = du;
            V(i) = dv;
        end

        % accumulate displacement from this pass
        U_tot = U_tot + U;
        V_tot = V_tot + V;
    end % multi-pass

    % ===== velocity for this frame pair (NOTE: correct dt_pair) =====
    dt_pair = dt * (idx2 - idx1);           % crucial for correct speed
    u = U_tot / px_per_meter / dt_pair;     % m/s
    v = V_tot / px_per_meter / dt_pair;
    speed = sqrt(u.^2 + v.^2);

    % physical validation: filter unrealistic speeds
    valid = ~isnan(speed) & (speed < maxSpeed_mps);

    if any(valid(:))
        this_mean = mean(speed(valid));
        all_mean_speeds(end+1) = this_mean;
        fprintf('Pair %d-%d: mean surface speed = %.3f m/s (%d valid vectors)\n', ...
                idx1, idx2, this_mean, nnz(valid));
    else
        fprintf('Pair %d-%d: no valid vectors\n', idx1, idx2);
    end
end

%% ---------------- Global mean & discharge ----------------
if isempty(all_mean_speeds)
    error('No valid vectors for any frame pair – adjust parameters or frameStep.');
end

% Correct global mean: average of per-pair means
global_mean_speed = mean(all_mean_speeds);

A = realWidth * meanDepth;
Q = global_mean_speed * surfaceToMeanFactor * A;

fprintf('\nOverall mean surface speed = %.3f m/s (from %d pairs)\n', ...
        global_mean_speed, numel(all_mean_speeds));
fprintf('Cross-section area:        %.1f m^2\n', A);
fprintf('Surface-to-mean factor:    %.2f\n', surfaceToMeanFactor);
fprintf('Estimated discharge Q = %.2f m^3/s\n', Q);

%% ---------------- Plot velocity field of last pair ----------------
figure;
imshow(I1, R, []);
hold on;
xq = xg / px_per_meter;
yq = yg / px_per_meter;
quiver(xq, yq, u, v, 0.8, 'r', 'LineWidth', 1);
axis ij equal;
xlabel('Width (m)');
ylabel('Length (m)');
title(sprintf('Last-pair velocity field (mean speed = %.2f m/s)', ...
              mean(speed(~isnan(speed(:))))));
colorbar;
