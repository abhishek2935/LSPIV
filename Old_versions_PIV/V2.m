clc; clear; close all;

%% ---------------- User parameters ----------------
videoFile = 'Main_1.mp4';

realWidth  = 20;      % meters (x-direction, across river)
realLength = 10;      % meters (y-direction, along flow)
meanDepth  = 2.5;     % meters
surfaceToMeanFactor = 0.85;

winSize        = 64;  % interrogation window size (pixels)
overlap        = 0.5; % 0–1
searchExpansion = 2;  % search window = searchExpansion * winSize

frameStep = 2;        % separation between frames in a pair (1 => consecutive)

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
fprintf('Read %d frames at %.2f fps (dt=%.4f s)\n', ...
        nFrames, v.FrameRate, dt);

if nFrames < (1+frameStep)
    error('Not enough frames for the chosen frameStep.');
end

%% ---------------- Select 4 rectification points ----------------
I0 = frames{1};
[H0, W0] = size(I0);

figure; imshow(I0, []);
title('Click 4 points CLOCKWISE around water surface ROI');
[xi, yi] = ginput(4);
close;

% Physical rectangle in meters
xr = [0 realWidth realWidth 0];
yr = [0 0         realLength realLength];

tform = fitgeotrans([xi yi],[xr' yr'],'projective');

%% ---------------- Define output referencing in meters ----------------
px_per_meter = W0 / realWidth;   % estimate pixel density

imgWidth_m  = realWidth;
imgHeight_m = realLength;

nRows = round(imgHeight_m * px_per_meter);
nCols = round(imgWidth_m  * px_per_meter);

xWorld = [0 imgWidth_m];
yWorld = [0 imgHeight_m];

R = imref2d([nRows nCols], xWorld, yWorld);

fprintf('Rectified image size: %d x %d pixels (%.2f px/m)\n', ...
        nRows, nCols, px_per_meter);

%% ---------------- Warp and preprocess all frames ----------------
for k = 1:nFrames
    I = imwarp(frames{k}, tform, 'OutputView', R);
    I = adapthisteq(I, 'ClipLimit', 0.02);
    I = medfilt2(I, [3 3]);
    frames{k} = I;
end

%% ---------------- Build interrogation grid once ----------------
I1 = frames{1};
[H, W] = size(I1);

halfWin    = winSize / 2;
halfSearch = searchExpansion * winSize / 2;

dx = winSize * (1 - overlap);

xStart = halfSearch + 1;
xEnd   = W - halfSearch - 1;
yStart = halfSearch + 1;
yEnd   = H - halfSearch - 1;

[xg, yg] = meshgrid(xStart:dx:xEnd, yStart:dx:yEnd);

nWindows = numel(xg);
fprintf('PIV grid: %d x %d = %d windows\n', ...
        size(xg,2), size(xg,1), nWindows);

searchH = 2 * halfSearch;
searchW = 2 * halfSearch;

%% ---------------- Loop over frame pairs ----------------
all_mean_speeds = [];

for idx1 = 1:frameStep:(nFrames - frameStep)
    idx2 = idx1 + frameStep;

    I1 = frames{idx1};
    I2 = frames{idx2};

    U = NaN(size(xg));
    V = NaN(size(yg));

    for i = 1:nWindows
        x = round(xg(i));
        y = round(yg(i));

        r1 = y-halfWin+1 : y+halfWin;
        c1 = x-halfWin+1 : x+halfWin;

        r2 = y-halfSearch+1 : y+halfSearch;
        c2 = x-halfSearch+1 : x+halfSearch;

        if r1(1) < 1 || r1(end) > H || c1(1) < 1 || c1(end) > W ...
           || r2(1) < 1 || r2(end) > H || c2(1) < 1 || c2(end) > W
            continue;
        end

        W1 = I1(r1, c1);
        W2 = I2(r2, c2);

        C = real(ifft2( fft2(W1, searchH, searchW) .* ...
                        conj(fft2(W2, searchH, searchW)) ));

        [~, idx] = max(C(:));
        [py, px] = ind2sub(size(C), idx);

        du = px - (halfSearch + 0.5);
        dv = py - (halfSearch + 0.5);

        U(i) = du;
        V(i) = dv;
    end

    % velocity for this frame pair
    dt_pair = dt * (idx2 - idx1);
    u = U / px_per_meter / dt_pair;
    v = V / px_per_meter / dt_pair;
    speed = sqrt(u.^2 + v.^2);

    valid = ~isnan(speed(:));
    if any(valid)
        this_mean = mean(speed(valid));
        all_mean_speeds(end+1) = this_mean;
        fprintf('Pair %d-%d: mean surface speed = %.3f m/s (%d valid vectors)\n', ...
                idx1, idx2, this_mean, sum(valid));
    else
        fprintf('Pair %d-%d: no valid vectors\n', idx1, idx2);
    end
end

%% ---------------- Global mean & discharge ----------------
if isempty(all_mean_speeds)
    error('No valid vectors for any frame pair – adjust parameters or frameStep.');
end

global_mean_speed = mean(all_mean_speeds);

A = realWidth * meanDepth;
Q = global_mean_speed * surfaceToMeanFactor * A;

fprintf('\nOverall mean surface speed = %.3f m/s (from %d pairs)\n', ...
        global_mean_speed, numel(all_mean_speeds));
fprintf('Cross-section area:        %.1f m^2\n', A);
fprintf('Surface-to-mean factor:    %.2f\n', surfaceToMeanFactor);
fprintf('Estimated discharge Q = %.2f m^3/s\n', Q);

%% ---------------- Plot velocity field of last pair ----------------
% Re-use last computed I1, u, v, xg, yg
figure;
imshow(I1, R, []);
hold on;
xq = xg / px_per_meter;
yq = yg / px_per_meter;
quiver(xq, yq, u, v, 0.8, 'r', 'LineWidth', 1);
axis ij equal;
xlabel('Width (m)');
ylabel('Length (m)');
title(sprintf('Last pair velocity field (mean speed = %.2f m/s)', ...
              mean(speed(~isnan(speed(:))))));
colorbar;
