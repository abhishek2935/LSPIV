clc; clear; close all;

%% ---------------- User parameters ----------------
videoFile = 'ngwerere_20191103.mp4';

realWidth  = 20;      % meters (x-direction, across frame)
realLength = 10;      % meters (y-direction, along flow)
meanDepth  = 2.5;     % meters
surfaceToMeanFactor = 0.85;

winSize        = 64;  % interrogation window size (pixels)
overlap        = 0.5; % 0–1
searchExpansion = 2;  % search window = searchExpansion * winSize

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

if nFrames < 2
    error('Need at least two frames for PIV.');
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
% Estimate pixel density from original width
px_per_meter = W0 / realWidth;   % simple first guess

imgWidth_m  = realWidth;
imgHeight_m = realLength;

nRows = round(imgHeight_m * px_per_meter);
nCols = round(imgWidth_m  * px_per_meter);

% World limits in meters
xWorld = [0 imgWidth_m];
yWorld = [0 imgHeight_m];

R = imref2d([nRows nCols], xWorld, yWorld);

fprintf('Rectified image size: %d x %d pixels (%.2f px/m)\n', ...
        nRows, nCols, px_per_meter);

%% ---------------- Warp all frames ----------------
for k = 1:nFrames
    frames{k} = imwarp(frames{k}, tform, 'OutputView', R);
end

%% ---------------- Preprocess frames ----------------
for k = 1:nFrames
    I = frames{k};
    I = adapthisteq(I, 'ClipLimit', 0.02);
    I = medfilt2(I, [3 3]);
    frames{k} = I;
end

%% ---------------- Choose frame pair for PIV ----------------
% You can increase frame separation (e.g., 1 and 5) if motion is very small.
idx1 = 1;
idx2 = 3;   % try 2,3,4... and see which gives best displacement

I1 = frames{idx1};
I2 = frames{idx2};

[H, W] = size(I1);

%% ---------------- Build interrogation grid ----------------
halfWin    = winSize / 2;
halfSearch = searchExpansion * winSize / 2;

dx = winSize * (1 - overlap);

% Start and end indices so that full search window is inside image
xStart = halfSearch + 1;
xEnd   = W - halfSearch - 1;
yStart = halfSearch + 1;
yEnd   = H - halfSearch - 1;

[xg, yg] = meshgrid(xStart:dx:xEnd, yStart:dx:yEnd);

U = NaN(size(xg));
V = NaN(size(yg));

fprintf('PIV grid: %d x %d = %d windows\n', ...
        size(xg,2), size(xg,1), numel(xg));

%% ---------------- Cross-correlation PIV ----------------
searchH = 2 * halfSearch;
searchW = 2 * halfSearch;

for i = 1:numel(xg)
    x = round(xg(i));
    y = round(yg(i));

    % Extract interrogation window in frame 1
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

    % FFT-based cross-correlation with padding to search window size
    C = real(ifft2( fft2(W1, searchH, searchW) .* conj(fft2(W2, searchH, searchW)) ));

    [~, idx] = max(C(:));
    [py, px] = ind2sub(size(C), idx);

    % zero displacement is at center (halfSearch + 0.5, halfSearch + 0.5)
    du = px - (halfSearch + 0.5);
    dv = py - (halfSearch + 0.5);

    U(i) = du;
    V(i) = dv;
end

%% ---------------- Convert to physical velocities ----------------
u = U / px_per_meter / (dt * (idx2 - idx1));   % m/s
v = V / px_per_meter / (dt * (idx2 - idx1));
speed = sqrt(u.^2 + v.^2);

nValid = sum(~isnan(speed(:)));
fprintf('Valid vectors: %d / %d (%.1f%%)\n', ...
        nValid, numel(speed), 100*nValid/numel(speed));

%% ---------------- Plot velocity field ----------------
figure;
imshow(I1, R, []);  % show rectified image in meters
hold on;

% Quiver positions in meters (xWorld,yWorld)
xq = xg / px_per_meter;
yq = yg / px_per_meter;
quiver(xq, yq, u, v, 0.8, 'r', 'LineWidth', 1);

axis ij equal;
xlabel('Width (m)');
ylabel('Length (m)');

mean_speed = mean(speed(~isnan(speed(:))));
title(sprintf('River Surface Velocity (%d vectors, mean speed = %.2f m/s)', ...
              nValid, mean_speed));
colorbar;

%% ---------------- Discharge estimate ----------------
A = realWidth * meanDepth;               % m^2
Q = mean_speed * surfaceToMeanFactor * A;

fprintf('\nRiver Discharge Estimate (using this frame pair):\n');
fprintf('Surface mean velocity: %.3f m/s\n', mean_speed);
fprintf('Cross-section area:    %.1f m^2\n', A);
fprintf('Surface-to-mean factor: %.2f\n', surfaceToMeanFactor);
fprintf('Estimated Q = %.2f m^3/s\n', Q);
