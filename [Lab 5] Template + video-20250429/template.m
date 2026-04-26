clear all
format long
R=VideoReader ('visiontraffic.avi'); %creates object v to read video data from the file named filename
X_Data=Matrix_video(R,171,200);%Put the 30 frames in a matrix
[n,m]=size(X_Data);
X = double(X_Data); % matricea cu cele 30 de frame-uri din video

p = 0.007;
lambda = 0.0002;

Lk = zeros(n, m);
Sk = zeros(n, m);
Lambda = zeros(n, m);
residual = [];
rank_history = [];
iter = 0;
max_iter = 200;

ST = @(A, t) sign(A) .* max(abs(A) - t, 0);

while rank(Lk) > 1 || iter == 0
    if iter >= max_iter
        break
    end

    Y = X - Sk - (1/p) * Lambda;
    Lk = SVT(Y, 1/p);

    Y = X - Lk - (1/p) * Lambda;
    Sk = ST(Y, lambda/p);

    Lambda = Lambda + p * (Lk + Sk - X);

    res = norm(Lk + Sk - X, 'fro');
    residual = [residual, res];
    rank_history = [rank_history, rank(Lk)];

    iter = iter + 1;
end

% Afișare aleatorie a unui frame
frame_idx = randi(m);
original = reshape(X(:,frame_idx), [360, 640]);
background = reshape(uint8(Lk(:,frame_idx)), [360, 640]);
foreground = reshape(uint8(Sk(:,frame_idx)), [360, 640]);

figure('Name','Original Frame','NumberTitle','off');
imshow(uint8(original));
title('Original Frame');

figure('Name','Background','NumberTitle','off');
imshow(background);
title('Background (Low-rank)');

figure('Name','Foreground','NumberTitle','off');
imshow(foreground);
title('Foreground (Sparse)');

% Plotare reziduu
figure('Name','Reziduu','NumberTitle','off');
semilogy(residual);
xlabel('Iterații');
ylabel('Reziduu ‖L+S−X‖');
title('Convergența ADMM');

% Salvare video output: Original | Background | Foreground
outputVideo = VideoWriter('output.avi');
outputVideo.FrameRate = 10;
open(outputVideo);

for j = 1:m
    Img_org = reshape(X(:,j), 360, 640);
    Img_bg  = reshape(uint8(Lk(:,j)), 360, 640);
    Img_fg  = reshape(uint8(Sk(:,j)), 360, 640);

    combined = [uint8(Img_org), Img_bg, Img_fg];
    writeVideo(outputVideo, combined);
end

close(outputVideo);
disp('Video salvat ca output.avi');
