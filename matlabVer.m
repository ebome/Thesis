%% histogram equalization
clear
I=imread('099.png'); % ？？3 channels?
% I1=I(:,:,1);I2=I(:,:,2);I3=I(:,:,3);subplot(1,3,1);imshow(I1);subplot(1,3,2);imshow(I2);subplot(1,3,3);imshow(I3);
I = I(:,:,1);
I_new = mat2gray(I,[0 255]) % normalize image into [0,1]
J=histeq(I_new);

figure;
subplot(2,2,1),imshow(I_new);title('original image');
subplot(2,2,2),imshow(J);title('histogram equalized image');
subplot(2,2,3),imhist(I_new,64);title('origial histo');
subplot(2,2,4),imhist(J,64);title('equalized histo');

%% downsmapling J to 512*512
DownSampled = imresize(J,[512 512])

subplot(1,2,1),imshow(J);title('original image');
subplot(1,2,2),imshow(DownSampled);title('downsampled image');

%% band-pass filter: apply a band-pass filter on image (attenuate bony confounder effects)

% Gaussian kenrnel convoluted with the image: the high-pass spatial size threshold
sigma_high = 10;% σis the radius r，the larger the radius the larger the more unclear the image
img_undist=imgaussfilt(DownSampled,sigma_high); %  Gaussian blur:img_undist=G_h(L)

% High pass component of the filter
numerator = DownSampled-img_undist; % L-G_h(L)
deter = det(numerator); % |L-G_h(L)|
denumerator = imgaussfilt(deter,sigma_high);
highPass = 1/2+(1/4)*(numerator/denumerator);

sigma_low=5;
bandPass = imgaussfilt(highPass,sigma_low);

subplot(121);imshow(highPass);title("High pass component");
subplot(122);imshow(bandPass);title("Band pass component");

%% texture analysis
%% 1. Directionality: image gradients along x and y directions
[Gx2, Gy2] = imgradientxy(img_undist,'sobel'); % Gx2 = ▽H; Gy2 = ▽V
subplot(121);imshow(Gx2);subplot(122);imshow(Gy2);
vector_magnitude = (det(Gx2)+det(Gy2))/2; % vector_magnitude =|▽G|
vector_angle = arctan(Gx2*pinv(Gy2))+pi/2; % B/A = B*inv(A)
% directionality

%% 2. LBP
lbpSmaple = extractLBPFeatures(img_undist);
numTrain = 512;
featuresMat = zeros(numTrain,length(lbpSmaple));
for i = 1:numTrain
    featuresMat(i,:) = extractLBPFeatures(img_undist, 'Upright',false);
end
figure, imshow(featuresMat);title('LBP texture');
%% 3. GLCM
P=img_undist
P_u = unique(P);        % get all grey levels
n = length(P_u);        % number of grey levels
G = zeros(n, n);        % initialize
% four loop, outside two loops for assigning to each location in GLCM
% inner 2 loops go through each pixel and accumulate the # of apperance
for p = 1:n,
    for q = 1:n,

        cnt = 0;           
        for i = 1:r,
            for j = 1:c,
                if  (j+1) <= c && ((P(i, j) == p && P(i, j+1) == q) || P(i, j) == q && P(i, j+1) == p),
                    cnt = cnt + 1;
                end
            end
        end
        G(p, q) = cnt;
    end
end
