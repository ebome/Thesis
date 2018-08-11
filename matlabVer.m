%% histogram equalization
clear
I=imread('099.png');
J=histeq(I);

% figure;
% subplot(1,2,1),imshow(I);title('original image');
% subplot(1,2,2),imshow(J);title('histogram equalized image');
% figure;
% subplot(2,1,1),imhist(I,64);title('origial histo');
% subplot(2,1,2),imhist(J,64);title('equalized histo');

%% downsmapling J to 512*512
B = imresize(J,[512 512])
% subplot(1,2,1),imshow(J);title('original image');
% subplot(1,2,2),imshow(B);title('downsampled image');
imwrite(B,'result.jpg')

%% segmentation... C++?

%% texture analysis
% apply a band-pass filter on image (attenuate bony confounder effects)
h = fspecial('gaussian', 3,1.5);  % Gaussian kernel
filtered = imfilter(B, h);
figure, imshow(filtered);title('filtered image');
%% texture analysis
% 1. directionality? image gradients along x and y directions
[Gx2, Gy2] = imgradientxy(filtered(:,:,1),'sobel');
figure
imshowpair(uint8(Gx2), uint8(Gy2),'montage')
title('Directional Gradients: x-direction, Gx (left), y-direction, Gy (right), using Sobel method')
% 2. LBP
lbpSmaple = extractLBPFeatures(filtered(:,:,1),'Upright',true,'cellsize',[18,18]);
numTrain = 512;
featuresMat = zeros(numTrain,length(lbpSmaple));
for i = 1:numTrain
    featuresMat(i,:) = extractLBPFeatures(image, 'Upright',false);
end
figure, imshow(lbpSmaple);title('LBP texture');
% 3. GLCM
P=filtered(:,:,1)
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









