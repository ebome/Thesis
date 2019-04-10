%% histogram equalization
clear
I=imread('D:\AMME4111\#Code\#Partitioned\#1_clipped.png'); % PNG has 3 channels
K = rgb2gray(I); % Converts RGB channels into greysclae

I_new = mat2gray(K,[0 255]); % Normalize image into [0,1]
J = adapthisteq(I_new,'ClipLimit',0.25);

figure; 
subplot(2,2,1),imshow(I_new);title('original image');
subplot(2,2,2),imshow(J);title('histogram equalized image');
subplot(2,2,3),imhist(I_new,64);title('origial histo');
subplot(2,2,4),imhist(J,64);title('equalized histo');

%% downsmapling J to 512*512
DownSampled = imresize(J,[512 512]);

subplot(1,2,1),imshow(J);title('original image');
subplot(1,2,2),imshow(DownSampled);title('downsampled image');

%% band-pass spatial filter: attenuate bony confounder effects

% sigma is the radius r, the larger the radius the larger the more unclear the image
% Size of the Gaussian filter must be odd number, we set (11,11)
% high sigma = 11; 
blur_HIGH = imgaussfilt(DownSampled,11,'FilterSize',3); %  Gaussian blur:img_undist=G_h(L)

var_0 = DownSampled - blur_HIGH; %  L-G_h(L)
var_1 = abs(DownSampled - blur_HIGH) ; %  |L-G_h(L)|
blur_HIGH_denominater = imgaussfilt(DownSampled,11,'FilterSize',3);
% blur_HIGH,blur_LOW,var_0,var_1, blur_HIGH_denominater are all matrices with float type

% Prevent 0/0 when do bit-wise division, so if there is 0 in var_0,we will
% keep this value in var_0 and not change it
var_3 =  blur_HIGH_denominater./var_0;
var_3(isinf(var_3))=0; 

HIGH_PASS_filter = 0.25*var_3+0.5; % element-wise summation and multiplication
% high sigma = 5
BAND_PASS_filter = imgaussfilt(DownSampled,5,'FilterSize',3);

% Up to now, BAND_PASS_filter has value range in [0,1], we can change it to
% [0,255] using im2uint8 or imadjust. If opposite, use mat2gray
Filtered_image = im2uint8(BAND_PASS_filter);
% Matlab will mark the empty cells to 64, but I check it in Python, the
% same empty cell will be 128. So this will cause the difference of Fdir

subplot(121);imshow(HIGH_PASS_filter);title("High pass component");
subplot(122);imshow(Filtered_image);title("Band pass component");

%% texture analysis
%% 1. Directionality: image gradients along x and y directions
tamura = Tamura(Filtered_image)

%% 2. LBP
% Rotation invariance is not very relevant, so set no rotation invariance
LBP_features=extractLBPFeatures(Filtered_image,'Radius',3,'NumNeighbors',24,'Upright',true);
[m n]=size(Filtered_image);
LBPimg=zeros(m,n);


for i=1:m
    for j=1:n        
        b0=0;  b1=0;  b2=0;  b3=0;  b4=0;  b5=0;  b6=0;  b7=0;

        if(i-1>0 && j-1>0 && i+1<=m && j+1<=n) % This ensure to ignore the pixels on the edge
            if(Filtered_image(i-1,j-1)>Filtered_image(i,j))
                b0=1;
            end

            if(Filtered_image(i-1,j)>Filtered_image(i,j))
                b1=1;
            end            

            if(Filtered_image(i-1,j+1)>Filtered_image(i,j))
                b2=1;
            end           

            if(Filtered_image(i,j+1)>Filtered_image(i,j))
                b3=1;
            end            

            if(Filtered_image(i+1,j+1)>Filtered_image(i,j))
                b4=1;
            end            

            if(Filtered_image(i+1,j)>Filtered_image(i,j))
                b5=1;
            end 

            if(Filtered_image(i+1,j-1)>Filtered_image(i,j))
                b6=1;
            end      

            if(Filtered_image(i,j-1)>Filtered_image(i,j))
                b7=1;
            end        

            if(Filtered_image(i+1,j-1)>Filtered_image(i,j))
                b5=1;
            end              
        b=b0+b1*2^1+b2*2^2+b3*2^3+b4*2^4+b5*2^5+b6*2^6+b7*2^7;
        LBPimg(i,j)=b;
        end
        
    end
end

figure
subplot(1,2,1),imshow(Filtered_image);title('downsampled image');
subplot(1,2,2),imshow(LBPimg);title('LBP image');

%% 3. GLCM 
glcm = graycomatrix(DownSampled,'NumLevels',256);
imshow(glcm);

%% 4. Gabor filter
[mag0, phase0] = imgaborfilt(DownSampled,5,0);
[mag1, phase1] = imgaborfilt(DownSampled,5,45);
[mag2, phase2] = imgaborfilt(DownSampled,5,90);
[mag3, phase3] = imgaborfilt(DownSampled,5,135);
[mag4, phase4] = imgaborfilt(DownSampled,5,180);
