%% histogram equalization
clear
I=imread('126.png'); % PNG has 3 channels
K = rgb2gray(I); % Converts RGB channels into greysclae

I_new = mat2gray(K,[0 255]) % Normalize image into [0,1]
% J = adapthisteq(I_new,'ClipLimit',0.25)
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
sigma_high = 10;% sigma is the radius r, the larger the radius the larger the more unclear the image
img_undist=imgaussfilt(DownSampled,sigma_high); %  Gaussian blur:img_undist=G_h(L)

% High pass component of the filter
numerator = DownSampled-img_undist; % L-G_h(L)
deter = det(numerator); % |L-G_h(L)|
denumerator = imgaussfilt(deter,sigma_high);
highPass = 1/2+(1/4)*(numerator/denumerator);

sigma_low=0.3;
bandPass = imgaussfilt(highPass,sigma_low);

subplot(121);imshow(highPass);title("High pass component");
subplot(122);imshow(bandPass);title("Band pass component");

%% texture analysis
%% 1. Directionality: image gradients along x and y directions
[Gx2, Gy2] = imgradientxy(DownSampled,'sobel'); % Gx2 = gard(H); Gy2 = grad(V)
subplot(121);imshow(Gx2);title('horizental gradient');subplot(122);imshow(Gy2);title('vertical gradient');
vector_magnitude = (abs(Gx2)+abs(Gy2))/2; % vector_magnitude =|grad(G)|

% There are some NaN in vector_angle since the division
vector_angle = atan(Gy2./Gx2)+pi/2; % Element-wise division; notice that Angle is in radian
vector_angle_degree = rad2deg(vector_angle); 


%% 2. LBP
[m n]=size(DownSampled);
LBPimg=zeros(m,n);


for i=1:m
    for j=1:n        
        b0=0;  b1=0;  b2=0;  b3=0;  b4=0;  b5=0;  b6=0;  b7=0;

        if(i-1>0 && j-1>0 && i+1<=m && j+1<=n) % This ensure to ignore the pixels on the edge
            if(DownSampled(i-1,j-1)>DownSampled(i,j))
                b0=1;
            end

            if(DownSampled(i-1,j)>DownSampled(i,j))
                b1=1;
            end            

            if(DownSampled(i-1,j+1)>DownSampled(i,j))
                b2=1;
            end           

            if(DownSampled(i,j+1)>DownSampled(i,j))
                b3=1;
            end            

            if(DownSampled(i+1,j+1)>DownSampled(i,j))
                b4=1;
            end            

            if(DownSampled(i+1,j)>DownSampled(i,j))
                b5=1;
            end 

            if(DownSampled(i+1,j-1)>DownSampled(i,j))
                b6=1;
            end      

            if(DownSampled(i,j-1)>DownSampled(i,j))
                b7=1;
            end        

            if(DownSampled(i+1,j-1)>DownSampled(i,j))
                b5=1;
            end              
        b=b0+b1*2^1+b2*2^2+b3*2^3+b4*2^4+b5*2^5+b6*2^6+b7*2^7;
        LBPimg(i,j)=b;
        end
        
    end
end

figure
subplot(1,2,1),imshow(DownSampled);
title('downsampled image');
subplot(1,2,2),imshow(LBPimg);
title('LBP image');

%% 3. GLCM 
glcm = graycomatrix(DownSampled,'NumLevels',256);
imshow(glcm);

%% 4. Gabor filter
[mag0, phase0] = imgaborfilt(DownSampled,5,0);
[mag1, phase1] = imgaborfilt(DownSampled,5,45);
[mag2, phase2] = imgaborfilt(DownSampled,5,90);
[mag3, phase3] = imgaborfilt(DownSampled,5,135);
[mag4, phase4] = imgaborfilt(DownSampled,5,180);

