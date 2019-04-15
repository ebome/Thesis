%% histogram equalization
I=imread('D:\AMME4111\#Code\#Partitioned\#3_clipped.png'); % PNG has 3 channels
K = rgb2gray(I); % Converts RGB channels into greysclae

I_new = mat2gray(K,[0 255]); % Normalize image into [0,1]
J = adapthisteq(I_new,'ClipLimit',0.25);

figure; 
subplot(2,2,1),imshow(I_new);title('original image');
subplot(2,2,2),imshow(J);title('histogram equalized image');
subplot(2,2,3),imhist(I_new,64);title('origial histo');
subplot(2,2,4),imhist(J,64);title('equalized histo');

%% downsmapling J to 512*512
DownSampled = imresize(J,[512 512]); % DownSampled is normalized image in [0 1]

subplot(1,2,1),imshow(J);title('original image');
subplot(1,2,2),imshow(DownSampled);title('downsampled image');

%% band-pass spatial filter: attenuate bony confounder effects

% sigma is the radius r, the larger the radius the larger the more unclear the image
% Size of the Gaussian filter must be odd number, we set (11,11)
% B = imgaussfilt(A,sigma)
% Size of the Gaussian filter, specified as a scalar or 2-element vector of positive, odd integers. If you specify a scalar, then imgaussfilt uses a square filter.
blur_HIGH = imgaussfilt(DownSampled,10,'FilterSize',11); %  Gaussian blur:img_undist=G_h(L)

nominator = DownSampled - blur_HIGH; %  L-G_h(L)
var_1 = abs(DownSampled - blur_HIGH) ; %  |L-G_h(L)|
blur_HIGH_denominater = imgaussfilt(var_1,10,'FilterSize',11);
% blur_HIGH,blur_LOW,var_0,var_1, blur_HIGH_denominater are all matrices with float type

% Prevent 0/0 when do bit-wise division, so if there is 0 in var_0,we will
% keep this value in var_0 and not change it
var_3 = nominator./blur_HIGH_denominater;
var_3(isinf(var_3))=0; 

HIGH_PASS_filter = 0.25*var_3+0.5; % element-wise summation and multiplication
% ----------------------------------------------------------------------
% low sigma = 5
BAND_PASS_filter = imgaussfilt(HIGH_PASS_filter,5,'FilterSize',5);

% Up to now, BAND_PASS_filter has value range in [0,1], we can change it to
% [0,255] using im2uint8 or imadjust. If opposite, use mat2gray
Filtered_image = im2uint8(BAND_PASS_filter);
% Matlab will mark the empty cells to 64, but I check it in Python, the
% same empty cell will be 128. So this will cause the difference of Fdir

subplot(121);imshow(HIGH_PASS_filter);title("High pass component");
subplot(122);imshow(Filtered_image);title("Band pass component");

%% texture analysis
%% 1. Directionality: image gradients along x and y directions
% tamura= [ Fcrs, Fcon,Fdir,Flin,Frgh]
tamura = Tamura(Filtered_image);

%% 2. LBP
% Rotation invariance is not very relevant, so set no rotation invariance
% Uniformity is not very relevant as well, so ignore the uniformity
% features = extractLBPFeatures(I) returns extracted uniform local 

% Extract unnormalized LBP features so that we can apply a custom normalization.
% By default, the normalization method will be L2, but here we choose L1,
% so set 'Normalization' = 'None'
% the histogram has 58 separate bins for uniform patterns, 1 bin for all
% other non-uniform patterns. Hence total 59 bins
LBP_features=extractLBPFeatures(Filtered_image,'Radius',1,'NumNeighbors',8,'Upright',true,'Normalization','None');

% Reshape the LBP features into a number of neighbors -by- number of cells array to access histograms for each individual cell.
numNeighbors = 8;
numBins = numNeighbors*(numNeighbors-1)+3;
lbpCellHists = reshape(LBP_features,numBins,[]);

% Normalize each LBP cell histogram using L1 norm.
lbpCellHists = bsxfun(@rdivide,lbpCellHists,sum(lbpCellHists));
% Reshape the LBP features vector back to 1-by- N feature vector.
LBP_features = reshape(lbpCellHists,1,[]); % now LBP_features are L1 normalized

% so we have 59(the bin number of features/uniform patterns) *64(number of bins) = 3776
% To reduce the vector size, we can use PCA. Reduce it to 1*8 or other
% vectors
%% 3. GLCM 
% Grey-level =256 >  16?we could reduce the level to 16 for reducing computation time
% NumLevels = Number of gray levels, specified as an integer.
glcm_4direction = graycomatrix(Filtered_image,'NumLevels',16,'Offset',[0 1; -1 1; -1 0; -1 -1],'Symmetric',true);
% If your glcm is computed with 'Symmetric' flag you can set the flag 'pairs' to 0
% the function will normailze GLCM
glcm_features = GLCM_Features(glcm_4direction,0);

% Four second-order statistical features for 4 directions
glcm_4dir_energy = glcm_features.energ;
glcm_4dir_contrast = glcm_features.contr;
glcm_4dir_homogenity = glcm_features.homom;
glcm_4dir_entropy = glcm_features.entro;

% Concatenate the vectors
GLCM_descriptor = cat(2, glcm_4dir_energy, glcm_4dir_contrast,glcm_4dir_homogenity,glcm_4dir_entropy);
%% 4. Gabor Response
% 3 scales, 4 orientations, 39*39 Gaussian Kernels
u = 3; v = 4;
gaborArray = gaborFilterBank(u,v,39,39);

% figure('NumberTitle','Off','Name','Magnitudes of Gabor filters');
% for i = 1:u
%     for j = 1:v        
%         subplot(u,v,(i-1)*v+j);        
%         imshow(abs(gaborArray{i,j}),[]);
%     end
% end
% 
% % Show real parts of Gabor filters:
% figure('NumberTitle','Off','Name','Real parts of Gabor filters');
% for i = 1:u
%     for j = 1:v        
%         subplot(u,v,(i-1)*v+j);        
%         imshow(real(gaborArray{i,j}),[]);
%     end
% end


% In Lee's paper: Gabor is applied on NORMAILZED image, so don't use filter_image
gaborResult = gaborFeatures(DownSampled,gaborArray);

% figure('NumberTitle','Off','Name','Real parts of Gabor filters');
% for i = 1:u
%     for j = 1:v        
%         subplot(u,v,(i-1)*v+j)    
%         imshow(real(gaborResult{i,j}),[]);
%     end
% end
% 
% 
% % Show magnitudes of Gabor-filtered images
% figure('NumberTitle','Off','Name','Magnitudes of Gabor filters');
% for i = 1:u
%     for j = 1:v        
%         subplot(u,v,(i-1)*v+j)    
%         imshow(abs(gaborResult{i,j}),[]);
%     end
% end

% ----------------------------------------------------------------------
% Gabor Feature extraction for each filtered image
gabor_feature_vector = normalized_gabor(DownSampled,gaborResult,u,v); 
% gabor_feature_vector is the average absolute deviation from zero-mean


%% Local Fuzzy Patterns

% http://www.photon.ac.cn/article/2013/1004-4213-42-11-1375.html
