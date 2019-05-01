%% Load VLFeat.org Toolbox fro fast dense SIFT
% REF: http://www.vlfeat.org/install-matlab.html
run('D:\AMME4111\VLFEATROOT\vlfeat-0.9.9\toolbox\vl_setup');

%% Load ground Truth
dataPath = 'D:\AMME4111\#Code\#Partitioned\score.txt';
[categorical_label_First,categorical_label_Second] = get_ground_truth(dataPath);
   
%% Store all the image data into STORE matrix
imgPath = 'D:\AMME4111\#Code\#Partitioned\';
Files = dir(strcat(imgPath,'*.png'));
img_name = struct2cell(Files);
img_name = img_name(1,:);
LengthFiles = length(Files);

store = zeros([512,512,LengthFiles]);
for i = 1:LengthFiles               
    img = imread( imgPath+string(img_name(i))); 
    K = rgb2gray(img); % Converts RGB channels into greysclae
    I_new = mat2gray(K,[0 255]); %  Normalize image into [0,1]
    store(:,:,i) = I_new;
end

% After CLAHE, remove the extract noise from them. The image are in [0 1]
histo_equalized = zeros([512,512,LengthFiles]);
for i = 1:LengthFiles               
    G = adapthisteq(store(:,:,i) ,'ClipLimit',0.25); 
    G_orign = store(:,:,i);
    G(G_orign == 0) = 0;     
    histo_equalized(:,:,i) = G;
end

%% For SIFT function
%  The vl_sift command requires a single precision gray scale image. It also expects the range to be normalized in the [0,255] interval 
sift_equalized = zeros([512,512,LengthFiles]);
for i = 1:LengthFiles               
    S=im2uint8(histo_equalized(:,:,i));
    S = single(S);
    sift_equalized(:,:,i) = S;
end

  
%% Gaussian filtered data
gaussian_filtered_images = zeros([512,512,LengthFiles]);
for i = 1:LengthFiles 
    DownSampled = imresize(store(:,:,i),[512 512]); % DownSampled is normalized image in [0 1]
    blur_HIGH = imgaussfilt(DownSampled,10,'FilterSize',11); %  Gaussian blur:img_undist=G_h(L)
    nominator = DownSampled - blur_HIGH; %  L-G_h(L)
    var_1 = abs(DownSampled - blur_HIGH) ; %  |L-G_h(L)|
    blur_HIGH_denominater = imgaussfilt(var_1,10,'FilterSize',11);
    var_3 = nominator./blur_HIGH_denominater;
    var_3(isinf(var_3))=0; 
    HIGH_PASS_filter = 0.25*var_3+0.5; % element-wise summation and multiplication
    BAND_PASS_filter = imgaussfilt(HIGH_PASS_filter,5,'FilterSize',3);
    Filtered_image = im2uint8(BAND_PASS_filter);
    gaussian_filtered_images(:,:,i) = Filtered_image;
end

% i = 14;
% aa_1 = store(:,:,i);
% aa_2 = gaussian_filtered_images(:,:,i);
% figure
% imshowpair(aa_1, aa_2, 'montage')
% title('raw img(left),filtered (right)')



% remove the shadows around filtered images
removed_noise_images = zeros([512,512,LengthFiles]);
for i = 1:LengthFiles 
    G_orign = store(:,:,i);
    G = gaussian_filtered_images(:,:,i); 
    G(G_orign == 0) = 64;     
    removed_noise_images(:,:,i) = G;
    
end


% i = 1;
% aa_1 = store(:,:,i);
% aa_2 = gaussian_filtered_images(:,:,i);
% aa_3 = removed_noise_images(:,:,i);
% figure
% imshowpair(aa_1, aa_3, 'montage')
% title('Manual segmented image(left),Gaussian filted image(right)')
% 
% 
% [Gmag, Gdir] = imgradient(aa_3,'prewitt');
% [Gmag2, Gdir2] = imgradient(aa_32,'prewitt');
% figure
% imshowpair(Gmag, Gdir, 'montage');
% title('Gradient Magnitude, Gmag (left), and Gradient Direction, Gdir (right), using Prewitt method 4')
% figure
% imshowpair(Gmag2, Gdir2, 'montage');
% title('Gradient Magnitude, Gmag (left), and Gradient Direction, Gdir (right), using Prewitt method 14 MILD')

%% annother way to store images
imdsTrain = imageDatastore(fullfile(imgPath),... 
    'FileExtensions','.png',... 
    'LabelSource','foldernames');
imdsTrain.Labels = categorical_label_First; % the imageDatastore object cannot store the images


%% Extract individual vectors

% Total directionality
tic
Direction_descriptor_long = [];
for i = 1:LengthFiles               
    DirVec = get_directionality(removed_noise_images(:,:,i),100);
    Direction_descriptor_long = [Direction_descriptor_long;DirVec];
end
Direction_descriptor_global = [];
for i = 1:LengthFiles               
    DirVec = Fdir(removed_noise_images(:,:,i),100);
    Direction_descriptor_global = [Direction_descriptor_global;DirVec];
end
toc 
Dir_descriptor = cat(2,Direction_descriptor_long,Direction_descriptor_global);


% Tamura feature: each image is input into code globally
tic
Tamura_descriptor = [];
for i = 1:LengthFiles               
    TamVec = get_tamura(removed_noise_images(:,:,i));
    Tamura_descriptor = [Tamura_descriptor;TamVec];
end
toc
tamVector = cat(2,Direction_descriptor_long,Tamura_descriptor);


% LBP histogram were computed globally
% No Need datas_normal = mapminmax(datas), since it is L2 normed
tic
Lbp_descriptor_long = [];
for i = 1:LengthFiles  
    J = histo_equalized(:,:,i);
    LbpVec = extractLBPFeatures(J,'Radius',1,'NumNeighbors',8,'Upright',false,'CellSize',[16 16]);
    
    %LbpVec = get_LBP();
    Lbp_descriptor_long = [Lbp_descriptor_long;LbpVec];
end 
toc 
% pca(X):n samples have m dimensions; score has n samples and p dimensions
[coeff,score,latent] = pca(Lbp_descriptor_long);
Lbp_descriptor = score;


% HOG
tic
hog_descriptor_long = [];
for i = 1:LengthFiles  
    J = histo_equalized(:,:,i); % histo_equalized  removed_noise_images
    HogVec = extractHOGFeatures(J, 'UseSignedOrientation',true,'CellSize',[8 8],'BlockSize',[4 4]);
    
    hog_descriptor_long = [hog_descriptor_long;HogVec];
end 
toc 
% pca(X):n samples have m dimensions; score has n samples and p dimensions
[coeff2,score2,latent2] = pca(hog_descriptor_long);
hog_descriptor = score2;
% [HogVec,visualization] = extractHOGFeatures(removed_noise_images(:,:,3), 'UseSignedOrientation',true,'CellSize',[8 8],'BlockSize',[4 4]);
% plot(visualization)


% GLCM features in four directions were computed globally
% NEED datas_normal = mapminmax(datas)
tic
GLCM_descriptor = [];
for i = 1:LengthFiles               
    GlcmVec = get_GLCM(histo_equalized(:,:,i));
    GLCM_descriptor = [GLCM_descriptor;GlcmVec];
end
toc % in seconds

% Mean Gabor filter responses for each region together formed a 48 dimension feature
% NEED datas_normal = mapminmax(datas)
tic
Gabor_descriptor = [];
for i = 1:LengthFiles               
    GaborVec = get_Gabor(histo_equalized(:,:,i));
    Gabor_descriptor = [Gabor_descriptor;GaborVec];
end
toc % in seconds

%  Haar-like
tic
haar_descriptor_long = [];
for i = 1:LengthFiles  
    J = histo_equalized(:,:,i); % haar is not texture, and hence no need for Gaussian filter
    J_ii = get_integral_image(J); 
    haarVec = extract_haar_features(J_ii);
    haar_descriptor_long = [haar_descriptor_long;haarVec];
end 
toc 
% pca(X):n samples have m dimensions; score has n samples and p dimensions
[coeff3,score3,latent3] = pca(haar_descriptor_long);
haar_descriptor = score3;

%% Dense SIFT implemented by VLFeat.org
tic
sift_descriptor_long = [];
for i = 1:LengthFiles  
    J = sift_equalized(:,:,i); % histo_equalized  removed_noise_images
    SiftVec = 0;
    
    
    
    sift_descriptor_long = [sift_descriptor_long;SiftVec];
end 
toc 
% pca(X):n samples have m dimensions; score has n samples and p dimensions
[coeff,score,latent] = pca(sift_descriptor_long);
hog_descriptor = score;

G = histo_equalized(:,:,3);


%% 1v1 SVM for LBP/Gabor/GLCM/Harr-like/HOG
clc
true_labels = categorical_label_First;
datas = haar_descriptor; % Lbp_descriptor    GLCM_descriptor   Gabor_descriptor  hog_descriptor haar_descriptor
datas_normal = rescale(datas); % mapping row minimum and maximum values to [-1 1]
% Specify t as a binary learner, or one in a set of binary learners
% t is an SVM template. Most of the template object properties are empty. 
% When training the ECOC classifier, the software sets the applicable properties to their default values.
% Linear kernel, default for two-class learning
t = templateSVM('KernelFunction','gaussian'); % Gabor/hog should remove 'KernelFunction','gaussian', the rest deacriptors should keep
% Mdl is a ClassificationECOC classifier. You can access its properties using dot notation.
Mdl = fitcecoc(datas_normal,true_labels,'Learners',t); % Gabor can use normalized data for better performance
% https://www.cnblogs.com/pinard/p/6126077.html


% Cross-validate Mdl using 10-fold cross-validation.
% Cross-validation partition, specified as the comma-separated pair consisting of 
% 'CVPartition' and a cvpartition partition object created by cvpartition. 
k = 10;
cvp = cvpartition(true_labels,'KFold',k,'Stratify',true);
CVMdl = crossval(Mdl,'cvpartition',cvp);

% the classification error is the proportion of observations misclassified by the classifier.
% https://www.mathworks.com/help/stats/classreg.learning.partition.classificationpartitionedkernelecoc.kfoldloss.html#mw_10e79305-0ea9-40a7-9a4a-1f382cdb84d7
loss = kfoldLoss(CVMdl)

%%
predict_label = kfoldPredict(CVMdl);
% compute the confusion matrix
ConfMat = confusionchart(true_labels,predict_label,'RowSummary','total-normalized'...
          ,'ColumnSummary','total-normalized','Title','Confusion matrix of 1v1 SVM');

%% 1v1 SVM for Tamura/Directionality
clc
true_labels = categorical_label_First;
datas = Tamura_descriptor;  % Dir_descriptor,Tamura_descriptor        TamVec are combineation of 4(regions direction)+3 global tamura feature
datas_normal = rescale(datas); % mapping row minimum and maximum values to [-1 1]
t2 = templateSVM('KernelFunction','gaussian');
Md2 = fitcecoc(datas,true_labels,'Learners',t2);

k2 = 10;
cvp2 = cvpartition(true_labels,'KFold',k2,'Stratify',true);
CVMd2 = crossval(Md2,'cvpartition',cvp2);

loss2 = kfoldLoss(CVMd2)


%%
predict_label2 = kfoldPredict(CVMd2);
% compute the confusion matrix
ConfMat2 = confusionchart(true_labels,predict_label2,'RowSummary','total-normalized'...
          ,'ColumnSummary','total-normalized','Title','Confusion matrix of 1v1 SVM');

%% SVM for All features combined
clc
true_labels = categorical_label_First;

Lbp_normal = rescale(Lbp_descriptor);
Gabor_normal = rescale(Gabor_descriptor);
Tam_normal = rescale(tamVector); % mapping row minimum and maximum values to [-1 1]

combined_features = horzcat(Tam_normal,Gabor_normal,Lbp_normal,GLCM_descriptor);
% datas = horzcat(Lbp_normal,tamVector);
% datas = horzcat(tamVector,Gabor_descriptor);
% datas = horzcat(Lbp_normal,GLCM_descriptor);

  
t_combined = templateSVM(); % Gabor should remove 'KernelFuncti on','gaussian'
Md_combined = fitcecoc(datas,true_labels,'Learners',t_combined); % GLCM can try normalized data

k = 10;
cvp_combined = cvpartition(true_labels,'KFold',k,'Stratify',true);
CVMd_combined = crossval(Md_combined,'cvpartition',cvp_combined);

loss_combined = kfoldLoss(CVMd_combined)
