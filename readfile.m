%% Load ground Truth
data=load('D:\AMME4111\#Code\#Partitioned\score.txt'); 
data_img_name=data(:,1);
data_First_observer=data(:,2); 
data_Second_observer=data(:,3);
% change 5 & 10 --> 1; 15 --> 2; 20 & 25 --> 3 as three class labels
data_First_observer(data_First_observer==5)=1;
data_First_observer(data_First_observer==10)=1;
data_First_observer(data_First_observer==15)=2;
data_First_observer(data_First_observer==20)=3;
data_First_observer(data_First_observer==25)=3;

data_Second_observer(data_Second_observer==5)=1;
data_Second_observer(data_Second_observer==10)=1;
data_Second_observer(data_Second_observer==15)=2;
data_Second_observer(data_Second_observer==20)=3;
data_Second_observer(data_Second_observer==25)=3;
% % 
% % name = linspace(1,139,139);
% % name = name.';
% % 
% % new_data = cat(2,name,data_First_observer);
% % dlmwrite('dataexport.txt',new_data,'delimiter','\t','newline','pc')


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

% annother way to store images
imdsTrain = imageDatastore(fullfile(imgPath),... 
    'FileExtensions','.png',... 
    'LabelSource','foldernames');
imdsTrain.Labels = data_First_observer; % the imageDatastore object cannot store the images



%% Extract individual vectors
tic
Tamura_descriptor = [];
for i = 1:LengthFiles               
    TamVec = get_tamura(store(:,:,i));
    Tamura_descriptor = [Tamura_descriptor;TamVec];
end
toc % in seconds

tic
Direction_descriptor = [];
for i = 1:LengthFiles               
    DirVec = directionality(store(:,:,i));
    Direction_descriptor = [Direction_descriptor;DirVec];
end
toc % in seconds

% LBP histogram were computed globally
% No Need datas_normal = mapminmax(datas), since it is L1 normed

tic
Lbp_descriptor = [];
for i = 1:LengthFiles  
    %J = adapthisteq(store(:,:,i),'ClipLimit',0.25);
    LbpVec = extractLBPFeatures(store(:,:,i),'Radius',1,'NumNeighbors',8,'Upright',false,'CellSize',[16 16]);
    
    %LbpVec = get_LBP();
    Lbp_descriptor = [Lbp_descriptor;LbpVec];
end 
toc 
% pca(X):n samples£¬m dimensions; score has n samples and p dimensions
[coeff,score,latent] = pca(Lbp_descriptor);

% GLCM features in four directions were computed globally
% NEED datas_normal = mapminmax(datas)
tic
GLCM_descriptor = [];
for i = 1:LengthFiles               
    GlcmVec = get_GLCM(store(:,:,i));
    GLCM_descriptor = [GLCM_descriptor;GlcmVec];
end
toc % in seconds

% Mean Gabor filter responses for each region together formed a 48 dimension feature
% NEED datas_normal = mapminmax(datas)
tic
Gabor_descriptor = [];
for i = 1:LengthFiles               
    GaborVec = get_Gabor(store(:,:,i));
    Gabor_descriptor = [Gabor_descriptor;GaborVec];
end
toc % in seconds


%% Another 1v1 SVM

true_labels = data_First_observer;
datas = score;
datas_normal = mapminmax(datas); % mapping row minimum and maximum values to [-1 1]
% Specify t as a binary learner, or one in a set of binary learners
% t is an SVM template. Most of the template object properties are empty. 
% When training the ECOC classifier, the software sets the applicable properties to their default values.
% Linear kernel, default for two-class learning
t = templateSVM();
% Mdl is a ClassificationECOC classifier. You can access its properties using dot notation.
Mdl = fitcecoc(datas,true_labels,'Learners',t);

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





% %% one-versus-one SVM  
% 
% % datas is the input, labels are corresponding labels
% labels = data_First_observer;
% datas = GLCM_descriptor;
% 
% % Normalize data
% datas_normal = mapminmax(datas); % mapping row minimum and maximum values to [-1 1]
%  
% 
% % Cross Validation
% k = 10;% divide data into k folds
% sum_accuracy_svm = 0;
% [m,n] = size(datas_normal);
%  
% %indices: n*1 matrix(vertical vector), it shows each traning sample belongs to which k-fold
%  
% 
% indices = crossvalind('Kfold',m,k);
% 
% for i = 1:k
%     test_indic = (indices == i);
%     train_indic = ~test_indic;
%     train_datas = datas_normal(train_indic,:);% find traning data and labels
%     train_labels = labels(train_indic,:);
%     test_datas = datas_normal(test_indic,:);% find testing data and labels
%     test_labels = labels(test_indic,:);
%     
%     % start SVM training,'fitcsvm' for binary b classification,'fitcecoc' for multiclass
%     SVM_classifer = fitcecoc(train_datas,train_labels);% Training
%     predict_label  = predict(SVM_classifer, test_datas);% Testing
%     accuracy_svm = length(find(predict_label == test_labels))/length(test_labels); % Accuracy rate
%     sum_accuracy_svm = sum_accuracy_svm + accuracy_svm;
% end
% 
% % get average accuracy
% mean_accuracy_svm = sum_accuracy_svm / k;
% disp('Average Accuracy of k-fold cross validation£º');   
% disp( mean_accuracy_svm);