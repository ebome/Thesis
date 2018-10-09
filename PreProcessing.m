%% Histogram equalization & the downsampled to 512*512
clear

image_folder = 'D:\AMME4111\#Code\Fibrosis'; %  Enter name of folder from which you want to upload pictures with full path
new_folder = 'D:\AMME4111\#Code\PreProcessedImage';

filenames = dir(fullfile(image_folder, '*.png'));  % read all images with specified extention
total_images = numel(filenames);    % count total number of photos present in that folder

for n = 1:total_images
    full_name= fullfile(image_folder, filenames(n).name);         % it will specify images names with full path and extension
    our_images = imread(full_name);              % Read images  
    K = rgb2gray(our_images);                    % Converts RGB channels into greysclae

    image_new = mat2gray(K,[0 255])          % Normalize image into [0,1]
    J=histeq(image_new);
    DownSampled = imresize(J,[512 512])

    baseFileName = sprintf('#%d.png', n);
    fullFileName = fullfile(new_folder, baseFileName);
    imwrite(DownSampled, fullFileName);
    
end

%% Segmentation stage 0: manual label on the data
