function Gabor_feature_vector = get_Gabor(I_new)
J = adapthisteq(I_new,'ClipLimit',0.25);
DownSampled = imresize(J,[512 512]); % DownSampled is normalized image in [0 1]

%4. Gabor Response
% 3 scales, 4 orientations, 39*39 Gaussian Kernels
u = 3; v = 4;
gaborArray = gaborFilterBank(u,v,39,39);

four_region_img = mat2cell(DownSampled,[256 256],[256 256]); % this 4 regions img will feed to other descriptors
[a,b]=size(four_region_img);
B=[];
for i=1:a
    for j=1:b
        each_region = cell2mat( four_region_img(i,j) ); % convert the cell to matrix
        
        % GaborResult are 12 filtered images
        gaborResult = gaborFeatures(each_region,gaborArray);
        
        % Gabor Feature extraction for each filtered image
        feature_vector = normalized_gabor(DownSampled,gaborResult,u,v); 
        % gabor_feature_vector is the average absolute deviation from zero-mean

        B = [B,feature_vector];
    end
end
Gabor_feature_vector = B;
end % end of get_Gabor(I_new)
