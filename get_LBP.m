function LBP_feature_vector = get_LBP(I_new)
% 2. LBP
% Rotation invariance is not very relevant, so set no rotation invariance
% Uniformity is not very relevant as well, so ignore the uniformity
% features = extractLBPFeatures(I) returns extracted uniform local 

% Extract unnormalized LBP features so that we can apply a custom normalization.
% By default, the normalization method will be L2, but here we choose L1,
% so set 'Normalization' = 'None'
% the histogram has 58 separate bins for uniform patterns, 1 bin for all
% other non-uniform patterns. Hence total 59 bins

J = adapthisteq(I_new,'ClipLimit',0.25);
% downsmapling J to 512*512
DownSampled = imresize(J,[512 512]); % DownSampled is normalized image in [0 1]
four_region_img = mat2cell(DownSampled,[256 256],[256 256]); % this 4 regions img will feed to other descriptors
[a,b]=size(four_region_img);
B=[];
for i=1:a
    for j=1:b
        each_region = cell2mat( four_region_img(i,j) ); % convert the cell to matrix
        block_LBP_vector = extractLBPFeatures(each_region,'Radius',1,'NumNeighbors',8,'Upright',true,'Normalization','None');
        % Reshape the LBP features into a number of neighbors -by- number of cells array to access histograms for each individual cell.
        numNeighbors = 8;
        numBins = numNeighbors*(numNeighbors-1)+3;
        lbpCellHists = reshape(block_LBP_vector,numBins,[]);

        % Normalize each LBP cell histogram using L1 norm.
        lbpCellHists = bsxfun(@rdivide,lbpCellHists,sum(lbpCellHists));
        % Reshape the LBP features vector back to 1-by- N feature vector.
        block_LBP_vector = reshape(lbpCellHists,1,[]); % now LBP_features are L1 normalized

        % so we have 59(the bin number of features/uniform patterns) *64(number of bins) = 3776
        % To reduce the vector size, we can use PCA.
        B = [B,block_LBP_vector];
    end
end

LBP_feature_vector = B;

end % end of get_LBP(I_new)
