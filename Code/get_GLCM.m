function GLCM_descriptor = get_GLCM(I_new)
    J = adapthisteq(I_new,'ClipLimit',0.25);
    DownSampled = imresize(J,[512 512]); % DownSampled is normalized image in [0 1]

    blur_HIGH = imgaussfilt(DownSampled,10,'FilterSize',11); %  Gaussian blur:img_undist=G_h(L)

    nominator = DownSampled - blur_HIGH; %  L-G_h(L)
    var_1 = abs(DownSampled - blur_HIGH) ; %  |L-G_h(L)|
    blur_HIGH_denominater = imgaussfilt(var_1,10,'FilterSize',11);
    var_3 = nominator./blur_HIGH_denominater;
    var_3(isinf(var_3))=0; 

    HIGH_PASS_filter = 0.25*var_3+0.5; % element-wise summation and multiplication
    % low sigma = 5
    BAND_PASS_filter = imgaussfilt(HIGH_PASS_filter,5,'FilterSize',5);

    % Up to now, BAND_PASS_filter has value range in [0,1], we can change it to
    % [0,255] using im2uint8 or imadjust. If opposite, use mat2gray
    Filtered_image = im2uint8(BAND_PASS_filter);
 
    block_GLCM_4direction_vector = graycomatrix(Filtered_image,'NumLevels',16,'Offset',[0 1; -1 1; -1 0; -1 -1],'Symmetric',true);
        
    % If your glcm is computed with 'Symmetric' flag you can set the flag 'pairs' to 0
    % the function will normailze GLCM
    glcm_features = GLCM_Features(block_GLCM_4direction_vector,0);
        
    % Four second-order statistical features for 4 directions
    glcm_4dir_energy = glcm_features.energ;
    glcm_4dir_contrast = glcm_features.contr;
    glcm_4dir_homogenity = glcm_features.homom;
    glcm_4dir_entropy = glcm_features.entro;
        
    % Concatenate the vectors
    GLCM_descriptor = cat(2, glcm_4dir_energy, glcm_4dir_contrast,glcm_4dir_homogenity,glcm_4dir_entropy);
end
