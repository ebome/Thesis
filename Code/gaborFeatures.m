function gaborResult = gaborFeatures(img,gaborArray,d1,d2)

% GABORFEATURES extracts the Gabor features of the image.
% It creates a column vector, consisting of the image's Gabor features.
% The feature vectors are normalized to zero mean and unit variance.
%
%
% Inputs:
%       img         :	Matrix of the input image 
%       gaborArray	:	Gabor filters bank created by the function gaborFilterBank
%       d1          :	The factor of downsampling along rows.
%                       if n is the number of rows in img, d1 must be a factor of n.
%       d2          :	The factor of downsampling along columns.
%                       if m is the number of columns in img,d2 must be a factor of m.
%               
% Output:
%       featureVector	:   A column vector with length (m*n*u*v)/(d1*d2). 
%                           This vector is the Gabor feature vector of an 
%                           m by n image. u is the number of scales and
%                           v is the number of orientations in 'gaborArray'.
%
%
% Sample use:
% 
% img = imread('cameraman.tif');
% gaborArray = gaborFilterBank(5,8,39,39);  % Generates the Gabor filter bank
% featureVector = gaborFeatures(img,gaborArray,4,4);   % Extracts Gabor feature vector, 'featureVector', from the image, 'img'.
% 
% 
%   Details can be found in:
%   
%   M. Haghighat, S. Zonouz, M. Abdel-Mottaleb, "Identification Using 
%   Encrypted Biometrics," Computer Analysis of Images and Patterns, 
%   Springer Berlin Heidelberg, pp. 440-448, 2013.
% 
% 
% (C)	Mohammad Haghighat, University of Miami
%       haghighat@ieee.org
%       I WILL APPRECIATE IF YOU CITE OUR PAPER IN YOUR WORK.


if (nargin ~= 4)    % Check correct number of arguments
    error('Use correct number of input arguments!')
end

if size(img,3) == 3	% % Check if the input image is grayscale
    img = rgb2gray(img);
end

img = double(img);


%% Filtering

% Filter input image by each Gabor filter
[u,v] = size(gaborArray);
gaborResult = cell(u,v);
for i = 1:u
    for j = 1:v
        gaborResult{i,j} = conv2(img,gaborArray{i,j},'same'); % 'same' returns the central part of the convolution, which is the same size as img
    end
end