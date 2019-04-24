% Gabor Feature Normalization for zero-mean
function a = normalized_gabor(filtered_img,gaborResult,u,v)
% d1=32;d2=32;
[n,m] = size(filtered_img);
% s = (n*m)/(d1*d2); %s=256
% L = s*u*v; % 12 filters * 1 number/filtered_images 
% featureVector = zeros(L,1); 
% c = 0;
% for i = 1:u
%     for j = 1:v
%         
%         c = c+1;
%         gaborAbs = abs(gaborResult{i,j}); % Magnitudes of Gabor filters
%         gaborAbs = downsample(gaborAbs,d1);
%         gaborAbs = downsample(gaborAbs.',d2);
%         gaborAbs = reshape(gaborAbs.',[],1);
%         
%         % Normalized to zero mean and unit variance. (if not applicable, please comment this line)
%         gaborAbs = (gaborAbs-mean(gaborAbs))/std(gaborAbs,1);
%         
%         featureVector(((c-1)*s+1):(c*s)) = gaborAbs;
%         
%     end
% end

% After get the combined vector, we need to get one value from each gabor
% function
a = zeros(1,u*v);% 1*12 vector
c=1;
for i = 1:u
    for j = 1:v
        each_filtered_img_response = gaborResult{i,j};
        temp = abs( tanh(0.5*each_filtered_img_response) );      
        energy = sum(temp(:))/(n^2);
        a(:,c)= energy;
        c = c+1;
    end
end


end
