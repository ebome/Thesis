function img_undist = GaussianFilter(Img,Sigma)
for x = 1: 3  % Vertical direction
    for y = 1:3  % Horizontal direction
        WeightMatrix(x, y)=exp(-((x-1)^2+(y-1)^2)/(2*Sigma^2))/(2*pi*Sigma^2); % Gaussian function
    end
end
WeightMatrix=WeightMatrix./sum(sum(WeightMatrix)); % Let the 3*3 matrix has a summation equal to 1
[row, col] = size( Img );
for i = 1: row  % Vertical direction
    for j = 1:col  % Horizontal direction 
        if i==1 || j==1 || i==row || j==col
            img_undist(i, j)=Img(i, j);% The rim of image is not processed
        else
            miniMatrix=single(Img(i-1:i+1, j-1:j+1));
            img_undist(i, j)=sum(sum( miniMatrix.*WeightMatrix ));% Gaussian blur:img_undist=G_h(L)
        end
    end
end

subplot(1,2,1),imshow(Img);title('original image');
subplot(1,2,2),imshow(img_undist);title('Gaussian blurred image');

