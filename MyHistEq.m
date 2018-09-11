clear
I=imread('126.png'); % PNG has 3 channels
K = rgb2gray(I); % Converts RGB channels into greysclae
[height,width]=size(K); % get the size of the image

% Get the original histogram
% [counts1, x] = imhist(K,256);
% counts2 = counts1/height/width;
% stem(x, counts2); % Plot discrete sequence data

% Get the number of each grey level
NumPixel = zeros(1,256);
for i = 1:height
	for j = 1: width
    % The grey level will increase 1 each time
	% the index of NumPixel starts from 1, but the raneg of pixel is 0~255, so use NumPixel(K(i,j) + 1)
	NumPixel(K(i,j) + 1) = NumPixel(K(i,j) + 1) + 1;
	end
end

% Then transfer the number of grey level into frequency
ProbPixel = zeros(1,256);
for i = 1:256
	ProbPixel(i) = NumPixel(i) / (height * width * 1.0);
end

% Use cumsum to get culmulative distribution function?map the freuqncy(0.0~1.0)to 0~255 intergers?
CumuPixel = cumsum(ProbPixel);
CumuPixel = uint8(255 .* CumuPixel + 0.5);

for i = 1:height
	for j = 1:width
		K(i,j) = CumuPixel(K(i,j));
	end
end


imshow(Img);
[counts1, x] = imhist(Img,256);
counts2 = counts1/height/width;
stem(x, counts2);