%% Directionality
function Fdirection=Fdir(G)  
[r,c]=size(G);  % G must be Gaussian filtered
%-------------------Directionality-------------------
PrewittH = [-1 0 1;-1 0 1;-1 0 1];%for measuring horizontal differences
PrewittV = [1 1 1;0 0 0;-1 -1 -1];%for measuring vertical differences

%Applying PerwittH operator
deltaH=zeros(r,c);
for i=2:r-1
    for j=2:c-1
        deltaH(i,j)=sum(sum(G(i-1:i+1,j-1:j+1).*PrewittH));
    end
end
%Modifying borders
for j=2:c-1
    deltaH(1,j)=G(1,j+1)-G(1,j);
    deltaH(r,j)=G(r,j+1)-G(r,j);  
end
for i=1:r
    deltaH(i,1)=G(i,2)-G(i,1);
    deltaH(i,c)=G(i,c)-G(i,c-1);  
end

%Applying PerwittV operator
deltaV=zeros(r,c);
for i=2:r-1
    for j=2:c-1
        deltaV(i,j)=sum(sum(G(i-1:i+1,j-1:j+1).*PrewittV));
    end
end
%Modifying borders
for j=1:c
    deltaV(1,j)=G(2,j)-G(1,j);
    deltaV(r,j)=G(r,j)-G(r-1,j);  
end
for i=2:r-1
    deltaV(i,1)=G(i+1,1)-G(i,1);
    deltaV(i,c)=G(i+1,c)-G(i,c);  
end

%Magnitude
deltaG=(abs(deltaH)+abs(deltaV))/2;

%Local edge direction (0<=theta<pi)
theta=zeros(r,c);
for i=1:r
    for j=1:c
        if (deltaH(i,j)==0)&&(deltaV(i,j)==0)
            theta(i,j)=0;
        elseif deltaH(i,j)==0
            theta(i,j)=pi;           
        else          
            theta(i,j)=atan(deltaV(i,j)/deltaH(i,j))+pi/2;
        end
    end
end

deltaGt = deltaG(:); % 512*512 image to vertical 262144 vector
theta1=theta(:);

%Set a Threshold value for delta G
n = 16;
HD = zeros(1,n);
%  counting the number of pixels with magnitude greater than a threshold
% The higher the threshold, the lower the Fdirection will be
Threshold=5; % the Threshold here has shown that input image G should be [0 255]
counti=0;
for m=0:(n-1)
    countk=0;
    for k = 1:length(deltaGt)
        if ((deltaGt(k)>=Threshold) && (theta1(k)>=(2*m-1)*pi/(2*n)) && (theta1(k)<(2*m+1)*pi/(2*n)))
            countk=countk+1;
            counti=counti+1;
        end
    end
    HD(m+1) = countk;
end
HDf = HD/counti;

% peakdet function to find peak values
[m p]=peakdet(HDf,0.000005);

Fd=0;
for np = 1:length(m)
    phaiP=m(np)*(pi/n);
    for phi=1:length(HDf)
            Fd=Fd+(phi*(pi/n)-phaiP)^2*HDf(phi);
    end
end
r = (1/n);
Fdirection = 1 - r*np*Fd;


end  % end of directionality function


%-------------------Peakdet-------------------
%% Peakdet Function

% The first argument is the vector to examine, and the second is the peak threshold: 
% We require a difference of at least 0.5 between a peak and its surrounding in order to declare it as a peak. Same goes with valleys.
% 
% The returned vectors "maxtab" and "mintab" contain the peak and valley points, 
% as evident by their plots (note the colors).
% 
% The vector's X-axis values can be passed as a third argument, in which case 
% peakdet() returns these values instead of indices
% 
% Ref: http://billauer.co.il/peakdet.html

function [maxtab, mintab]=peakdet(v, delta, x)
%PEAKDET Detect peaks in a vector
%        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
%        maxima and minima ("peaks") in the vector V.
%        MAXTAB and MINTAB consists of two columns. Column 1
%        contains indices in V, and column 2 the found values.
%      
%        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
%        in MAXTAB and MINTAB are replaced with the corresponding
%        X-values.
%
%        A point is considered a maximum peak if it has the maximal
%        value, and was preceded (to the left) by a value lower by
%        DELTA.

% Eli Billauer, 3.4.05 (Explicitly not copyrighted).
% This function is released to the public domain; Any use is allowed.

maxtab = [];
mintab = [];

v = v(:); % Just in case this wasn't a proper vector

if nargin < 3
  x = (1:length(v))';
else 
  x = x(:);
  if length(v)~= length(x)
    error('Input vectors v and x must have same length');
  end
end
  
if (length(delta(:)))>1
  error('Input argument DELTA must be a scalar');
end

if delta <= 0
  error('Input argument DELTA must be positive');
end

mn = Inf; mx = -Inf;
mnpos = NaN; mxpos = NaN;

lookformax = 1;

for i=1:length(v)
  this = v(i);
  if this > mx, mx = this; mxpos = x(i); end
  if this < mn, mn = this; mnpos = x(i); end
  
  if lookformax
    if this < mx-delta
      maxtab = [maxtab ; mxpos mx];
      mn = this; mnpos = x(i);
      lookformax = 0;
    end  
  else
    if this > mn+delta
      mintab = [mintab ; mnpos mn];
      mx = this; mxpos = x(i);
      lookformax = 1;
    end
  end
end

end % end of peakdet