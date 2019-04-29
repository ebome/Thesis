function tamura_feature_vector = get_tamura(G)
      tamura_feature_vector=zeros(1,3);  
      Fcontrast=contrast(G);
      Fcoarseness=coarseness(G);
      Fdirection=Fdir(G,100);
      tamura_feature_vector(1,:)=[ Fcoarseness, Fcontrast,Fdirection];

end

%%
function Fcon=contrast(graypic)  
graypic=double(graypic); 
x=graypic(:);  
M4=mean((x-mean(x)).^4); 
delta2=var(x,1); 
alfa4=M4/(delta2^2);   
delta=std(x,1); 
Fcon=delta/(alfa4^(1/4));   
end  
%-------------------Contrast-------------------
% function Fcontrast = contrast(G)
% [r,c] = size(G);%size of array
% [counts,graylevels]=imhist(G);%histogram of image
% % figure;
% % imhist(I);
% % title('Gray-level distribution')
% PI=counts/(r*c);
% averagevalue=sum(graylevels.*PI);%mean value
% u4=sum((graylevels-repmat(averagevalue,[256,1])).^4.*PI);%4th moment about mean
% variance=sum((graylevels-repmat(averagevalue,[256,1])).^2.*PI);%variance(2nd moment about mean)
% alpha4=u4/variance^2;%kurtosis
% %Contrast Value
% Fcontrast=sqrt(variance)/alpha4.^(1/4);
% end

%% -------------------Coarseness-------------------
function Fcoarseness = coarseness(G)
[r,c] = size(G);%size of array

%initialization
%Average of neighbouring pixels
A1=zeros(r,c);A2=zeros(r,c);
A3=zeros(r,c);A4=zeros(r,c);
A5=zeros(r,c);A6=zeros(r,c);
%Sbest for coarseness
Sbest=zeros(r,c);
%Subtracting for Horizontal and Vertical case
E1h=zeros(r,c);E1v=zeros(r,c);
E2h=zeros(r,c);E2v=zeros(r,c);
E3h=zeros(r,c);E3v=zeros(r,c);
E4h=zeros(r,c);E4v=zeros(r,c);
E5h=zeros(r,c);E5v=zeros(r,c);
E6h=zeros(r,c); E6v=zeros(r,c);
flag=0;%To avoid errors

%2x2  E1h and E1v
%subtracting average of neighbouring 2x2 pixels 
for x=2:r
    for y=2:c
        A1(x,y)=(sum(sum(G(x-1:x,y-1:y))));
    end
end
for x=2:r-1
    for y=2:c-1
        E1h(x,y) = A1(x+1,y)-A1(x-1,y);
        E1v(x,y) = A1(x,y+1)-A1(x,y-1);
    end
end
E1h=E1h/2^(2*1);
E1v=E1v/2^(2*1);

%4x4  E2h and E2v
if (r<4||c<4)
    flag=1;
end
%subtracting average of neighbouring 4x4 pixels
if(flag==0)
    for x=3:r-1
        for y=3:c-1
            A2(x,y)=(sum(sum(G(x-2:x+1,y-2:y+1))));
        end
    end
    for x=3:r-2
        for y=3:c-2
            E2h(x,y) = A2(x+2,y)-A2(x-2,y);
            E2v(x,y) = A2(x,y+2)-A2(x,y-2);
        end
    end
end
E2h=E2h/2^(2*2);
E2v=E2v/2^(2*2);

%8x8 E3h and E3v
if (r<8||c<8)
    flag=1;
end
%subtracting average of neighbouring 8x8 pixels
if(flag==0)
    for x=5:r-3
        for y=5:c-3
            A3(x,y)=(sum(sum(G(x-4:x+3,y-4:y+3))));
        end
    end
    for x=5:r-4
        for y=5:c-4
            E3h(x,y) = A3(x+4,y)-A3(x-4,y);
            E3v(x,y) = A3(x,y+4)-A3(x,y-4);
        end
    end
end
E3h=E3h/2^(2*3);
E3v=E3v/2^(2*3);
 
%16x16 E4h and E4v
if (r<16||c<16)
    flag=1;
end
%subtracting average of neighbouring 16x16 pixels
if(flag==0)
    for x=9:r-7
        for y=9:c-7
            A4(x,y)=(sum(sum(G(x-8:x+7,y-8:y+7))));
        end
    end
    for x=9:r-8
        for y=9:c-8
            E4h(x,y) = A4(x+8,y)-A4(x-8,y);
            E4v(x,y) = A4(x,y+8)-A4(x,y-8);
        end
    end
end
E4h=E4h/2^(2*4);
E4v=E4v/2^(2*4);
 
%32x32 E5h and E5v
if (r<32||c<32)
    flag=1;
end
%subtracting average of neighbouring 32x32 pixels
if(flag==0)
    for x=17:r-15
        for y=17:c-15
            A5(x,y)=(sum(sum(G(x-16:x+15,y-16:y+15))));
        end
    end
    for x=17:r-16
        for y=17:c-16
            E5h(x,y) = A5(x+16,y)-A5(x-16,y);
            E5v(x,y) = A5(x,y+16)-A5(x,y-16);
        end
    end
end
E5h=E5h/2^(2*5);
E5v=E5v/2^(2*5);
 
%64x64 E6h and E6v
if (r<64||c<64)
    flag=1;
end
%subtracting average of neighbouring 64x64 pixels
if(flag==0)
    for x=33:r-31
        for y=33:c-31
            A6(x,y)=(sum(sum(G(x-32:x+31,y-32:y+31))));
        end
    end
    for x=33:r-32
        for y=33:c-32
            E6h(x,y) = A6(x+32,y)-A6(x-32,y);
            E6v(x,y) = A6(x,y+32)-A6(x,y-32);
        end
    end
end
E6h=E6h/2^(2*6);
E6v=E6v/2^(2*6);

% %plots
% figure
% subplot(131);
% imshow(IColor);
% title('Original image')
% subplot(132);
% imshow(E1h);
% title('Horizontal case')
% subplot(133)
% imshow(E1v);
% title('Vertical case')

%at each point pick best size "Sbest", which gives highest output value
for i=1:r
    for j=1:c
        [maxv,index]=max([abs(E1h(i,j)),abs(E1v(i,j)),abs(E2h(i,j)),abs(E2v(i,j)),...
            abs(E3h(i,j)),abs(E3v(i,j)),abs(E4h(i,j)),abs(E4v(i,j)),abs(E5h(i,j)),...
            abs(E5v(i,j)),abs(E6h(i,j)),abs(E6v(i,j))]);
        k=floor((index+1)/2);%'k'corresponding to highest E in either direction
        Sbest(i,j)=2.^k;
    end
end 
% figure;
% plot(Sbest)
% title('Output of best size detector')
%Coarseness Value
Fcoarseness=sum(sum(Sbest))/(r*c);

end


