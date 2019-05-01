%2 Iterate all the points to get the integral image
function ii = get_integral_image(I) 
[row,col] = size(I);
ii=zeros(row,col);
for i=1:row
    for j=1:col
        s=sum(I(1:i,j));
        if(j-1<=0)
            ii(i,j) = s;
        else
            ii(i,j)=s+ii(i,j-1);
        end
        
   end
end
