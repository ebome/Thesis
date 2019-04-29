function feature=Tamura(imag)
             feature=zeros(1,3);   
             Fcrs=coarseness(imag,4);
             Fcon=contrast(imag);
             [Fdir,sita]=directionality(imag);
             feature(1,:)=[ Fcrs, Fcon,Fdir];
            % [Fdir,sita]=directionality(imag);
            % Flin=linelikeness(imag,sita,4);
            %  Freg{k,1}=regularity(I{1,k},32);
            % Frgh=Fcrs+Fcon;
            % feature(1,:)=[ Fcrs, Fcon,Fdir,Flin,Frgh];
end

%% Coarseness

function Fcrs = coarseness( graypic,kmax )
[h,w]=size(graypic);  
A=zeros(h,w,2^kmax);

for i=2^(kmax-1)+1:h-2^(kmax-1)  
    for j=2^(kmax-1)+1:w-2^(kmax-1)  
        for k=1:kmax  
            A(i,j,k)=mean2(graypic(i-2^(k-1):i+2^(k-1)-1,j-2^(k-1):j+2^(k-1)-1));  
        end  
    end  
end  

for i=1+2^(kmax-1):h-2^(kmax-1)  
    for j=1+2^(kmax-1):w-2^(kmax-1)  
        for k=1:kmax  
            Eh(i,j,k)=abs(A(i+2^(k-1),j,k)-A(i-2^(k-1),j));  
            Ev(i,j,k)=abs(A(i,j+2^(k-1),k)-A(i,j-2^(k-1)));  
        end  
    end  
end  

%Sbest = zeros( (h-2^(kmax-1))  );
for i=2^(kmax-1)+1:h-2^(kmax-1)  
    for j=2^(kmax-1)+1:w-2^(kmax-1)  
        [maxEh,p]=max(Eh(i,j,:));  
        [maxEv,q]=max(Ev(i,j,:));  
        if maxEh>maxEv  
            maxkk=p;  
        else  
            maxkk=q;  
        end  
        Sbest(i,j)=2^maxkk;  
    end  
end  
 
Fcrs=mean2(Sbest);  

end

%% Contrast

function Fcon=contrast(graypic)  
graypic=double(graypic); 
x=graypic(:);  
M4=mean((x-mean(x)).^4); 
delta2=var(x,1); 
alfa4=M4/(delta2^2);   
delta=std(x,1); 
Fcon=delta/(alfa4^(1/4));   
end  

%% Directionality
function [Fdir,sita]=directionality(graypic)  
[h,w]=size(graypic);  
  
GradientH=[-1 0 1;-1 0 1;-1 0 1];  
GradientV=[ 1 1 1;0 0 0;-1 -1 -1];  

MHconv=conv2(graypic,GradientH);  
MH=MHconv(3:h,3:w);  
MVconv=conv2(graypic,GradientV);  
MV=MVconv(3:h,3:w);  
 
MG=(abs(MH)+abs(MV))./2;  
  
validH=h-2;  
validW=w-2;  
 
for i=1:validH  
    for j=1:validW  
        sita(i,j)=atan(MV(i,j)/MH(i,j))+(pi/2);  
    end  
end  
n=16;  
t=12;  
Nsita=zeros(1,n);  

for i=1:validH  
    for j=1:validW  
        for k=1:n  
            if sita(i,j)>=(2*(k-1)*pi/2/n) && sita(i,j)<((2*(k-1)+1)*pi/2/n) && MG(i,j)>=t  
                Nsita(k)=Nsita(k)+1;  
            end  
        end  
    end  
end  
for k=1:n  
    HD(k)=Nsita(k)/sum(Nsita(:));  
end  

FIp=max(HD);  
Fdir=0;  
for k=1:n  
    Fdir=Fdir+(k-FIp)^2*HD(k);
end  
end  
%%  Linelikeness
function Flin=linelikeness(graypic,sita,d) 
n=16;  
[h,w]=size(graypic);  
  
PDd1=zeros(n,n);  
PDd2=zeros(n,n);  
PDd3=zeros(n,n);  
PDd4=zeros(n,n);  
PDd5=zeros(n,n);  
PDd6=zeros(n,n);  
PDd7=zeros(n,n);  
PDd8=zeros(n,n);  
for i=d+1:h-d-2  
    for j=d+1:w-d-2  
        for m1=1:n  
            for m2=1:n  
                   
                if (sita(i,j)>=(2*(m1-1)*pi/2/n) && sita(i,j)<((2*(m1-1)+1)*pi/2/n)) && (sita(i+d,j)>=(2*(m2-1)*pi/2/n) && sita(i+d,j)<((2*(m2-1)+1)*pi/2/n))  
                    PDd1(m1,m2)=PDd1(m1,m2)+1;  
                end  
                 
                if (sita(i,j)>=(2*(m1-1)*pi/2/n) && sita(i,j)<((2*(m1-1)+1)*pi/2/n)) && (sita(i-d,j)>=(2*(m2-1)*pi/2/n) && sita(i-d,j)<((2*(m2-1)+1)*pi/2/n))  
                    PDd2(m1,m2)=PDd2(m1,m2)+1;  
                end  
                  
                if (sita(i,j)>=(2*(m1-1)*pi/2/n) && sita(i,j)<((2*(m1-1)+1)*pi/2/n)) && (sita(i,j+d)>=(2*(m2-1)*pi/2/n) && sita(i,j+d)<((2*(m2-1)+1)*pi/2/n))  
                    PDd3(m1,m2)=PDd3(m1,m2)+1;  
                end  
                  
                if (sita(i,j)>=(2*(m1-1)*pi/2/n) && sita(i,j)<((2*(m1-1)+1)*pi/2/n)) && (sita(i,j-d)>=(2*(m2-1)*pi/2/n) && sita(i,j-d)<((2*(m2-1)+1)*pi/2/n))  
                    PDd4(m1,m2)=PDd4(m1,m2)+1;  
                end  
                 
                if (sita(i,j)>=(2*(m1-1)*pi/2/n) && sita(i,j)<((2*(m1-1)+1)*pi/2/n)) && (sita(i+d,j+d)>=(2*(m2-1)*pi/2/n) && sita(i+d,j+d)<((2*(m2-1)+1)*pi/2/n))  
                    PDd5(m1,m2)=PDd5(m1,m2)+1;  
                end  
                  
                if (sita(i,j)>=(2*(m1-1)*pi/2/n) && sita(i,j)<((2*(m1-1)+1)*pi/2/n)) && (sita(i-d,j+d)>=(2*(m2-1)*pi/2/n) && sita(i-d,j+d)<((2*(m2-1)+1)*pi/2/n))  
                    PDd6(m1,m2)=PDd6(m1,m2)+1;  
                end  
                  
                if (sita(i,j)>=(2*(m1-1)*pi/2/n) && sita(i,j)<((2*(m1-1)+1)*pi/2/n)) && (sita(i+d,j-d)>=(2*(m2-1)*pi/2/n) && sita(i+d,j-d)<((2*(m2-1)+1)*pi/2/n))  
                    PDd7(m1,m2)=PDd7(m1,m2)+1;  
                end  
                 
                if (sita(i,j)>=(2*(m1-1)*pi/2/n) && sita(i,j)<((2*(m1-1)+1)*pi/2/n)) && (sita(i-d,j-d)>=(2*(m2-1)*pi/2/n) && sita(i-d,j-d)<((2*(m2-1)+1)*pi/2/n))  
                    PDd8(m1,m2)=PDd8(m1,m2)+1;  
                end  
            end  
        end  
    end  
end  
f=zeros(1,8);  
g=zeros(1,8);  
for i=1:n  
    for j=1:n  
        f(1)=f(1)+PDd1(i,j)*cos((i-j)*2*pi/n);  
        g(1)=g(1)+PDd1(i,j);  
        f(2)=f(2)+PDd2(i,j)*cos((i-j)*2*pi/n);  
        g(2)=g(2)+PDd2(i,j);  
        f(3)=f(3)+PDd3(i,j)*cos((i-j)*2*pi/n);  
        g(3)=g(3)+PDd3(i,j);  
        f(4)=f(4)+PDd4(i,j)*cos((i-j)*2*pi/n);  
        g(4)=g(4)+PDd4(i,j);  
        f(5)=f(5)+PDd5(i,j)*cos((i-j)*2*pi/n);  
        g(5)=g(5)+PDd5(i,j);  
        f(6)=f(6)+PDd6(i,j)*cos((i-j)*2*pi/n);  
        g(6)=g(6)+PDd6(i,j);  
        f(7)=f(7)+PDd7(i,j)*cos((i-j)*2*pi/n);  
        g(7)=g(7)+PDd7(i,j);  
        f(8)=f(8)+PDd8(i,j)*cos((i-j)*2*pi/n);  
        g(8)=g(4)+PDd8(i,j);  
    end  
end  
tempM=f./g;  
Flin=max(tempM);
end  


%% Regularity
function Freg=regularity(graypic,windowsize) 
[h,w]=size(graypic);  
k=0;  
for i=1:windowsize:h-windowsize  
    for j=1:windowsize:w-windowsize  
        k=k+1;  
        crs(k)=coarseness(graypic(i:i+windowsize-1,j:j+windowsize-1),4);   
        con(k)=contrast(graypic(i:i+windowsize-1,j:j+windowsize-1));   
        [dire(k),sita]=directionality(graypic(i:i+windowsize-1,j:j+windowsize-1)); 
        lin=linelikeness(graypic(i:i+windowsize-1,j:j+windowsize-1),sita,4)*10; 
    end  
end 
Dcrs=std(crs,1);  
Dcon=std(con,1);  
Ddir=std(dire,1);  
Dlin=std(lin,1);
Freg=1-(Dcrs+Dcon+Ddir+Dlin)/4/100;
end
