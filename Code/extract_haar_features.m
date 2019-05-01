function F = extract_haar_features(ii)

F=[];
[row,col]=size(ii);
k=floor(min(row,col)/24);
    for t=1:5
        if t==1 % Get two-rectangle feature, white on top & black on bottom
            minw = 2*k; % minimal width
            minh = 4*k; % minimal height
            aw=1; % scaling rate,along direction of w
            ah=1; % scaling rate,along direction of h
            F1=[];
            F0=[];
            w=minw;
            h=minh;
            while(minh*ah<=row)    
                while(minw*aw<=col)
                    n=0;
                    for i=1:k:row-h
                        for j=1:k:col-w
                            n=n+1;
                            white = ii(i,j)+ii(i+h/2,j+w)-ii(i+h/2,j)-ii(i,j+w);
                            black= ii(i+h/2,j)+ii(i+h,j+w)-ii(i+h,j)-ii(i+h/2,j+w);
                            F0(n)=white-black;
                        end
                    end
                    F1=[F1,F0];
                    F0=[];
                    aw=aw+1;
                    w=minw*aw;
                end
                ah=ah+1;
                h=minh*ah;
                w=minw; % back to the inital
                aw=1;   % back to the inital
            end
        end
        if t==2 % Get two-rectangle feature, white on left & black on right
            minw = 4*k; % minimal width
            minh = 2*k; % minimal height
            aw=1; % scaling rate,along direction of w
            ah=1; % scaling rate,along direction of h
            F2=[];
            F0=[];
            w=minw;
            h=minh;
            [row,col]=size(ii);
            while(minh*ah<=row)    
                while(minw*aw<=col)
                    n=0;
                    for i=1:k:row-h
                        for j=1:k:col-w
                            n=n+1;
                            white = ii(i,j)+ii(i+h,j+w/2)-ii(i,j+w/2)-ii(i+h,j);
                            black = ii(i,j+w/2)+ii(i+h,j+w)-ii(i,j+w)-ii(i+h,j+w/2);
                            F0(n)=white-black;
                        end
                    end
                    F2=[F2,F0];   
                    F0=[];
                    aw=aw+1;
                    w=minw*aw;
                end
                ah=ah+1;
                h=minh*ah;
                w=minw; % back to the inital
                aw=1;   % back to the inital
            end
        end
        if t==3  % Get three-rectangle feature, white on top & black on middle & white on bottom
            minw = 2*k; % minimal width
            minh = 3*k; % minimal height
            aw=1; % scaling rate,along direction of w
            ah=1; % scaling rate,along direction of h
            F3=[];
            F0=[];
            w=minw;
            h=minh;
            [row,col]=size(ii);
            while(minh*ah<=row)    
                while(minw*aw<=col)
                    n=0;
                    for i=1:k:row-h
                        for j=1:k:col-w
                            n=n+1;
                            awhite = ii(i,j)+ii(i+h/3,j+w)-ii(i+h/3,j)-ii(i,j+w);% upper white
                            black= ii(i+h/3,j)+ii(i+2*h/3,j+w)-ii(i+2*h/3,j)-ii(i+h/3,j+w);% middle back
                            bwhite = ii(i+2*h/3,j)+ii(i+h,j+w)-ii(i+h,j)-ii(i+2*h/3,j+w);% lower white
                            F0(n)=awhite+bwhite-2*black;
                        end
                    end
                    F3=[F3,F0];
                    F0=[];
                    aw=aw+1;
                    w=minw*aw;
                end
                ah=ah+1;
                h=minh*ah;
                w=minw; % back to the inital
                aw=1;   % back to the inital
            end
        end 
        if t==4  % Get three-rectangle feature, white on left & black on middle & white on right
            minw = 3*k; % minimal width
            minh = 2*k; % minimal height
            aw=1; % scaling rate,along direction of w
            ah=1; % scaling rate,along direction of h
            F4=[];
            F0=[];
            w=minw;
            h=minh;
            [row,col]=size(ii);
            while(minh*ah<=row)    
                while(minw*aw<=col)
                    n=0;
                    for i=1:k:row-h
                        for j=1:k:col-w
                            n=n+1;
                            lwhite = ii(i,j)+ii(i+h,j+w/3)-ii(i,j+w/3)-ii(i+h,j);% left white
                            black= ii(i,j+w/3)+ii(i+h,j+2*w/3)-ii(i,j+2*w/3)-ii(i+h,j+w/3);% midlle back
                            rwhite = ii(i,j+2*w/3)+ii(i+h,j+w)-ii(i,j+w)-ii(i+h,j+2*w/3);% right white
                            F0(n)=lwhite+rwhite-2*black;
                        end
                    end
                    F4=[F4,F0];
                    F0=[];
                    aw=aw+1;
                    w=minw*aw;
                end
                ah=ah+1;
                h=minh*ah;
                w=minw; % back to the inital
                aw=1;   % back to the inital
            end
        end 
        if t==5% Get four-rectangle feature, upper left and lower right are white & upper right and lower left are black 
            minw = 4*k; % minimal width
            minh = 4*k; % minimal height
            aw=1; % scaling rate,along direction of w
            ah=1; % scaling rate,along direction of h
            F5=[];
            F0=[];
            w=minw;
            h=minh;
            [row,col]=size(ii);
            while(minh*ah<=row)    
                while(minw*aw<=col)
                    n=0;
                    for i=1:k:row-h
                        for j=1:k:col-w
                            n=n+1;
                            lwhite = ii(i,j)+ii(i+h/2,j+w/2)-ii(i+h/2,j)-ii(i,j+w/2); % upper left white
                            lblack= ii(i+h/2,j)+ii(i+h,j+w/2)-ii(i+h,j)-ii(i+h/2,j+w/2); % lower left black
                            rblack = ii(i,j+w/2)+ii(i+h/2,j+w)-ii(i+h/2,j+w/2)-ii(i,j+w); % upper right black
                            rwhite =ii(i+h/2,j+w/2)+ii(i+h,j+w)-ii(i+h,j+w/2)-ii(i+h/2,j+w); % lower right white 
                            F0(n)=lwhite+rwhite-lblack-rblack;
                        end
                    end
                    F5=[F5,F0];
                    F0=[];
                    aw=aw+1;
                    w=minw*aw;
                end
                ah=ah+1;
                h=minh*ah;
                w=minw; % back to the inital
                aw=1;   % back to the inital
            end
        end
    end
    F=[F1,F2,F3,F4,F5];


    

