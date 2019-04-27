function [categorical_label_First,categorical_label_Second] = get_ground_truth(dataPath)
%% Load ground Truth
data=load(dataPath); 
data_img_name=data(:,1);
data_First_observer=data(:,2); 
data_Second_observer=data(:,3);

% CXR images were classed as showing “mild”, “moderate” or “severe” disease, 
% according to clinician scores. These corresponded to scores of 20-or-above, 
% 15, or 10-or-below respectively, necessary as a consequence of the extremely 
% low number of images with scores of 5 or 25 (section 6.1.2), 
% too few to generate separate categories reliably. A further “mild” versus
% “severe” analysis was performed with the relevant subset of images.
data_First_observer(data_First_observer==5)=1;
data_First_observer(data_First_observer==10)=1;
data_First_observer(data_First_observer==15)=2;
data_First_observer(data_First_observer==20)=3;
data_First_observer(data_First_observer==25)=3;

data_Second_observer(data_Second_observer==5)=1;
data_Second_observer(data_Second_observer==10)=1;
data_Second_observer(data_Second_observer==15)=2;
data_Second_observer(data_Second_observer==20)=3;
data_Second_observer(data_Second_observer==25)=3;

% get clincial description label
a = length(data_First_observer);
categorical_label_First = cell(a,1);
categorical_label_Second= cell(a,1);
label='';label_2='';
for i = 1:a
    if data_First_observer(i,:)==1
        label ='severe';      
    end
    if data_First_observer(i,:)==2
        label ='moderate';
    end
    if data_First_observer(i,:)==3
        label ='mild';
    end    
    categorical_label_First{i}= label;
    
    if data_Second_observer(i,:)==1
        label_2 ='severe';      
    end
    if data_Second_observer(i,:)==2
        label_2 ='moderate';
    end
    if data_Second_observer(i,:)==3
        label_2 ='mild';
    end    
    categorical_label_Second{i}= label_2;

end    


end