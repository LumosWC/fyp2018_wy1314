function [random_bbx] = generate_random_bbx (bbx_ratio,qimlist,dataset_folder)

% Input: 
%    bbx_ratio: each element is the ratio of the bounding box area [No.Query x 1]
%    qimlist
%    dataset_foldr

% Output:
%    random_bbx: each row is the randomly generated bounding box in the gnd
%    file manner [No.Query x 1]

random_bbx = zeros (numel (bbx_ratio),4);
for i = 1: numel (bbx_ratio)
    img = imread ([dataset_folder, qimlist{i}, '.jpg']) ;
    ratio = sqrt ( bbx_ratio(i) );
 
    mid_x = size (img,2)/2;
    length_x = size (img,2)*ratio/2;
    
    mid_y = size (img,1)/2;
    length_y = size (img,1)*ratio/2;
    
    random_bbx (i,:) = [mid_x-length_x mid_y-length_y mid_x+length_x mid_y+length_y];
end