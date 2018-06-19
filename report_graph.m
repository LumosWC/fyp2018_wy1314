% load gnd_oxford5k
% gnd = gnd_oxford5k;
% dataset			= 'Oxford_dataset/oxbuild_images';  
addpath local_data/

load gnd_holiday
gnd = gnd_holiday;
dataset = 'Holiday_dataset';
main_folder = '/Users/wc/Desktop/CBIR CCTV/';  
dataset_folder = [main_folder, dataset, '/'];
query = floor ( rand * numel ( gnd.qidx ));
figure;
im_index =  gnd.qidx(query);
image = imread ( strcat(dataset_folder,gnd.imlist{im_index} ,'.jpg'));
imshow(image);

load gnd_paris6k
gnd = gnd_paris6k;
dataset			= 'Paris6k_dataset/paris_images';  

main_folder = '/Users/wc/Desktop/CBIR CCTV/';  
dataset_folder = [main_folder, dataset, '/'];

query = floor ( rand * numel ( gnd.qidx ));
figure;
im_index =  gnd.qidx(query);
image = imread ( strcat(dataset_folder,gnd.imlist{im_index} ,'.jpg'));
imshow(image);
bounding_box_tmp  = gnd.gnd(query).bbx;
bounding_box = zeros (1,4);
bounding_box (1) = bounding_box_tmp(1);
bounding_box (2) = bounding_box_tmp(2);
bounding_box (3) = bounding_box_tmp(3) - bounding_box_tmp(1);
bounding_box (4) = bounding_box_tmp(4) - bounding_box_tmp(2);

rectangle ('Position',bounding_box  ,'EdgeColor','g','LineWidth',2);
title (strcat('query',string(query)));

figure;
for i = 1 : 6
    [image , map ] = imread(strcat(dataset_folder,gnd_oxford5k.imlist{ranks(i,query)},'.jpg'));
    subplot(2,3,i), imshow(image ,map) 
    
    if ismember   (ranks(i,query),gnd_oxford5k.gnd(query).ok )
        rectangle ('Position', [1 1 size(image,2)-1 size(image,1)-1] ,'EdgeColor','g','LineWidth',2);
    elseif ismember   (ranks(i,query),gnd_oxford5k.gnd(query).junk )
        rectangle ('Position', [1 1 size(image,2)-1 size(image,1)-1] ,'EdgeColor','b','LineWidth',2);
    else
        rectangle ('Position', [1 1 size(image,2)-1 size(image,1)-1] ,'EdgeColor','r','LineWidth',2);  
    end 
end