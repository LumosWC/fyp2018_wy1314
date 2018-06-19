addpath helpers/
addpath local_data/
% version: matconvnet-1.0-beta25
run ./matconvnet-1.0-beta25/matlab/vl_setupnn

load gnd_holiday

if exist('aml') ~= 3
	mex -compatibleArrayDims aml.c
end

data_folder = '/Users/wc/Desktop/CBIR CCTV/'; % oxford5k/ and paris6k/ should be in here
dataset			= 'Holiday_dataset';     % dataset to evaluate on 
dataset_folder = [data_folder, dataset, '/'];

fprintf('Loading Neural Networks... \n');
load VGG16Net

fprintf('Preparing Database... \n');
load fc4096_holiday.mat

fprintf('Preparing Preprocessing... \n');
[coeff, score, latent] = pca(fc4096_holiday');
pca_reduce_holiday= coeff(:,1:128);
fc128_holiday = pca_reduce_holiday' * fc4096_holiday;

fprintf ('Extracing Query Images... \n');

qvecs_4096 = fc4096_holiday(:,gnd_holiday.qidx);
qvecs_128 = pca_reduce_holiday'*qvecs_4096;

fprintf('Initial Filtering... \n');

% retrieval with inner product
[sim,ranks] = sort(fc4096_holiday'*qvecs_4096, 'descend');
map = compute_map (ranks, gnd_holiday.gnd); % can be used in my own code
fprintf('mAP (4096), without re-ranking = %.4f\n', map);

[sim,ranks] = sort(fc128_holiday'*qvecs_128, 'descend');
map = compute_map (ranks, gnd_holiday.gnd); % can be used in my own code
fprintf('mAP (128), without re-ranking = %.4f\n', map);


query = floor ( rand * numel ( gnd_holiday.qidx ));
figure;
im_index =  gnd_holiday.qidx(query);
image = imread ( strcat(dataset_folder,gnd_holiday.imlist{im_index} ,'.jpg'));
imshow(image);
title (strcat('query',string(query)));

figure;
for i = 1 : 6
    [image , map ] = imread(strcat(dataset_folder,gnd_holiday.imlist{ranks(i,query)},'.jpg'));
    subplot(2,3,i), imshow(image ,map) 
    
    if ismember   (ranks(i,query),gnd_holiday.gnd(query).ok )
        rectangle ('Position', [1 1 size(image,2)-1 size(image,1)-1] ,'EdgeColor','g','LineWidth',2);

    else
        rectangle ('Position', [1 1 size(image,2)-1 size(image,1)-1] ,'EdgeColor','r','LineWidth',2);  
    end 
end

