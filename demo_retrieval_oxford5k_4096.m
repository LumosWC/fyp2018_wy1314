addpath helpers/
addpath local_data/
addpath matconvnet-1.0-beta25/

run ./matconvnet-1.0-beta25/matlab/vl_setupnn

load gnd_oxford5k

if exist('aml') ~= 3
	mex -compatibleArrayDims aml.c
end

main_folder    = '/Users/wc/Desktop/CBIR CCTV/';      % oxford5k/ and paris6k/ should be in here
dataset        = 'Oxford_dataset/oxbuild_images';     % dataset to evaluate on 
dataset_folder = [main_folder, dataset, '/'];

fprintf('Loading Neural Networks... \n');
load VGG16Net

fprintf('Preparing Database... \n');
tic;
load fc4096_oxford5k
fprintf('Loading all local database image features took %.4f Seconds\n', toc);

fprintf('Preparing Preprocessing... \n');
tic;
[coeff, score, latent] = pca(fc4096_oxford5k');
fprintf('Calculating data eigenvectors took %.4f Seconds\n', toc);
pca_reduce_oxford5k = coeff(:,1:128); % each coloum in coeff is the normalized eigenvectors of convariance matrix of fc4096_oxford5k'
fc128_oxford5k = pca_reduce_oxford5k' * fc4096_oxford5k;

fprintf ('Extracing Query Images... \n');
qimlist = {gnd_oxford5k.imlist{gnd_oxford5k.qidx}}; 
qim = arrayfun(@(x) crop_qim([dataset_folder, qimlist{x}, '.jpg'], gnd_oxford5k.gnd(x).bbx), 1:numel(gnd_oxford5k.qidx), 'un', 0);

qvecs = zeros (4096,numel (gnd_oxford5k.qidx));
for i = 1:numel (gnd_oxford5k.qidx)
%     nameImg = strcat(dataset_folder,qimlist{i},'.jpg');
%     oriImg = imread(nameImg);
%     image = single(oriImg);
    
    image = single(qim{i}) ; % cropped groundtruth version
    image = imresize(image, VGG16Net.meta.normalization.imageSize(1:2)) ; 
    image = image - VGG16Net.meta.normalization.averageImage ;
    res = vl_simplenn(VGG16Net, image) ;
    
    feat = res(20).x; %the output of layer 20th in vgg16, which is the 2nd fc layer with 4096 neurons
    feat = feat(:);
    feat = feat./norm(feat);
    
    qvecs (:,i) = feat;
    fprintf('Extracting Image %d ... \n', i);
end

fprintf('Initial Filtering... \n');
[sim,ranks] = sort(fc4096_oxford5k'*qvecs, 'descend');
map = compute_map (ranks, gnd_oxford5k.gnd);
fprintf('mAP (4096), without re-ranking = %.4f\n', map);

qvecs = pca_reduce_oxford5k' * qvecs ; 
[sim,ranks] = sort(fc128_oxford5k'*qvecs, 'descend');
map = compute_map (ranks, gnd_oxford5k.gnd);
fprintf('mAP (128), without re-ranking = %.4f\n', map);

% -------------------------------
query = floor ( rand * numel ( gnd_oxford5k.qidx ));
figure;
im_index =  gnd_oxford5k.qidx(query);
image = imread ( strcat(dataset_folder,gnd_oxford5k.imlist{im_index} ,'.jpg'));
imshow(image);
bounding_box_tmp  = gnd_oxford5k.gnd(query).bbx;
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
