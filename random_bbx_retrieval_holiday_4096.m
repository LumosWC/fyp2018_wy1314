addpath helpers/
addpath local_data/

if exist('aml') ~= 3
	mex -compatibleArrayDims aml.c
end

run ./matconvnet-1.0-beta25/matlab/vl_setupnn.m

main_folder = '/Users/wc/Desktop/CBIR CCTV/'; % oxford5k/ and paris6k/ should be in here

dataset			= 'Holiday_dataset';     % dataset to evaluate on 
%dataset        = 'Oxford_dataset/oxbuild_images';  
%dataset        = 'Paris6k_dataset/paris_images';
dataset_folder = [main_folder, dataset, '/'];

% --------------------------------------------------------------------------------
fprintf('Loading Neural Networks... \n');
load VGG16Net
load gnd_holiday

fprintf('Preparing Database... \n');
load fc128_holiday
load fc4096_holiday

fprintf('Preparing Preprocessing... \n');
load pca_reduce_holiday

fprintf ('Cropping Query Images... \n');
load bbx_distribution
%bbx_ratio = generate_random_bbx_ratio ('gnd_holiday',bbx_distribution);
bbx_ratio = generate_specified_bbx_ratio ('gnd_holiday',0.5);
qimlist = {gnd_holiday.imlist{gnd_holiday.qidx}}; 
random_bbx = generate_random_bbx (bbx_ratio,qimlist,dataset_folder);

qim = arrayfun(@(x) crop_qim([dataset_folder, qimlist{x}, '.jpg'], random_bbx(x,1:4)), 1:numel(gnd_holiday.qidx), 'un', 0);

fprintf ('Encoding Query Features... \n');
qvecs = zeros (4096,numel (gnd_holiday.qidx));
for i = 1:numel (gnd_holiday.qidx)
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
% retrieval with inner product
[sim,ranks] = sort(fc4096_holiday'*qvecs, 'descend');
map = compute_map (ranks, gnd_holiday.gnd);
fprintf('mAP (4096), without re-ranking = %.4f\n', map);

qvecs_128 = pca_reduce_holiday' * qvecs ; 
[sim,ranks] = sort(fc128_holiday'*qvecs_128, 'descend');
map = compute_map (ranks, gnd_holiday.gnd);
fprintf('mAP (128), without re-ranking = %.4f\n', map);



%------------------------------

query = floor ( rand * numel ( gnd_holiday.qidx ));
figure;
im_index =  gnd_holiday.qidx(query);
image = imread ( strcat(dataset_folder,gnd_holiday.imlist{im_index} ,'.jpg'));
imshow(image);
bounding_box_tmp  = random_bbx(query,:);
bounding_box = zeros (1,4);
bounding_box (1) = bounding_box_tmp(1);
bounding_box (2) = bounding_box_tmp(2);
bounding_box (3) = bounding_box_tmp(3) - bounding_box_tmp(1);
bounding_box (4) = bounding_box_tmp(4) - bounding_box_tmp(2);

rectangle ('Position',bounding_box  ,'EdgeColor','g','LineWidth',2);
title (strcat('query',string(query),'Correspondence Number',string(numel(gnd_holiday.gnd(query).ok))));
xlabel (strcat ('crop ratio','   ',string (bbx_ratio(query))));

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