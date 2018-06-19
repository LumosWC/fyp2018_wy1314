addpath local_data/
addpath helpers/
addpath matconvnet-1.0-beta25/
run ./matconvnet-1.0-beta25/matlab/vl_setupnn

load AlexNet.mat
load Xm_oxford5k.mat
load eigval_oxford5k.mat
load eigvec_oxford5k.mat
load rmac_paris6k.mat
load conv3d_paris6k.mat
load gnd_paris6k.mat

main_folder     = '/Users/wc/Desktop/CBIR CCTV/'; 
dataset			= 'Paris6k_dataset/paris_images';   
dataset_folder  = [main_folder, dataset, '/'];
% ---------------------------- Query: Crop the Specified Object ---------------------------------%
% image = 'shengmu.jpeg';
image = 'paris_general_002377.jpg';
figure;
imshow(image); 
rect = getrect;
object = imcrop(imread(image), rect);
imshow (object);

% query = floor ( rand * numel ( gnd_paris6k.qidx ));
% query = 20;
% figure;
% im_index =  gnd_paris6k.qidx(query);
% imfn = strcat(dataset_folder,gnd_paris6k.imlist{im_index} ,'.jpg');
% image = imread ( imfn);
% imshow(image);
% bounding_box_tmp  = gnd_paris6k.gnd(query).bbx;
% bounding_box = zeros (1,4);
% bounding_box (1) = bounding_box_tmp(1);
% bounding_box (2) = bounding_box_tmp(2);
% bounding_box (3) = bounding_box_tmp(3) - bounding_box_tmp(1);
% bounding_box (4) = bounding_box_tmp(4) - bounding_box_tmp(2);
% 
% rectangle ('Position',bounding_box  ,'EdgeColor','g','LineWidth',2);
% title (strcat('Query ',string(querypro
% object = crop_qim (imfn,gnd_paris6k.gnd(query).bbx);
% figure;
% title (strcat('Cropped Query',string(query)));
% imshow (object);
% ---------------------------- Query: Encode the Query ---------------------------------%
qvec = rmac_regionvec(object, AlexNet, 3); % get all regional-MAC features
qvec = vecpostproc (qvec); % L2 Normalisation
qvec = apply_whiten (qvec,Xm_oxford5k, eigvec_oxford5k, eigval_oxford5k); % PCA-Whitening
qvec = vecpostproc (qvec); % L2 Normalisation
qvec = sum(qvec, 2); % Sum
qvec = vecpostproc (qvec); % L2 Normalisation

qvec_loc = mac(object, AlexNet); % keep a single MAC feature (for localization&reranking)
qvec_loc = vecpostproc (qvec_loc); % L2 Normalisation

% ---------------------------- Retrieval 1 : Initial Filtering ---------------------------------%
[distance,ranks] = sort(rmac_paris6k'*qvec, 'descend');
% imshow ( strcat ( data_folder , dataset_test , '/', char ( gnd_test.imlist(ranks(1)) ),'.jpg' ));
ranks_rerank = ranks;

% ---------------------------- Retrieval 2 : Localization (AML) ---------------------------------%
qratio = size(object, 1) / size(object, 2);
ids_toplist = ranks(1:1000);
conv3d_toplist = arrayfun(@(y)floor((15+((conv3d_paris6k{y}>=128)*128+conv3d_paris6k{y}.*(conv3d_paris6k{y}<128)))/16), ids_toplist, 'un', 0);
bestbox = cellfun(@(x) aml(double(x), int32(10), double(qvec_loc), qratio, 1.1, 3, 3, 5), conv3d_toplist, 'un', 0);
%bestbox= cellfun(@(x) aml(double(x), int32(10), double(qvecs_loc{q}), qratio, qratio_t, step_box, rf_step, rf_iter), conv3d_toplist, 'un', 0);   

% get R-MAC from the localized windows
vecs_bestbox = cellfun(@(x, b) vecpostproc(rmac_regionvec_act(x(b(2):b(3), b(4):b(5), :), 3)), conv3d_toplist, bestbox, 'un', 0);
vecs_bestbox = cellfun(@(y) vecpostproc(sum(vecpostproc(apply_whiten(y, Xm_oxford5k, eigvec_oxford5k, eigval_oxford5k)), 2)), vecs_bestbox, 'un', 0);
	  
% re-compute similarity and re-rank
scores_rerank = cell2mat(vecs_bestbox')' * qvec;
[~, idx] = sort(scores_rerank, 'descend');
ranks_rerank(1:1000) = ranks_rerank(idx);


for i = 1:10 % the most similar reranked results
    imfn = strcat (dataset_folder,char (gnd_paris6k.imlist(ranks_rerank(i))),'.jpg');
    image_tmp = imread (imfn);
    conv3d_tmp = conv3d_toplist{idx(i)};
    bestbox_tmp = bestbox{idx(i)};
    [bbx,im_crop] = feature2image_coordinate (image_tmp,conv3d_tmp,bestbox_tmp);
    location = bbx2location (bbx);

    figure;
    hold on;
    imshow (image_tmp);
    rectangle ('Position',location,'EdgeColor','g','LineWidth',2);
    hold off; 
end