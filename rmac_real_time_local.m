% ---------------------------- Preparation 1 : Set-Up -------------------------------%
addpath('helpers');
if exist('aml') ~= 3
	mex -compatibleArrayDims aml.c
end

run matconvnet-1.0-beta25/matlab/vl_setupnn;

main_folder    = '/Users/wc/Desktop/CBIR CCTV/';     % oxford5k/ and paris6k/ should be in here
dataset_train  = 'Oxford_dataset/oxbuild_images';    % dataset to learn the PCA-whitening on
dataset_test   = 'Paris6k_dataset/paris_images';     % dataset to evaluate on 

% ---------------------------- Preparation 2: Data Pre-Processing -------------------------------%
fprintf('Pre-loading database image features... \n');
load 'preprocessing.mat';

% ---------------------------- Query: Crop the Specified Object ---------------------------------%
image = 'paris_defense_000605.jpg';
%image = 'cleaning-eiffel-tower.jpg';
figure;
imshow(image); 
rect = getrect;
object = imcrop(imread(image), rect);
imshow (object);

% ---------------------------- Query: Encode the Query ---------------------------------%
qvec = rmac_regionvec(object, net, 3); % get all regional-MAC features
qvec = vecpostproc (qvec); % L2 Normalisation
qvec = apply_whiten (qvec,Xm, eigvec, eigval); % PCA-Whitening
qvec = vecpostproc (qvec); % L2 Normalisation
qvec = sum(qvec, 2); % Sum
qvec = vecpostproc (qvec); % L2 Normalisation

qvec_loc = mac(object, net); % keep a single MAC feature (for localization&reranking)
qvec_loc = vecpostproc (qvec_loc); % L2 Normalisation

% ---------------------------- Retrieval 1 : Initial Filtering ---------------------------------%
[distance,ranks] = sort(vecs'*qvec, 'descend');
% imshow ( strcat ( main_folder , dataset_test , '/', char ( gnd_test.imlist(ranks(1)) ),'.jpg' ));
ranks_rerank = ranks;
ranks_rerank_qe = ranks;

% ---------------------------- Retrieval 2 : Localization (AML) ---------------------------------%
qratio = size(object, 1) / size(object, 2);
ids_toplist = ranks(1:1000);
conv3d_toplist = arrayfun(@(y)floor((15+((conv3d{y}>=128)*128+conv3d{y}.*(conv3d{y}<128)))/16), ids_toplist, 'un', 0);
bestbox = cellfun(@(x) aml(double(x), int32(10), double(qvec_loc), qratio, 2, 3, 3, 5), conv3d_toplist, 'un', 0);

% ---------------------------- Retrieval 3 : Reranking ---------------------------------%
vec_bestbox = cellfun(@(x, b) rmac_regionvec_act(x(b(2):b(3), b(4):b(5), :), 3), conv3d_toplist, bestbox, 'un', 0); 
vec_bestbox = cellfun(@(x) vecpostproc (x),vec_bestbox,'un', 0);
vec_bestbox = cellfun(@(x) apply_whiten(x, Xm, eigvec, eigval),vec_bestbox,'un', 0);
vec_bestbox = cellfun(@(x) vecpostproc (x),vec_bestbox,'un', 0);
vec_bestbox = cellfun(@(x) sum(x,2),vec_bestbox,'un', 0);
vec_bestbox = cellfun(@(x) vecpostproc (x),vec_bestbox,'un', 0);

%(Optional Shortening)
%vecs_bestbox = cellfun(@(x, b) vecpostproc(rmac_regionvec_act(x(b(2):b(3), b(4):b(5), :), 3)), conv3d_toplist, bestbox, 'un', 0);
%vecs_bestbox = cellfun(@(y) vecpostproc(sum(vecpostproc(apply_whiten(y, Xm, eigvec, eigval)), 2)), vecs_bestbox, 'un', 0);

scores_rerank = qvec'*cell2mat(vec_bestbox');
[~, idx] = sort(scores_rerank, 'descend');
ranks_rerank(1:1000) = ranks_rerank(idx);
% ---------------------------- Retrieval 4 : Query Expansion ---------------------------------%
scores_rerank_qe = mean([cell2mat({vec_bestbox{idx(1:5)}}), qvec]')*cell2mat(vec_bestbox');
[~, idx] = sort(scores_rerank_qe, 'descend');
ranks_rerank_qe(1:1000) = ranks_rerank_qe(idx);