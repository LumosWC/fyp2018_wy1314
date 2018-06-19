% Code for the methods presented in the paper 
% G. Tolias, R. Sicre and H. Jegou, Particular object retrieval with integral max-pooling of CNN activations, ICLR 2016.
% This version of the code is not optimized to run efficiently
% but to be easily readable and to reproduce the results of the paper
%
% Authored by G. Tolias, 2015. 

addpath('helpers');

if exist('aml') ~= 3
	mex -compatibleArrayDims aml.c
end

main_folder = '/Users/wc/Desktop/CBIR CCTV/'; % oxford5k/ and paris6k/ should be in here

dataset_train				= 'Oxford_dataset/oxbuild_images';    % dataset to learn the PCA-whitening on
dataset_test 				= 'Paris6k_dataset/paris_images';     % dataset to evaluate on 

% config files for Oxford and Paris datasets
gnd_test = load(['local_data/', '/gnd_', 'paris6k', '.mat']);  gnd_test = gnd_test.gnd_paris6k;  
gnd_train = load(['local_data/', '/gnd_', 'oxford5k', '.mat']); gnd_train = gnd_train.gnd_oxford5k;

% image files are expected under each dataset's folder
im_folder_test = [main_folder, dataset_test, '/'];
im_folder_train = [main_folder, dataset_train, '/'];

% parameters of the method
use_rmac 				= 1;  %1 	% use R-MAC, otherwise use MAC
rerank 					= 1000;  %1000	% number of images to re-rank, no re-ranking if 0
L 						= 3;  % number of levels in the region pyramid of R-MAC

step_box 				= 3;		% parameter t in the paper
qratio_t 				= 1.1;   % parameter s in the paper
rf_step 				= 3;		% fixed step for refinement
rf_iter 				= 5;		% number of iterations of refinement
nqe 					= 5;		% number of images to be used for QE

use_gpu	 				= 0;		% use GPU to get CNN responses

% choose pre-trained CNN model
modelfn = 'imagenet-caffe-alex.mat';;  lid = 15;				% use AlexNet
% modelfn = 'imagenet-vgg-verydeep-16.mat'; ; lid = 31;		% use VGG

% the models used in our paper are downloaded
% current models on matconvnet site are slightly different

% if exist(modelfn, 'file') ~= 2
% 	system(sprintf('wget http://cmp.felk.cvut.cz/~toliageo/ext/iclr16/%s', modelfn)); 
% end

% matconvnet is a prerequisite
% run vl_setupnn for your installation to avoid downloading and compiling again
if exist('vl_nnconv') ~= 3
	if ~exist('matconvnet-1.0-beta25')
		system('wget http://www.vlfeat.org/matconvnet/download/matconvnet-1.0-beta25.tar.gz');
		system('tar -xzvf matconvnet-1.0-beta25.tar.gz');
	end
	cd matconvnet-1.0-beta25/matlab/
	if numel(dir(fullfile('mex', 'vl_nnconv.mex*'))) == 0
		vl_compilenn('verbose', 1, 'enableGPU', 1, 'cudaRoot', '/usr/local/cuda-8.0');
	end
	vl_setupnn;
	cd ../../
end

net = load(modelfn);
net.layers = {net.layers{1:lid}}; % remove fully connected layers
if use_gpu
	net = vl_simplenn_move(net, 'gpu') ;
end

% compatibility with matconvnet-1.0-beta25 (otherwise tested with matconvnet-1.0-beta15)
for i=1:numel(net.layers), if strcmp(net.layers{i}.type,'conv'), net.layers{i}.dilate=[1 1]; net.layers{i}.opts={}; end, end
for i=1:numel(net.layers), if strcmp(net.layers{i}.type,'relu'), net.layers{i}.leak=0; end, end
for i=1:numel(net.layers), if strcmp(net.layers{i}.type,'pool'), net.layers{i}.opts={}; end, end

im_fn_test = cellfun(@(x) [im_folder_test, x, '.jpg'], gnd_test.imlist, 'un', 0); % test is 'paris6k'
im_fn_train = cellfun(@(x) [im_folder_train, x, '.jpg'], gnd_train.imlist, 'un', 0);% train is 'oxford5k'

% extract features
fprintf('Extracting features\n');

% rerank % additionally keep the 3D tensor of responses in case re-ranking is performed
[vecs, conv3d] = cellfun(@(x) rmac_regionvec(imread(x), net, L), im_fn_train, 'un', 0); % Step 1
% conv3d is the (W x H x 256) feature map from conv_5 layer
[vecs] = cellfun(@(x) vecpostproc(x), vecs, 'un', 0);% Step 2

% rerank

% for training set, no need to keep conv3d feature maps
% just get all (training_size x 256) rmac features to perform PCA
% withening
vecs_train = cellfun(@(x) vecpostproc(rmac_regionvec(imread(x), net, L)), im_fn_test, 'un', 0);

% learn PCA
fprintf('Learning PCA-whitening\n');
[~, eigvec, eigval, Xm] = yael_pca (single(cell2mat(vecs_train')));
% for each training image, there are 20 regional MAC vectors of 256/512
% dimensions

% apply PCA-whitening
fprintf('Applying PCA-whitening\n');
vecs = cellfun(@(x) vecpostproc(apply_whiten (x, Xm, eigvec, eigval)), vecs, 'un', 0);% Step 3 and 4

% R-MAC: PCA-whitening is perform on region vectors, then they are aggregated
vecs = cellfun(@(x) vecpostproc(sum(x, 2)), vecs, 'un', 0);% Step 5 and 6 (seems like a missing of L2 norm)


% process query images
fprintf('Process query images\n');
qimlist = {gnd_train.imlist{gnd_train.qidx}}; % query image list
% cropped query image data, qim only contains the cropped interested
% instance/object
qim = arrayfun(@(x) crop_qim([im_folder_train, qimlist{x}, '.jpg'], gnd_train.gnd(x).bbx), 1:numel(gnd_train.qidx), 'un', 0);

qvecs = cellfun(@(x) vecpostproc(rmac_regionvec(x, net, L)), qim, 'un', 0); % step 1 and 2 (L2 normed)
qvecs_loc = cellfun(@(x) vecpostproc(mac(x, net)), qim, 'un', 0); % mac feature vector for all 55 queries


% apply PCA-whitening on query vectors
qvecs = cellfun(@(x) vecpostproc(apply_whiten (x, Xm, eigvec, eigval)), qvecs, 'un', 0); % step 3 and 4
qvecs = cellfun(@(x) vecpostproc(sum(x, 2)), qvecs, 'un', 0); % step 5 and 6


fprintf('Retrieval\n');

% final database vectors and query vectors
vecs = cell2mat(vecs');
qvecs = cell2mat(qvecs);

% retrieval with inner product
[sim,ranks] = sort(vecs'*qvecs, 'descend');
map = compute_map (ranks, gnd_train.gnd); % can be used in my own code
fprintf('mAP, without re-ranking = %.4f\n', map);

ranks_rerank = ranks;
ranks_rerank_qe = ranks;

% doing reranking
if rerank
	for q = 1:numel(qim)

		qratio = size(qim{q}, 1) / size(qim{q}, 2);
		ids_toplist = ranks(1:rerank, q);
		% this version of the code does not support saving of compressed files
		% but CNN responses are quantized in the same way to reproduce the results of the paper		 
		conv3d_toplist = arrayfun(@(y)floor((15+((conv3d{y}>=128)*128+conv3d{y}.*(conv3d{y}<128)))/16), ids_toplist, 'un', 0); 
        % conv3d_toplist is the compressed version of the top-1000 result
        % from the initial filtering stage, localization and reranking
        % starts here
        
		% perform the localization      
	    bestbox = cellfun(@(x) aml(double(x), int32(10), double(qvecs_loc{q}), qratio, qratio_t, step_box, rf_step, rf_iter), conv3d_toplist, 'un', 0);

	    % get R-MAC from the localized windows
        vecs_bestbox = cellfun(@(x, b) vecpostproc(rmac_regionvec_act(x(b(2):b(3), b(4):b(5), :), L)), conv3d_toplist, bestbox, 'un', 0);
		vecs_bestbox = cellfun(@(y) vecpostproc(sum(vecpostproc(apply_whiten(y, Xm, eigvec, eigval)), 2)), vecs_bestbox, 'un', 0);
	  

	    % re-compute similarity and re-rank
	    scores_rerank = qvecs(:, q)'*cell2mat(vecs_bestbox');
	    [~, idx] = sort(scores_rerank, 'descend');
	    ranks_rerank(1:rerank, q) = ranks_rerank(idx, q);

	    % perform average query expansion
        scores_rerank_qe = mean([cell2mat({vecs_bestbox{idx(1:nqe)}}), qvecs(:, q)]')*cell2mat(vecs_bestbox');
        [~, idx] = sort(scores_rerank_qe, 'descend');
        ranks_rerank_qe(1:rerank, q) = ranks_rerank_qe(idx, q);
	end
end

% mAP computation
map = compute_map (ranks_rerank, gnd_train.gnd);
fprintf('mAP, after re-ranking = %.4f\n', map);
map = compute_map (ranks_rerank_qe, gnd_train.gnd);
fprintf('mAP, after re-ranking and QE = %.4f\n', map);

fprintf('This experiment is trained with Paris6k and evaluate on Oxford5k \n');

save('test_done_file_full_oxford.mat','-v7.3');
fprintf('All file saving is compeleted \n');