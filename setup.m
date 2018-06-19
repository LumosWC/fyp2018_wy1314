% ---------------------------- Preparation 1 : Set-Up -------------------------------%
addpath('helpers');
if exist('aml') ~= 3
	mex -compatibleArrayDims aml.c
end

cd matconvnet-1.0-beta25/matlab/
vl_setupnn;
cd ../../

main_folder    = '/Users/wc/Desktop/CBIR CCTV/';     % oxford5k/ and paris6k/ should be in here
dataset_train  = 'Oxford_dataset/oxbuild_images';    % dataset to learn the PCA-whitening on
dataset_test   = 'Paris6k_dataset/paris_images';     % dataset to evaluate on 

% ---------------------------- Preparation 2: Data Pre-Processing -------------------------------%
fprintf('Pre-loading database image features... \n');
tic;
load 'preprocessing.mat';
fprintf('Loading features done, it takes %0.4f Second s... \n', toc);
