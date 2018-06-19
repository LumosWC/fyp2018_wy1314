% holiday_feature_extraction
addpath helpers/
addpath local_data/

if exist('aml') ~= 3
	mex -compatibleArrayDims aml.c
end

run ./matconvnet-1.0-beta25/matlab/vl_setupnn.m

main_folder = '/Users/wc/Desktop/CBIR CCTV/'; % oxford5k/ and paris6k/ should be in here
dataset			= 'Holiday_dataset';     % dataset to evaluate on 
dataset_folder = [main_folder, dataset, '/'];

% --------------------------- parameters of the method ---------------------------
L 						= 3;  % number of levels in the region pyramid of R-MAC
% --------------------------------------------------------------------------------

fprintf('Loading Neural Networks... \n');
load AlexNet
load gnd_holiday

fprintf('Preparing Preprocessing... \n'); % pre-trained on Paris6k
load eigval_paris6k
load eigvec_paris6k
load Xm_paris6k

fprintf('Preparing Database... \n');
im_fn_test = cellfun(@(x) [dataset_folder, x, '.jpg'], gnd_holiday.imlist, 'un', 0);
[vecs, conv3d] = cellfun(@(x) rmac_regionvec(imread(x), AlexNet, L), im_fn_test, 'un', 0); % Step 1

im = cellfun(@(x) (imread(x)), im_fn_test, 'un', 0); 
im = im';
vecs = cellfun(@(x) vecpostproc(rmac_regionvec(x, AlexNet, L)), im, 'un', 0); % step 1 and 2 (L2 normed)
vecs = cellfun(@(x) vecpostproc(apply_whiten (x, Xm_paris6k, eigvec_paris6k, eigval_paris6k)), vecs, 'un', 0);% Step 3 and 4
vecs = cellfun(@(x) vecpostproc(sum(x, 2)), vecs, 'un', 0);
vecs = cell2mat(vecs);

fprintf ('Extracing Query Images... \n');
load gnd_holiday
load bbx_distribution
bbx_ratio = generate_random_bbx_ratio ('gnd_holiday',bbx_distribution);
qimlist = {gnd_holiday.imlist{gnd_holiday.qidx}}; 
random_bbx = generate_random_bbx (bbx_ratio,qimlist,dataset_folder);
qim = arrayfun(@(x) crop_qim([dataset_folder, qimlist{x}, '.jpg'], random_bbx(x,1:4)), 1:numel(gnd_holiday.qidx), 'un', 0);

fprintf ('Preprocessing Query Features... \n');
qvecs = cellfun(@(x) vecpostproc(rmac_regionvec(x, AlexNet, L)), qim, 'un', 0); % step 1 and 2 (L2 normed)
qvecs = cellfun(@(x) vecpostproc(apply_whiten (x, Xm_paris6k, eigvec_paris6k, eigval_paris6k)), qvecs, 'un', 0); % step 3 and 4
qvecs = cellfun(@(x) vecpostproc(sum(x, 2)), qvecs, 'un', 0); % step 5 and 6
qvecs = cell2mat(qvecs);

fprintf('Initial Filtering... \n');
% retrieval with inner product
[sim,ranks] = sort(vecs'*qvecs, 'descend');
map = compute_map (ranks, gnd_holiday.gnd); % can be used in my own code
fprintf('mAP (rmac_holiday), without re-ranking = %.4f\n', map);