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

rerank 					= 20;  %50	% number of images to re-rank, no re-ranking if 0

step_box 				= 3;		% parameter t in the paper
qratio_t 				= 1.1;   % parameter s in the paper
rf_step 				= 3;		% fixed step for refinement
rf_iter 				= 5;		% number of iterations of refinement
nqe 					= 5;		% number of images to be used for QE
% --------------------------------------------------------------------------------
fprintf('Loading Neural Networks... \n');
load AlexNet

fprintf('Preparing Database... \n');
load rmac_holiday
load conv3d_holiday

fprintf('Preparing Preprocessing... \n');
load eigval_paris6k
load eigvec_paris6k
load Xm_paris6k

fprintf ('Extracing Query Images... \n');
load gnd_holiday
%load bbx_distribution
%bbx_ratio = generate_random_bbx_ratio ('gnd_holiday',bbx_distribution);
bbx_ratio = generate_specified_bbx_ratio ('gnd_holiday',0.05);
qimlist = {gnd_holiday.imlist{gnd_holiday.qidx}}; 
random_bbx = generate_random_bbx (bbx_ratio,qimlist,dataset_folder);

qim = arrayfun(@(x) crop_qim([dataset_folder, qimlist{x}, '.jpg'], random_bbx(x,1:4)), 1:numel(gnd_holiday.qidx), 'un', 0);

fprintf ('Preprocessing Query Features... \n');
qvecs_loc = cellfun(@(x) vecpostproc(mac(x, AlexNet)), qim, 'un', 0); % mac feature vector for all 500 queries
qvecs = cellfun(@(x) vecpostproc(rmac_regionvec(x, AlexNet, L)), qim, 'un', 0); % step 1 and 2 (L2 normed)
qvecs = cellfun(@(x) vecpostproc(apply_whiten (x, Xm_paris6k, eigvec_paris6k, eigval_paris6k)), qvecs, 'un', 0); % step 3 and 4
qvecs = cellfun(@(x) vecpostproc(sum(x, 2)), qvecs, 'un', 0); % step 5 and 6
qvecs = cell2mat(qvecs);

fprintf('Initial Filtering... \n');
[sim,ranks] = sort(rmac_holiday'*qvecs, 'descend');
map = compute_map (ranks, gnd_holiday.gnd); % can be used in my own code
fprintf('mAP (rmac_holiday), without re-ranking = %.4f\n', map);


% ----------------------------------------------------------

ranks_rerank = ranks;
ranks_rerank_qe = ranks;
conv3d = conv3d_holiday;
% doing reranking
if rerank
	for q = 1:numel(qim)
        fprintf ('Reranking the %.0i th \n',q);
		qratio = size(qim{q}, 1) / size(qim{q}, 2);
		ids_toplist = ranks(1:rerank, q);	 
		conv3d_toplist = arrayfun(@(y)floor((15+((conv3d{y}>=128)*128+conv3d{y}.*(conv3d{y}<128)))/16), ids_toplist, 'un', 0); 

	    bestbox = cellfun(@(x) aml(double(x), int32(10), double(qvecs_loc{q}), qratio, qratio_t, step_box, rf_step, rf_iter), conv3d_toplist, 'un', 0);

	    % get R-MAC from the localized windows
        vecs_bestbox = cellfun(@(x, b) vecpostproc(rmac_regionvec_act(x(b(2):b(3), b(4):b(5), :), L)), conv3d_toplist, bestbox, 'un', 0);
		vecs_bestbox = cellfun(@(y) vecpostproc(sum(vecpostproc(apply_whiten(y, Xm_paris6k, eigvec_paris6k, eigval_paris6k)), 2)), vecs_bestbox, 'un', 0);
	  

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

map = compute_map (ranks, gnd_holiday.gnd); % can be used in my own code
fprintf('mAP (rmac_holiday - Pure), without re-ranking = %.4f\n', map);

map = compute_map (ranks_rerank, gnd_holiday.gnd); % can be used in my own code
fprintf('mAP (rmac_holiday - AML) = %.4f\n', map);

map = compute_map (ranks_rerank_qe, gnd_holiday.gnd); % can be used in my own code
fprintf('mAP (rmac_holiday - AML + QE) = %.4f\n', map);
% ----------------------------------------------------------
ranks = ranks_rerank_qe;

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