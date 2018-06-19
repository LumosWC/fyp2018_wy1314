addpath helpers/
addpath local_data/

if exist('aml') ~= 3
	mex -compatibleArrayDims aml.c
end

run ./matconvnet-1.0-beta25/matlab/vl_setupnn.m


main_folder = '/Users/wc/Desktop/CBIR CCTV/'; % holiday/ and paris6k/ should be in here
dataset			= 'Holiday_dataset';     % dataset to evaluate on 
dataset_folder = [main_folder, dataset, '/'];

% --------------------------- parameters of the method ---------------------------
rerank 					= 1000;  %1000	% number of images to re-rank, no re-ranking if 0
L 						= 3;  % number of levels in the region pyramid of R-MAC

step_box 				= 3;		% parameter t in the paper
qratio_t 				= 1.1;   % parameter s in the paper
rf_step 				= 3;		% fixed step for refinement
rf_iter 				= 5;		% number of iterations of refinement
nqe 					= 5;		
% --------------------------------------------------------------------------------
fprintf('Loading Neural Networks... \n');
load AlexNet
load VGG16Net

fprintf('Preparing Database... \n');
load rmac_holiday
load conv3d_holiday

fprintf('Preparing Preprocessing... \n');
load eigval_paris6k
load eigvec_paris6k
load Xm_paris6k

fprintf ('Extracing Query Images... \n');
load gnd_holiday
load bbx_distribution
bbx_ratio = generate_random_bbx_ratio ('gnd_holiday',bbx_distribution);
%bbx_ratio = generate_specified_bbx_ratio ('gnd_holiday',0.05);
qimlist = {gnd_holiday.imlist{gnd_holiday.qidx}}; 
random_bbx = generate_random_bbx (bbx_ratio,qimlist,dataset_folder);
qim = arrayfun(@(x) crop_qim([dataset_folder, qimlist{x}, '.jpg'], random_bbx(x,1:4)), 1:numel(gnd_holiday.qidx), 'un', 0);

fprintf ('Preprocessing Query Features... \n');
qvecs_loc = cellfun(@(x) vecpostproc(mac(x, AlexNet)), qim, 'un', 0); % mac feature vector for all 55 queries
qvecs = cellfun(@(x) vecpostproc(rmac_regionvec(x, AlexNet, L)), qim, 'un', 0); % step 1 and 2 (L2 normed)
qvecs = cellfun(@(x) vecpostproc(apply_whiten (x, Xm_paris6k, eigvec_paris6k, eigval_paris6k)), qvecs, 'un', 0); % step 3 and 4
qvecs = cellfun(@(x) vecpostproc(sum(x, 2)), qvecs, 'un', 0); % step 5 and 6

fprintf('Initial Filtering... \n');
qvecs = cell2mat(qvecs);
% retrieval with inner product
[sim,ranks] = sort(rmac_holiday'*qvecs, 'descend');
map = compute_map (ranks, gnd_holiday.gnd); % can be used in my own code
fprintf('mAP, without re-ranking = %.4f\n', map);

ranks_rerank = ranks;

conv3d = conv3d_holiday;
Xm = Xm_paris6k;
eigvec = eigvec_paris6k ;
eigval = eigval_paris6k;
gnd_test = gnd_holiday;
if rerank
    load VGG16Net
	for q = 1:numel(qim)
		qratio = size(qim{q}, 1) / size(qim{q}, 2);
		ids_toplist = ranks(1:rerank, q); 
		conv3d_toplist = arrayfun(@(y)floor((15+((conv3d{y}>=128)*128+conv3d{y}.*(conv3d{y}<128)))/16), ids_toplist, 'un', 0); 
		% perform the AML      
	    bestbox = cellfun(@(x) aml(double(x), int32(10), double(qvecs_loc{q}), qratio, qratio_t, step_box, rf_step, rf_iter), conv3d_toplist, 'un', 0);
        
        % ---------------- Method 5 -----------------
        qvec_4096 = encode_4096_feature (qim{q},VGG16Net);% calculated the fc feature for query
        vecs_rerank_4096 = zeros (4096,rerank);
        tic;
        for i = 1:rerank
            image_tmp = imread (strcat (dataset_folder,char (gnd_test.imlist(ids_toplist(i))),'.jpg'));
            conv3d_tmp = conv3d_toplist{i};
            bestbox_tmp = bestbox{i};
            [bbx,im_crop] = feature2image_coordinate (image_tmp,conv3d_tmp,bestbox_tmp);% reflecting feature coord back to image coord
            vecs_rerank_4096 (:,i) = encode_4096_feature (im_crop,VGG16Net);
        end
        fprintf ('Reranking Image %.1f takes %.4f Second\n',q,toc);
        % --------------------------------------------------
	    % re-compute similarity and re-rank
        scores_rerank = vecs_rerank_4096' * qvec_4096;
	    [~, idx] = sort(scores_rerank, 'descend');
	    ranks_rerank(1:rerank, q) = ranks_rerank(idx, q);
	end
end
map = compute_map (ranks_rerank, gnd_holiday.gnd); 
fprintf('mAP, with fc AML re-ranking = %.4f\n', map);
