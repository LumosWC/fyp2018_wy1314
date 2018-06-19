addpath helpers/
addpath local_data/

if exist('aml') ~= 3
	mex -compatibleArrayDims aml.c
end

run ./matconvnet-1.0-beta25/matlab/vl_setupnn.m


main_folder = '/Users/wc/Desktop/CBIR CCTV/'; % oxford5k/ and paris6k/ should be in here
dataset			= 'Oxford_dataset/oxbuild_images';     % dataset to evaluate on 
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
load rmac_oxford5k
load conv3d_oxford5k

fprintf('Preparing Preprocessing... \n');
load eigval_paris6k
load eigvec_paris6k
load Xm_paris6k

fprintf ('Extracing Query Images... \n');
load gnd_oxford5k
load bbx_distribution
%bbx_ratio = generate_random_bbx_ratio ('gnd_oxford5k',bbx_distribution);
ratio = 0.5;
bbx_ratio = generate_specified_bbx_ratio ('gnd_oxford5k',ratio);
qimlist = {gnd_oxford5k.imlist{gnd_oxford5k.qidx}}; 
random_bbx = generate_random_bbx (bbx_ratio,qimlist,dataset_folder);
qim = arrayfun(@(x) crop_qim([dataset_folder, qimlist{x}, '.jpg'], random_bbx(x,1:4)), 1:numel(gnd_oxford5k.qidx), 'un', 0);

fprintf ('Preprocessing Query Features... \n');
qvecs_loc = cellfun(@(x) vecpostproc(mac(x, AlexNet)), qim, 'un', 0); % mac feature vector for all 55 queries
qvecs = cellfun(@(x) vecpostproc(rmac_regionvec(x, AlexNet, L)), qim, 'un', 0); % step 1 and 2 (L2 normed)
qvecs = cellfun(@(x) vecpostproc(apply_whiten (x, Xm_paris6k, eigvec_paris6k, eigval_paris6k)), qvecs, 'un', 0); % step 3 and 4
qvecs = cellfun(@(x) vecpostproc(sum(x, 2)), qvecs, 'un', 0); % step 5 and 6

fprintf('Initial Filtering... \n');
qvecs = cell2mat(qvecs);
% retrieval with inner product
[sim,ranks] = sort(rmac_oxford5k'*qvecs, 'descend');
map = compute_map (ranks, gnd_oxford5k.gnd); % can be used in my own code
fprintf('mAP, without re-ranking = %.4f\n', map);

ranks_rerank = ranks;

conv3d = conv3d_oxford5k;
Xm = Xm_paris6k;
eigvec = eigvec_paris6k ;
eigval = eigval_paris6k;
gnd_test = gnd_oxford5k;
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
        fprintf ('Reranking Image %.1f takes %.4f Second -- Experiment3 Oxford5k， ratio = %.4f \n',q,toc,ratio);
        % --------------------------------------------------
	    % re-compute similarity and re-rank
        scores_rerank = vecs_rerank_4096' * qvec_4096;
	    [~, idx] = sort(scores_rerank, 'descend');
	    ranks_rerank(1:rerank, q) = ranks_rerank(idx, q);
	end
end
map = compute_map (ranks, gnd_oxford5k.gnd); 
fprintf('mAP, without fc AML re-ranking = %.4f -- Oxford5k\n', map);

map = compute_map (ranks_rerank, gnd_oxford5k.gnd); 
fprintf('mAP, with fc AML re-ranking = %.4f -- Oxford5k\n', map);

% ----------------------------------------------------------
% ranks  = ranks_rerank;
% query = floor ( rand * numel ( gnd_oxford5k.qidx ));
% figure;
% im_index =  gnd_oxford5k.qidx(query);
% image = imread ( strcat(dataset_folder,gnd_oxford5k.imlist{im_index} ,'.jpg'));
% imshow(image);
% ok_number = size ( gnd_oxford5k.gnd(query).ok,2)
% junk_number = size ( gnd_oxford5k.gnd(query).junk,2)
% % bounding_box_tmp  = random_bbx(query,:);
% % bounding_box = zeros (1,4);
% % bounding_box (1) = bounding_box_tmp(1);
% % bounding_box (2) = bounding_box_tmp(2);
% % bounding_box (3) = bounding_box_tmp(3) - bounding_box_tmp(1);
% % bounding_box (4) = bounding_box_tmp(4) - bounding_box_tmp(2);
% % 
% % rectangle ('Position',bounding_box  ,'EdgeColor','g','LineWidth',2);
% % title (strcat('query',string(query)));
% % xlabel (strcat ('crop ratio','   ',string (bbx_ratio(query))));
% 
% figure;
% for i = 1 : 20
%     [image , map ] = imread(strcat(dataset_folder,gnd_oxford5k.imlist{ranks(i,query)},'.jpg'));
%     subplot(4,5,i), imshow(image ,map) 
%     
%     if ismember   (ranks(i,query),gnd_oxford5k.gnd(query).ok )
%         rectangle ('Position', [1 1 size(image,2)-1 size(image,1)-1] ,'EdgeColor','g','LineWidth',2);
%     elseif ismember   (ranks(i,query),gnd_oxford5k.gnd(query).junk )
%         rectangle ('Position', [1 1 size(image,2)-1 size(image,1)-1] ,'EdgeColor','b','LineWidth',2);
%     else
%         rectangle ('Position', [1 1 size(image,2)-1 size(image,1)-1] ,'EdgeColor','r','LineWidth',2);  
%     end 
% end