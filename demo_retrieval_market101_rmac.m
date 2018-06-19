% % demo_retrieval_market101
 addpath helpers/
 addpath local_data/
 addpath matconvnet-1.0-beta25/
 run ./matconvnet-1.0-beta25/matlab/vl_setupnn
% 
% main_folder    = '/Users/wc/Desktop/CBIR CCTV/';    
% dataset_train = 'Market1501_dataset/bounding_box_train'; 
% dataset_train_folder = [main_folder, dataset_train, '/'];
% 
% dataset_test        = 'Market1501_dataset/bounding_box_test';     % dataset to evaluate on 
% dataset_test_folder = [main_folder, dataset_test, '/'];
% 
% dataset_query =  'Market1501_dataset/query'; 
% dataset_query_folder = [ main_folder , dataset_query, '/'];
% 
% 
% load AlexNet.mat
% net = AlexNet;
% L = 3;
% 
% cd Market1501_dataset/bounding_box_train
% train_image  = dir('*.jpg');  
% cd ../..
% 
% cd Market1501_dataset/bounding_box_test
% test_image  = dir('*.jpg');  
% cd ../..
% 
% cd Market1501_dataset/query
% query_image  = dir('*.jpg');  
% cd ../..
% 
% for i = 1:  numel (train_image) 
%     fprintf_r('Extracting %.01f th Image from Training Data',i);
%     im_train_resize{i} = imresize ( imread (strcat(dataset_train_folder,train_image(i).name)) , [224,224]);
% end
% 
% for i = 1:  numel (test_image) 
%     fprintf_r('Extracting %.01f th Image from Testing Data',i);
%     im_test_resize{i} = imresize ( imread (strcat(dataset_test_folder,test_image(i).name)) , [224,224]);
% end
% 
% for i = 1 : numel (query_image)
%     fprintf_r('Extracting %.01f th Image from Query Data',i);
%     im_query_resize{i} = imresize ( imread (strcat(dataset_query_folder,query_image(i).name)) , [224,224]);
% end
% im_train_resize = im_train_resize';
% im_test_resize = im_test_resize';
% im_query_resize = im_query_resize';
%     
% fprintf('Learning PCA Withening from Training Data ...\n');
% load eigval_market101
% load eigvec_market101
% load Xm_market101
% eigval = eigval_market101;
% eigvec = eigvec_market101;
% Xm = Xm_market101;
% % vecs_train = cellfun(@(x) vecpostproc(rmac_regionvec(x, net, L)), im_train_resize, 'un', 0);
% % [~, eigvec, eigval, Xm] = yael_pca (single(cell2mat(vecs_train')));
% 
% fprintf('Extracting Testing Database Feature ...\n');
% load rmac_market101
% vecs = rmac_market101;
% % [vecs, ~] = cellfun(@(x) rmac_regionvec(x, net, L), im_test_resize, 'un', 0); % Step 1
% % [vecs] = cellfun(@(x) vecpostproc(x), vecs, 'un', 0);% Step 2
% % vecs = cellfun(@(x) vecpostproc(apply_whiten (x, Xm, eigvec, eigval)), vecs, 'un', 0);% Step 3 and 4
% % vecs = cellfun(@(x) vecpostproc(sum(x, 2)), vecs, 'un', 0);% Step 5 and 6
% % vecs = cell2mat(vecs');
% 
% fprintf('Encoding the Query Image ...\n');
% [qvec, ~] = cellfun(@(x) rmac_regionvec(x, net, L), im_query_resize, 'un', 0); % Step 1
% [qvec] = cellfun(@(x) vecpostproc(x), qvec, 'un', 0);% Step 2
% qvec = cellfun(@(x) vecpostproc(apply_whiten (x, Xm, eigvec, eigval)), qvec, 'un', 0);% Step 3 and 4
% qvec = cellfun(@(x) vecpostproc(sum(x, 2)), qvec, 'un', 0);% Step 5 and 6
% qvec = cell2mat(qvec');
% % query_folder = [main_folder, 'Market1501_dataset/query/'];
% % person = '0004_c1s6_016996_00.jpg';
% % person = imread ([query_folder,person]);
% % person_resize = imresize (person,[224,224]);
% % 
% % qvec = rmac_regionvec(person_resize, net, L); % get all regional-MAC features
% % qvec = vecpostproc (qvec); % L2 Normalisation
% % qvec = apply_whiten (qvec,Xm, eigvec, eigval); % PCA-Whitening
% % qvec = vecpostproc (qvec); % L2 Normalisation
% % qvec = sum(qvec, 2); % Sum
% % qvec = vecpostproc (qvec); % L2 Normalisation
% 
tic;
load ('demostration_market1501_rmac.mat');
time = toc;

fprintf('Retrieving the Query ...\n');
load gnd_market1501
[distance,ranks] = sort(vecs'*qvec, 'descend');
map = compute_map (ranks, gnd_market1501.gnd);
fprintf('mAP on Market1501, without re-ranking = %.4f\n', map);

fprintf('Loading all these data took %.4f Seconds ...',time);

query = floor ( rand * numel ( query_image ));
%query = 7;
figure;
imshow(im_query_resize{query});
title (strcat('query',string(query)));

figure;
for i = 1 : 20
    [image , map ] = imread(strcat(dataset_test,'/',test_image(ranks(i,query)).name));
    subplot(4,5,i), imshow(image ,map) 
    
    if ismember   (ranks(i,query),gnd_market1501.gnd(query).ok )
        rectangle ('Position', [1 1 63 127] ,'EdgeColor','g','LineWidth',2);
    elseif ismember (ranks(i,query),gnd_market1501.gnd(query).junk )
        rectangle ('Position', [1 1 63 127] ,'EdgeColor','b','LineWidth',2);
    else
        rectangle ('Position', [1 1 63 127] ,'EdgeColor','r','LineWidth',2);  
    end 
end
 