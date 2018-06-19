% % demo_retrieval_market101
 addpath helpers/
 addpath local_data/
 addpath matconvnet-1.0-beta25/
 run ./matconvnet-1.0-beta25/matlab/vl_setupnn

main_folder    = '/Users/wc/Desktop/CBIR CCTV/';    
dataset_train = 'Market1501_dataset/bounding_box_train'; 
dataset_train_folder = [main_folder, dataset_train, '/'];

dataset_test        = 'Market1501_dataset/bounding_box_test';     % dataset to evaluate on 
dataset_test_folder = [main_folder, dataset_test, '/'];

dataset_query =  'Market1501_dataset/query'; 
dataset_query_folder = [ main_folder , dataset_query, '/'];

load VGG16Net.mat

cd Market1501_dataset/bounding_box_train
train_image  = dir('*.jpg');  
cd ../..

cd Market1501_dataset/bounding_box_test
test_image  = dir('*.jpg');  
cd ../..

cd Market1501_dataset/query
query_image  = dir('*.jpg');  
cd ../..

% for i = 1:  numel (train_image) 
%     fprintf_r('Extracting %.01f th Image from Training Data',i);
%     im_train_resize{i} = imresize ( imread (strcat(dataset_train_folder,train_image(i).name)) , [224,224]);
% end

for i = 1:  numel (test_image) 
    fprintf_r('Extracting %.01f th Image from Testing Data',i);
    im_test_resize{i} = imresize ( imread (strcat(dataset_test_folder,test_image(i).name)) , [224,224]);
end

for i = 1 : numel (query_image)
    fprintf_r('Extracting %.01f th Image from Query Data',i);
    im_query_resize{i} = imresize ( imread (strcat(dataset_query_folder,query_image(i).name)) , [224,224]);
end
%im_train_resize = im_train_resize';
im_test_resize  = im_test_resize';
im_query_resize = im_query_resize';

fprintf ('Encoding testing dataset by feeding into AlexNet');
fc4096_market1501 = zeros (4096,numel (im_test_resize));
for i = 1:numel (im_test_resize)
%     nameImg = strcat(dataset_folder,qimlist{i},'.jpg');
%     oriImg = imread(nameImg);
%     image = single(oriImg) ; % note: 255 range

    image = single(im_test_resize{i}) ; % cropped groundtruth version
    image = imresize(image, VGG16Net.meta.normalization.imageSize(1:2)) ; 
    image = image - VGG16Net.meta.normalization.averageImage ;
    res = vl_simplenn(VGG16Net, image) ;
    
    feat = res(20).x; %the output of layer 20th in vgg16, which is the 2nd fc layer with 4096 neurons
    feat = feat(:);
    feat = feat./norm(feat);
    
    fc4096_market1501 (:,i) = feat;
    fprintf('Extracting Testing Image %d ... \n', i);
end

fprintf ('Encoding Query dataset by feeding into AlexNet');
qvecs = zeros (4096,numel (im_query_resize));
for i = 1:numel (im_query_resize)
%     nameImg = strcat(dataset_folder,qimlist{i},'.jpg');
%     oriImg = imread(nameImg);
%     image = single(oriImg) ; % note: 255 range

    image = single(im_query_resize{i}) ; % cropped groundtruth version
    image = imresize(image, VGG16Net.meta.normalization.imageSize(1:2)) ; 
    image = image - VGG16Net.meta.normalization.averageImage ;
    res = vl_simplenn(VGG16Net, image) ;
    
    feat = res(20).x; %the output of layer 20th in vgg16, which is the 2nd fc layer with 4096 neurons
    feat = feat(:);
    feat = feat./norm(feat);
    
    qvecs (:,i) = feat;
    fprintf('Extracting Query Image %d ... \n', i);
end

fprintf('Retrieving the Query ...\n');
load gnd_market1501
[distance,ranks] = sort(fc4096_market1501'*qvecs, 'descend');
map = compute_map (ranks, gnd_market1501.gnd);
fprintf('mAP on Market1501, without re-ranking = %.4f\n', map);

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