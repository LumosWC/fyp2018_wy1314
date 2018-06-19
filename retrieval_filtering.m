function [ranks,querytime] = retrieval_filtering (image,net,Xm, eigvec, eigval,vecs,gnd_test,show_img)
% ---------------------------- Query: Crop the Specified Object ---------------------------------%
%image = 'paris_defense_000605.jpg';
figure;
hold on;
imshow(image); 
rect = getrect;
object = imcrop(imread(image), rect);
imshow (object);
hold off;

tic;
% ---------------------------- Query: Encode the Query ---------------------------------%
qvec = rmac_regionvec(object, net, 3); % get all regional-MAC features
qvec = vecpostproc (qvec); % L2 Normalisation
qvec = apply_whiten (qvec,Xm, eigvec, eigval); % PCA-Whitening
qvec = vecpostproc (qvec); % L2 Normalisation
qvec = sum(qvec, 2); % Sum
qvec = vecpostproc (qvec); % L2 Normalisation

%qvec_loc = mac(object, net); % keep a single MAC feature (for localization&reranking)
%qvec_loc = vecpostproc (qvec_loc); % L2 Normalisation

% ---------------------------- Retrieval 1 : Initial Filtering ---------------------------------%
[~,ranks] = sort(vecs'*qvec, 'descend');

querytime = toc;

if show_img == 1
    figure;
    for i = 1 : 20
        [image , map ] = imread (strcat ('/Users/wc/Desktop/CBIR CCTV/Paris6k_dataset/paris_images/',char (gnd_test.imlist(ranks(i))),'.jpg'));
        subplot(4,5,i), imshow(image ,map)
    end
end
% imshow ( strcat ( data_folder , dataset_test , '/', char ( gnd_test.imlist(ranks(1)) ),'.jpg' ));