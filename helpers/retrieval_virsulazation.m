function retrieval_virsulazation(queryID, numRetrieval, featNorm,imgNamList)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


% virsulazation

QueryVec = featNorm(queryID, :);
n = size(featNorm);
n = n(1);
score = zeros(n, 1);

for loop = 1:n
    VecTemp = featNorm(loop, :);
    score(loop) = QueryVec*VecTemp';
end

score = (QueryVec*featNorm')';

[~, index] = sort(score, 'descend');
rank_image_ID = index(1:numRetrieval,1);
retre_image_name = imgNamList(rank_image_ID);


for i = 1:numRetrieval
    I = imread(char(retre_image_name(i)));
    figure(i);
    imshow(I);
end


%{
I2 = uint8(zeros(100, 100, 3, numRetrieval)); % 32 and 32 are the size of the output image
for i=1:numRetrieval
    imName = rgbImgList{rank_image_ID(i, 1), 1};
    im = imread(imName);
    im = imresize(im, [100 100]);
    if (ndims(im)~=3)
        I2(:, :, 1, i) = im;
        I2(:, :, 2, i) = im;
        I2(:, :, 3, i) = im;
    else
        I2(:, :, :, i) = im;
    end
end

figure('color',[1,1,1]);
montage(I2(:, :, :, (1:numRetrieval)));
title('search result');

QueryName = rgbImgList{queryID, 1};
im = imread(QueryName);
imQuery = imresize(im, [100 100]);
figure('color',[1,1,1]);
imshow(imQuery);
title('query image');


end
%}