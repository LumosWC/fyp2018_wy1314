function [qvec] = encode_4096_feature (image,net)
image = single(image) ; % cropped groundtruth version
image = imresize(image, net.meta.normalization.imageSize(1:2)) ; 
image = image - net.meta.normalization.averageImage ;
res = vl_simplenn(net, image) ;
    
feat = res(20).x; %the output of layer 20th in vgg16, which is the 2nd fc layer with 4096 neurons
feat = feat(:);
feat = feat./norm(feat);
    
qvec = feat;