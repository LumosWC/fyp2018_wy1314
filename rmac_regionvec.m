%
% Authored by G. Tolias, 2015. 
%
function [vecs, X] = rmac_regionvec(I, net, L)
    tic;
    if size(I,3) == 1
        I = repmat(I, [1 1 3]);
    end
    I = single(I) - mean(net.normalization.averageImage(:));

    if ~isa(net.layers{1}.weights{1}, 'gpuArray')
        rnet = vl_simplenn(net, I);  
        X = max(rnet(end).x, 0);
    else
        rnet = vl_simplenn(net, gpuArray(I));  
        X = gather(max(rnet(end).x, 0)); % max-pooling from the last conv layer
    end
    vecs = rmac_regionvec_act(X, L);
    fprintf('Time for extracting one image CNN activation is %.4f Second !!! \n',toc);