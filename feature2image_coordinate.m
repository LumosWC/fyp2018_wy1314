function [bbx,im_crop] = feature2image_coordinate (image_tmp,conv3d_tmp,bestbox_tmp)

%i = 70;
%image_tmp = imread (strcat ('/Users/wc/Desktop/CBIR CCTV/Paris6k_dataset/paris_images/',char (gnd_test.imlist(ids_toplist(i))),'.jpg'));
%conv3d_tmp = conv3d_toplist{i};
%bestbox_tmp = bestbox{i};



% ---------------Reflecting the bestbox in Feature Space------------------
bbx_feature =bestbox_tmp(2:5);

w_ratio = size (image_tmp,2)/size(conv3d_tmp,2);
h_ratio = size (image_tmp,1)/size(conv3d_tmp,1);

bbx (1) = (bbx_feature (1) - 1) * w_ratio;
bbx (2) = (bbx_feature (2) + 1) * h_ratio;
bbx (3) = (bbx_feature (3) - 1) * w_ratio;% - bbx (1) ;
bbx (4) = (bbx_feature (4) + 1) * h_ratio;% - bbx (2) ;

%figure;
bbx = round (bbx);
bbx(1) = max (1,bbx(1));
bbx(2) = min (size(image_tmp,1),bbx(2));
bbx(3) = max (1,bbx(3));
bbx(4) = min (size(image_tmp,2),bbx(4));

if bbx(2) <=  bbx(1)
    bbx(2) = bbx(1) + 1;
end
if bbx(4) <= bbx(3)
    bbx(4) = bbx(3) + 1;
end

im_crop = image_tmp(bbx(1):bbx(2), bbx(3):bbx(4), :);

%imshow (im_crop)
