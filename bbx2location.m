function [location] = bbx2location (bbx)
%---------------- transform bbx to bounding_box in matlabs rectangle draw
bounding_box = zeros (1,4);
bounding_box (1) = bbx(3);
bounding_box (2) = bbx(1);
bounding_box (3) = bbx(4) - bbx(3);%width
bounding_box (4) = bbx(2) - bbx(1);% height
location = bounding_box;
