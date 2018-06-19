addpath Market1501_dataset/gt_query/
addpath local_data/
load gnd_paris6k.mat
% gnd_market101_construction
cd Market1501_dataset/bounding_box_test/
tmp_imlist = dir('*.jpg'); 
cd ../..
% ---------------------------------- imlist ------------------------------------------
for i = 1 : numel (tmp_imlist)
    imlist{i} = strrep ( tmp_imlist(i).name,'.jpg','');
end
imlist = imlist';
clear tmp_imlist;

cd Market1501_dataset/query/
tmp_imlist = dir('*.jpg'); 
cd ../..
for i = 1 : numel (tmp_imlist)
    query_imlist{i} = strrep ( tmp_imlist(i).name,'.jpg','');
end
query_imlist = query_imlist';
clear tmp_imlist;

cd Market1501_dataset/gt_query//
tmp_gnd_list = dir('*.mat'); 
cd ../..
for i = 1 : numel (tmp_gnd_list)
    if mod(i,2) == 1 % even -> good_index
        good_list{i} = strrep ( tmp_gnd_list(i).name,'.mat','');
    else  % odd -> junk_index
        junk_list{i} = strrep ( tmp_gnd_list(i).name,'.mat','');
    end
end

for i = 1 : numel (query_imlist)
    A{i} = good_list{1+(i-1)*2};
end
A = A';
good_list = A;
for i = 1 : numel (query_imlist)
    B{i} = junk_list {2+(i-1)*2};
end
B = B';
junk_list = B;
clear A;
clear B;
clear tmp_gnd_list;
% ---------------------------------- gnd ------------------------------------------

for i = 1:numel (query_imlist)
    load (good_list{i});
    load (junk_list{i});
    tmp_ok = good_index;
    tmp_junk = junk_index;
    gnd(i).ok = good_index;
    gnd(i).junk = junk_index;
end

gnd_market1501.gnd = gnd';
gnd_market1501.imlist = imlist;

% ---------------------------------- qidx ------------------------------------------

% for i = 1 : 25
%     tmp = strcat(test_name (good_index(i)),'.jpg');
%     tmp = tmp{1};
%     [image , map ] = imread (tmp);
%     subplot(5,5,i), imshow(image ,map) 
% end
% figure;
% for i = 1 : 25
%     tmp = strcat(test_name (good_index(i+25)),'.jpg');
%     tmp = tmp{1};
%     [image , map ] = imread (tmp);
%     subplot(5,5,i), imshow(image ,map) 
% end
% figure;
% for i = 1 : 5
%     tmp = strcat(test_name (junk_index(i)),'.jpg');
%     tmp = tmp{1};
%     [image , map ] = imread (tmp);
%     subplot(2,3,i), imshow(image ,map) 
% end