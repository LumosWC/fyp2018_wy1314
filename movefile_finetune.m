% % demo_retrieval_market101
 addpath helpers/
 addpath local_data/
 addpath matconvnet-1.0-beta25/
 run ./matconvnet-1.0-beta25/matlab/vl_setupnn

main_folder    = '/Users/wc/Desktop/CBIR CCTV/';    
dataset_train = 'Market1501_dataset/bounding_box_train'; 
dataset_train_folder = [main_folder, dataset_train, '/'];

cd Market1501_dataset/bounding_box_train
train_image  = dir('*.jpg');  
cd ../..

for i = 1 : numel (train_image)
    im{i} = char ( train_image(i).name);
    class{i} = [im{i}(1),im{i}(2),im{i}(3),im{i}(4)];
end
im = im';
class = class';
class_unique = unique (class);

for i = 1: numel (class)
   class_int (i) = str2num(class{i});
end
class_int = class_int';

for i = 1: numel (class_unique)
   class_count (i) = sum (class_int == str2num (class_unique{i}));
end
class_count = class_count';

class_count_train = floor(class_count * 0.8);
class_count_val = class_count - class_count_train;

% for i = 1: numel (train_image)
%     i
%     for j = 1: numel (class_unique)
%         class_tmp = str2num(class_unique{j});
%         if str2num ( im{i}(1:4)) == class_tmp 
%             movefile (strcat('Market1501_dataset/bounding_box_train copy/', im{i}),strcat('Market1501_dataset/finetune_data/train/',class_unique{j}));
%         end
%     end
% end


for i = 1 : numel (class_unique) 
    i
     tmp_des = strcat('Market1501_dataset/finetune_data/val/',class_unique{i});
     tmp_folder = strcat('Market1501_dataset/finetune_data/train/',class_unique{i});
     cd (tmp_folder);
     tmp_list =  dir('*.jpg'); 
     cd ../../../..
     tmp_perm = randperm (numel(tmp_list));
     for j = 1 : class_count_val (i)
         movefile (strcat(tmp_folder,'/',tmp_list(tmp_perm(j)).name), tmp_des );
     end
end
         
         
         
