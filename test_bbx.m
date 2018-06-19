for i = 1: numel (gnd_paris6k.imlist)
    i
    size_feature (i,:) = size (conv3d{i});
    size_image (i,:) = size (imread(strcat(dataset_folder,gnd_paris6k.imlist{i},'.jpg')) );
end

size_feature (1,1) / size_image (1,1) 
size_feature (1,2) / size_image (1,2) 

size_feature (100,1) / size_image (100,1) 
size_feature (100,2) / size_image (100,2)
