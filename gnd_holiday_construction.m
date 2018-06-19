% construct a gnd_holiday file
path_imgDB = './Holiday_dataset/';
addpath(path_imgDB);

imgFiles = dir(path_imgDB);
imgNamList = {char ( imgFiles(~[imgFiles.isdir]).name)};
imlist = imgNamList';
imlist = imlist (3:end);

imlist_tmp = imlist;
% imlist is a (data size x 1) cell
for i = 1 : numel (imlist)
    imlist_tmp{i} = sscanf (imlist{i},'%i');
    
    tmp = sscanf (imlist{i},'%i');
    tmp = char (string(tmp));
    imlist{i} = tmp;
end

imlist_mat = cell2mat (imlist_tmp);

qname = 100000:100:149900;

% qidx is a (query number x 1) double
qidx = zeros(numel (qname),1);
for i = 1:numel (qname)
    qidx(i) = find (imlist_mat == qname(i));
end

% ok is a (1 x corresponding) double for each image
total = 1:numel (imlist_tmp);
total = total';

ok = zeros(500,1);

i = 1;
tmp_ok = (find (total == qidx (i)) ) : (find (total == qidx (i + 1)) - 1);
tmp_ok = tmp_ok';
gnd.ok = tmp_ok;
for i = 2: numel (qidx)
    if i < 500
        tmp_ok = (find (total == qidx (i)) ) : (find (total == qidx (i + 1)) - 1);
    else
        tmp_ok = (find (total == qidx (i)) ) : 1491;
    end
    tmp_ok = tmp_ok';

    gnd(i).ok = tmp_ok;
end
  
gnd_holiday.gnd = gnd';
gnd_holiday.qidx = qidx;
gnd_holiday.imlist = imlist;
