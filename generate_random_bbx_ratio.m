function [bbx_ratio] = generate_random_bbx_ratio (gnd,bbx_distribution)
% Input: gnd file name
% Output: bbx_ratio: the probability distribution each 10th area ratio


addpath local_data/
gnd_data = load (strcat (gnd,'.mat'));
if strcmp (gnd, 'gnd_holiday')
    gnd_data = gnd_data.gnd_holiday;
elseif strcmp ( gnd , 'gnd_oxford5k')
    gnd_data = gnd_data.gnd_oxford5k;
elseif  strcmp (gnd , 'gnd_paris6k')
    gnd_data = gnd_data.gnd_paris6k;
else
    fprint('Invadid Dataset')
    return
end


size = 0:10:100;
apperance = round(bbx_distribution * numel (gnd_data.qidx));

bbx_side = zeros (sum(apperance),1);

count = 1;
mid = 5;
start = sum ( apperance(1:count));

for i = 1 : sum(apperance)
    if i == start + 1
        mid = mid + 10;
        count = count + 1;
        start = sum ( apperance(1:count));
    end
        bbx_side (i) = mid + randn;
end

bbx_side_random = zeros (numel (bbx_side),1);
p = randperm (numel (bbx_side));
for i = 1:numel (bbx_side)
    bbx_side_random (i) = bbx_side( p (i) );
end

bbx_ratio = bbx_side_random/100;
