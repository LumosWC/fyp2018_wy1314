function [bbx_ratio] = generate_specified_bbx_ratio (gnd,mean)

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


bbx_ratio  = zeros (1, numel (gnd_data.qidx));

for i = 1: numel (gnd_data.qidx)
    bbx_ratio(i) = max(0.01,randn + mean*100);
end

bbx_ratio = bbx_ratio'/100;