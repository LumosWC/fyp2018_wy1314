addpath helpers/
addpath local_data/

load        gnd_oxford5k
dataset		= './Oxford_dataset/oxbuild_images/';


size_bbx = zeros (numel (gnd_oxford5k.qidx),1);
size_img = zeros (numel (gnd_oxford5k.qidx),1);
bbx_ratio = zeros (numel (gnd_oxford5k.qidx),1);

for i = 1:numel (gnd_oxford5k.qidx)
    img_name = strcat(dataset,gnd_oxford5k.imlist {gnd_oxford5k.qidx(i)},'.jpg' );
    img = imread (img_name);

    %figure;
    %imshow (img);
    %hold on;

    gnd_bbx  = gnd_oxford5k.gnd(i).bbx;
    coord_bbx = zeros (1,4);
    coord_bbx (1) = gnd_bbx(1);
    coord_bbx (2) = gnd_bbx(2);
    coord_bbx (3) = gnd_bbx(3) - gnd_bbx(1);
    coord_bbx (4) = gnd_bbx(4) - gnd_bbx(2);
    %rectangle ('Position', coord_bbx  ,'EdgeColor','g','LineWidth',2);
    
    size_bbx (i) = coord_bbx(3) * coord_bbx (4);
    size_img (i) = size (img,1) * size (img,2);
    bbx_ratio (i) = size_bbx(i)/ size_img (i);
end
bbx_ratio_oxford5k = bbx_ratio;

hist_edge = 0:0.1:1;
histogram (bbx_ratio,hist_edge);
oxford5k_distribution = histogram (bbx_ratio,hist_edge);
oxford5k_distribution.BinCounts = oxford5k_distribution.BinCounts/55;
hold on;
xlabel ('Bounding Box Ratio');
ylabel ('Probability Distribution');
title (' [Oxford5k] Bounding Box Ratio Distribution')
hold off;
%savefig ('oxford5k_distribution.fig');

% ----------------------------

load        gnd_paris6k
dataset		= './Paris6k_dataset/paris_images/';


size_bbx = zeros (numel (gnd_paris6k.qidx),1);
size_img = zeros (numel (gnd_paris6k.qidx),1);
bbx_ratio = zeros (numel (gnd_paris6k.qidx),1);

for i = 1:numel (gnd_paris6k.qidx)
    img_name = strcat(dataset,gnd_paris6k.imlist {gnd_paris6k.qidx(i)},'.jpg' );
    img = imread (img_name);

    %figure;
    %imshow (img);
    %hold on;

    gnd_bbx  = gnd_paris6k.gnd(i).bbx;
    coord_bbx = zeros (1,4);
    coord_bbx (1) = gnd_bbx(1);
    coord_bbx (2) = gnd_bbx(2);
    coord_bbx (3) = gnd_bbx(3) - gnd_bbx(1);
    coord_bbx (4) = gnd_bbx(4) - gnd_bbx(2);
    %rectangle ('Position', coord_bbx  ,'EdgeColor','g','LineWidth',2);
    
    size_bbx (i) = coord_bbx(3) * coord_bbx (4);
    size_img (i) = size (img,1) * size (img,2);
    bbx_ratio (i) = size_bbx(i)/ size_img (i);
end
bbx_ratio_paris6k = bbx_ratio;

figure;
hist_edge = 0:0.1:1;
histogram (bbx_ratio,hist_edge);
paris6k_distribution = histogram (bbx_ratio,hist_edge);
paris6k_distribution.BinCounts = paris6k_distribution.BinCounts/55;
hold on;
xlabel ('Bounding Box Ratio');
ylabel ('Probability Distribution');
title ('[Paris6k] Bounding Box Ratio Distribution')
hold off;
%savefig ('paris6k_distribution.fig');


% -----------------------------------------
bbx_ratio_avg = [bbx_ratio_paris6k ; bbx_ratio_oxford5k];

figure;
hist_edge = 0:0.1:1;
histogram (bbx_ratio_avg,hist_edge);
avg_distribution = histogram (bbx_ratio_avg,hist_edge);
avg_distribution.BinCounts = avg_distribution.BinCounts/110;
hold on;
xlabel ('Bounding Box Ratio');
ylabel ('Probability Distribution');
title ('[Average] Bounding Box Ratio Distribution')
hold off;
%savefig ('avg_distribution.fig');

bbx_distribution = avg_distribution.BinCounts;
bbx_distribution;


% ----------------------------

% ratio_edge = 0:5:100;
% ratio_edge = ratio_edge/100;
% ratio_histo = zeros (20,1);
% for j = 1 : 20
%     for i =  1 : numel (bbx_ratio)
%         ratio_tmp = bbx_ratio(i);
%         cond1 = ratio_tmp >ratio_edge(j);
%         cond2 = ratio_tmp <= ratio_edge(j+1);
%             
%         if cond1 && cond2
%             ratio_histo (j) = ratio_histo (j) + 1;
%         end
%     end
% end



        


% gnd.bbs is the corner coordinate

% Size and location of the rectangle, specified as a four-element vector of the form [x y w h]. 
% The x and y elements define the coordinate for the lower left corner of the rectangle. 
% The w and h elements define the dimensions of the rectangle.
