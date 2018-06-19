load experiment3

figure;
hold on;
ylim([0 0.9]);

plot (ratio,holiday_map_4096,'r','Linewidth',2);
scatter (ratio,holiday_map_4096,'r','^');

plot (ratio,holiday_map_rerank_qe,'m','Linewidth',2);
scatter (ratio,holiday_map_rerank_qe,'m','^');



% plot (ratio,holiday_map_pure,'Linewidth',1);
% plot (ratio,holiday_map_rerank,'Linewidth',1);
% plot (ratio,holiday_map_rerank_qe,'Linewidth',1);
% plot (ratio,holiday_map_5,'Linewidth',1);
title ('Holiday')
legend ('Method 1(fc4096)','','Method 5(rmac+aml+qe)','')
%legend ('Method 1(fc4096)','Method 1(fc128)','Method 2(rmac)','Method 3(rmac+aml)','Method 4(rmac+aml+qe)','Method 5(rmac+aml_{fc})')
xlabel ('Bounding Box Mean Area Ratio')
ylabel ('mAP')
hold off;


figure;
hold on;
ylim([0 0.9]);
%plot (ratio,oxford5k_map_4096,'r','Linewidth',2);
%scatter (ratio,oxford5k_map_4096,'r','^');


plot (ratio,oxford5k_map_4096,'r','Linewidth',2);
scatter (ratio,oxford5k_map_4096,'r','^');

plot (ratio,oxford5k_map_rerank_qe,'m','Linewidth',2);
scatter (ratio,oxford5k_map_rerank_qe,'m','^');

% plot (ratio,oxford5k_map_pure,'Linewidth',1);
% plot (ratio,oxford5k_map_rerank,'Linewidth',1);
% plot (ratio,oxford5k_map_rerank_qe,'Linewidth',1);
% plot (ratio,oxford5k_map_5,'Linewidth',1);
title ('Oxford5k')
legend ('Method 1(fc4096)','','Method 5(rmac+aml+qe)','')
% legend ('Method 1(fc4096)','Method 1(fc128)','Method 2(rmac)','Method 3(rmac+aml)','Method 4(rmac+aml+qe)','Method 5(rmac+aml_{fc})')
xlabel ('Bounding Box Mean Area Ratio')
ylabel ('mAP')
hold off;


figure;
hold on;
ylim([0 0.9]);


%plot (ratio,paris6k_map_128,'r','Linewidth',2);
%scatter (ratio,paris6k_map_128,'r','v');



plot (ratio,paris6k_map_4096,'r','Linewidth',2);
scatter (ratio,paris6k_map_4096,'r','^');

plot (ratio,paris6k_map_rerank_qe,'m','Linewidth',2);
scatter (ratio,paris6k_map_rerank_qe,'m','^');
% plot (ratio,paris6k_map_pure,'Linewidth',1);
% plot (ratio,paris6k_map_rerank,'Linewidth',1);
% plot (ratio,paris6k_map_rerank_qe,'Linewidth',1);
% plot (ratio,paris6k_map_5,'Linewidth',1);
title ('Paris6k')
legend ('Method 1(fc4096)','','Method 5(rmac+aml+qe)','')
% legend ('Method 1(fc4096)','Method 1(fc128)','Method 2(rmac)','Method 3(rmac+aml)','Method 4(rmac+aml+qe)','Method 5(rmac+aml_{fc})')
xlabel ('Bounding Box Mean Area Ratio')
ylabel ('mAP')
hold off;