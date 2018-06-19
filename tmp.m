qvecs = cellfun(@(x) vecpostproc(apply_whiten (x, Xm_paris6k, eigvec_paris6k, eigval_paris6k)), qvecs, 'un', 0); % step 3 and 4
qvecs = cellfun(@(x) vecpostproc(sum(x, 2)), qvecs, 'un', 0); % step 5 and 6
%qvecs = rmac_holiday (:,gnd_holiday.qidx);

fprintf('Initial Filtering... \n');
[sim,ranks] = sort(rmac_holiday'*qvecs, 'descend');
map = compute_map (ranks, gnd_holiday.gnd);
fprintf('mAP, without re-ranking = %.4f\n', map);