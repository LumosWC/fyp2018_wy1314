
dys =  -pi/2:0.000001:0;
for i = 1 : numel (dys)
    area (i) = 2 * abs(dys(i)) * cos(dys(i));
end
    
max(area)