% note: this script has only been verified to run on
% a) thun.ee.ethz.ch
% b) your account
% c) matlab 2015a
% d) cuda toolkit 6.5


% to profile:
% 0) add exit at end of matlab script
% 1) set executable to /usr/pack/matlab-8.5r2015a-fg/bin/matlab
% 2) set working dir to your matlab working dir
% 3) arguments -nojvm -nosplash -r name_of_the_script_without_m


%% compile
mex -v -largeArrayDims -lstdc++ -lc fmean.cu


IGpu = gpuArray(single(rand(240,320)));
tic;
ICumY = cumsum(IGpu, 1);
ICumYX = cumsum(ICumY, 2);
toc;
addpath('../');
IBox = boxfilter(double(gather(IGpu)), 2);


tic;
OGpu = permute(fmean(permute(IGpu, [2 1 3])), [2 1 3]);
toc;
sum(sum(ICumYX - OGpu))/(sum(size(IGpu)))