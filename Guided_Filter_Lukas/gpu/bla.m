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
mex -v -largeArrayDims -lstdc++ -lc mexGPUExample.cu

%% dummy run with timing
%x = ones(4,4,'gpuArray');

gd = gpuDevice(); 

x = single(1:2000000);
xGpu = gpuArray(x);

tic
for i = 1:100
    yGpu = mexGPUExample(xGpu);
end
wait(gd); % synchronize
toc

y = gather(yGpu);
isOk = all(y == 2*x) %#ok<NOPTS>

% 2nd option for even better timing: 
fh = @() mexGPUExample(xGpu);
gputimeit(fh,1) * 100 %#ok<NOPTS> % 2nd arg indicates number of outputs

