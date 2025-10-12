%%% PRMask_FromONNX_MLP - Create Physical Realizability boolean mask using ONNX MLP
% Result variable written to the input MAT file:
%   - pr_mask : [H,W] logical

clear; clc;

%% Configuration
data_file     = 'sample/PPRIM.mat';         % must contain nM (H,W,12) or X_flat (N,12)
mlp_onnx      = 'model/pixel_mlp.onnx';     % exported from PyTorch (logit output)
mlp_func_path = 'model/MLPFunction';        % generated function location/name (no .m)

batch_size = 500000;   % adjust to your memory
thresh     = 0.5;      % PR threshold
use_gpu    = canUseGPU();

fprintf('=== PRMask_FromONNX_MLP ===\n');

%% Check files
assert(isfile(data_file), 'Data file not found: %s', data_file);
assert(isfile(mlp_onnx),  'ONNX file not found: %s', mlp_onnx);

%% Load data
S = load(data_file);

if isfield(S, 'nM')
    nM = S.nM;                         % [H,W,12]
    [H, W, C] = size(nM);
    assert(C == 12, 'Expected 12 features in nM.');
    X = reshape(nM, [], 12);           % [N,12]
    reshape_back = true;
    fprintf('Loaded nM: %d x %d x %d  → N=%d\n', H, W, C, H*W);
elseif isfield(S, 'X_flat')
    X = S.X_flat;                      % [N,12]
    [N, C] = size(X); %#ok<ASGLU>
    assert(C == 12, 'Expected 12 features in X_flat.');
    reshape_back = false;
    H = []; W = [];
    fprintf('Loaded X_flat: N=%d x 12\n', size(X,1));
else
    error('Provide nM (H x W x 12) or X_flat (N x 12) in %s', data_file);
end

N = size(X,1);
X = single(X);

%% Prepare generated function
[funcFolder, funcName] = fileparts(mlp_func_path);
if ~isempty(funcFolder)
    addpath(funcFolder);
end

if ~isfile(fullfile(funcFolder, [funcName '.m']))
    fprintf('Generating MLP function (first time)...\n');
    importONNXFunction(mlp_onnx, funcName);
end

try
    params_mlp = importONNXFunction(mlp_onnx, funcName);
    fprintf('MLP parameters loaded successfully.\n');
catch ME
    error('Failed to load ONNX model: %s', ME.message);
end

mlp_fh = str2func(funcName);

%% Inference (batched)
fprintf('Running MLP inference on %d pixels ...\n', N);
tic;

mask = false(N,1);

for i = 1:batch_size:N
    j   = min(i + batch_size - 1, N);
    idx = i:j;

    Xb  = X(idx, :);
    Xdl = dlarray(Xb);

    if use_gpu
        Xdl = gpuArray(Xdl);
    end

    y_logits = mlp_fh(Xdl, params_mlp);     % [batch,1] dlarray
    y_logits = gather(extractdata(y_logits));

    % Sigmoid + threshold (directly to boolean)
    mask(idx) = (1 ./ (1 + exp(-single(y_logits)))) > thresh;
end

runtime = toc;
fprintf('Done in %.3f s (%.0f pix/s)\n', runtime, N/runtime);

%% Reshape and save
if reshape_back
    pr_mask = reshape(mask, H, W);
else
    pr_mask = mask;
end

S.pr_mask = pr_mask;
save(data_file, '-struct', 'S');

fprintf('Saved pr_mask to: %s\n', data_file);