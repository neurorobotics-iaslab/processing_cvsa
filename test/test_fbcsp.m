%% FBCSP MATLAB simulation
% Replicates the ROS processing_fbcsp_node chunk by chunk with causal IIR
% filters and a ring buffer, mirroring configure() + apply() in Fbcsp.cpp.
%
% KEY alignment principle:
%   ROS starts processing at seq = first_seq (first frame actually received).
%   MATLAB must also start at first_seq to have identical IIR filter initial
%   state. For the GDF input an additional 1-frame acquisition pipeline delay
%   is measured via cross-correlation and corrected automatically.
%
% Workflow (GDF):
%   1. roslaunch processing_bci test_node_fbcsp_gdf.launch
%   2. Wait for file to finish, Ctrl+C → fbcsp_gdf_output.csv +
%      fbcsp_gdf_output_first_seq.txt are written.
%   3. Run this script.
%
% Workflow (CSV):
%   1. roslaunch processing_bci test_node_fbcsp.launch
%   2. Wait until Ctrl+C → fbcsp_processing.csv + fbcsp_processing_first_seq.txt
%   3. Run this script with input_mode = 'csv'.

clear all; clc; close all;

%% --- input mode ---
input_mode = 'gdf';   % 'gdf' | 'csv'

%% --- paths ---
pkgpath         = './src/processing_bci/';
car_yaml        = [pkgpath 'cfg/car.yaml'];
csp_yaml        = [pkgpath 'cfg/csp/mi/csp_test.yaml'];

if strcmp(input_mode, 'gdf')
    input_file    = [pkgpath 'test/prova32ch.gdf'];
    ros_file      = [pkgpath 'test/fbcsp_gdf_output.csv'];
    framerate     = 16;
    plot_start_s  = 2;    % skip this many seconds at the start ([] = show all)
else
    input_file    = [pkgpath 'test/rawdata.csv'];
    ros_file      = [pkgpath 'test/fbcsp_processing.csv'];
    framerate     = 20;
    plot_start_s  = [];   % show full recording
end

%% --- read first_seq ---
first_seq_file = strrep(ros_file, '.csv', '_first_seq.txt');
first_seq = 0;
if isfile(first_seq_file)
    first_seq = readmatrix(first_seq_file);
    fprintf('ROS first_seq = %d (lost %d frame(s) at startup)\n', first_seq, first_seq);
else
    fprintf('first_seq file not found – assuming first_seq = 0.\n');
end

%% --- load CAR config ---
car_cfg      = yaml.ReadYaml(car_yaml);
eog_ch_names = car_cfg.CarCfg.params.EOG_ch_names;

%% --- load CSP config ---
csp_cfg    = yaml.ReadYaml(csp_yaml);
csp_params = csp_cfg.CspCfg.params;

ncsp_bands   = length(csp_params.csp_matrices);
csp_matrices = cell(ncsp_bands, 1);
for b = 1:ncsp_bands
    csp_matrices{b} = cell2mat(csp_params.csp_matrices{b});
end
ncomponents = size(csp_matrices{1}, 1);

% frequency bands from CSP yaml
bands_raw = csp_params.bands;   % cell array of 2-element cells
bands = cell2mat(bands_raw);

% selected channels (optional)
if isfield(csp_params, 'selected_channels')
    csp_ch_names = csp_params.selected_channels;
else
    csp_ch_names = {};
end

filterOrder = 4;

fprintf('--- FBCSP config ---\n');
fprintf('  bands       : %d\n', ncsp_bands);
fprintf('  components  : %d\n', ncomponents);
fprintf('  EOG exclude : [%s]\n', strjoin(eog_ch_names, ', '));
if ~isempty(csp_ch_names)
    fprintf('  CSP channels: [%s]\n', strjoin(csp_ch_names, ', '));
else
    fprintf('  CSP channels: all\n');
end

%% --- load data ---
[~, ~, ext] = fileparts(input_file);
if strcmpi(ext, '.gdf')
    [data_raw, hdr] = sload(input_file);   % BIOSIG required
    sampleRate = hdr.SampleRate;
    ch_names   = cellstr(hdr.Label);
    n_eeg = sum(~cellfun(@(c) contains(lower(c), {'status','trigger','mkr'}), ch_names));
    data     = data_raw(:, 1:n_eeg);
    ch_names = ch_names(1:n_eeg);
    fprintf('GDF: %d samples x %d EEG channels @ %.0f Hz\n', size(data,1), n_eeg, sampleRate);
else
    data       = readmatrix(input_file);
    sampleRate = 500;
    ch_names   = {'Fp1','Fp2','F3','Fz','F4','FC1','FC2','C3','Cz','C4', ...
                  'CP1','CP2','P3','Pz','P4','POz','O1','O2','CPz','F1', ...
                  'F2','FC5','FCz','FC6','C1','C2','CP5','CP6','P5','P1','P2','P6'};
    ch_names   = ch_names(1:size(data,2));
    fprintf('CSV: %d samples x %d channels @ %.0f Hz\n', size(data,1), size(data,2), sampleRate);
end

nchannels = size(data, 2);
chunkSize = round(sampleRate / framerate);
bufferSize = round(sampleRate);  % ring buffer = 1 second

fprintf('  chunkSize   : %d samples\n', chunkSize);
fprintf('  bufferSize  : %d samples\n', bufferSize);

%% --- resolve EOG channels (1-based) ---
EOG_ch = zeros(1, numel(eog_ch_names));
for k = 1:numel(eog_ch_names)
    m = find(strcmpi(ch_names, eog_ch_names{k}), 1);
    if isempty(m)
        error('EOG channel "%s" not found.\nAvailable: %s', eog_ch_names{k}, strjoin(ch_names, ', '));
    end
    EOG_ch(k) = m;
end
non_eog_ch = setdiff(1:nchannels, EOG_ch);
fprintf('EOG channels: [%s] → indices [%s]\n', strjoin(eog_ch_names, ', '), num2str(EOG_ch));

%% --- resolve CSP channel subset (1-based) ---
if isempty(csp_ch_names)
    csp_ch = 1:nchannels;
else
    csp_ch = zeros(1, numel(csp_ch_names));
    for k = 1:numel(csp_ch_names)
        m = find(strcmpi(ch_names, csp_ch_names{k}), 1);
        if isempty(m)
            error('CSP channel "%s" not found.\nAvailable: %s', csp_ch_names{k}, strjoin(ch_names, ', '));
        end
        csp_ch(k) = m;
    end
    fprintf('CSP channel indices: [%s]\n', num2str(csp_ch));
end

%% --- design causal IIR filters (same order as Fbcsp.cpp: LP then HP) ---
nyq   = sampleRate / 2;
b_lp  = cell(ncsp_bands, 1);  a_lp  = cell(ncsp_bands, 1);
b_hp  = cell(ncsp_bands, 1);  a_hp  = cell(ncsp_bands, 1);
zi_lp = cell(ncsp_bands, 1);  zi_hp = cell(ncsp_bands, 1);
for b = 1:ncsp_bands
    [b_lp{b}, a_lp{b}] = butter(filterOrder, bands(b,2) / nyq, 'low');
    [b_hp{b}, a_hp{b}] = butter(filterOrder, bands(b,1) / nyq, 'high');
    zi_lp{b} = zeros(max(length(a_lp{b}), length(b_lp{b})) - 1, nchannels);
    zi_hp{b} = zeros(max(length(a_hp{b}), length(b_hp{b})) - 1, nchannels);
end

%% --- init ring buffers (NaN = not yet full, mirrors C++ RingBuffer) ---
% Shape: [bufferSize x nchannels x ncsp_bands]
bufs = nan(bufferSize, nchannels, ncsp_bands);

%% --- chunk-by-chunk processing (starts at first_seq, same as ROS) ---
total_samples  = size(data, 1);
n_frames       = floor(total_samples / chunkSize);
n_total_feat   = ncomponents * ncsp_bands;
% Pre-allocate seq-indexed: row f+1 = features for seq f
matlab_features = zeros(n_frames, n_total_feat);

for seq = first_seq : n_frames - 1
    f   = seq + 1;   % 1-based
    idx = seq * chunkSize + 1 : (seq + 1) * chunkSize;
    chunk = data(idx, :);   % [chunkSize x nchannels]

    % CAR: subtract mean of non-EOG channels (per sample)
    car_mean  = mean(chunk(:, non_eog_ch), 2);   % [chunkSize x 1]
    chunk_car = chunk - car_mean;                 % [chunkSize x nchannels]

    % Per-band: LP → HP (causal, state preserved), then push into ring buffer
    for b = 1:ncsp_bands
        [lp_out, zi_lp{b}] = filter(b_lp{b}, a_lp{b}, chunk_car, zi_lp{b}, 1);
        [bp_out, zi_hp{b}] = filter(b_hp{b}, a_hp{b}, lp_out,    zi_hp{b}, 1);

        % Shift-register: drop oldest chunkSize rows, append new
        bufs(:, :, b) = [bufs(chunkSize+1:end, :, b); bp_out];
    end

    % Buffer not full yet (NaN still present)
    if any(isnan(bufs(:)))
        continue;
    end

    % Compute FBCSP features: CSP spatial filter → mean(x²) per component
    csp_features = zeros(ncomponents, ncsp_bands);   % [n_comp x n_bands]
    for b = 1:ncsp_bands
        buf_sel  = bufs(:, csp_ch, b);                    % [bufferSize x n_csp_ch]
        csp_out  = buf_sel * csp_matrices{b}';            % [bufferSize x ncomponents]
        csp_features(:, b) = sum(csp_out .^ 2, 1)' / bufferSize;  % mean(x²)
    end

    % Column-major flatten to match Eigen / C++ memcpy order:
    % [comp1_band1, comp2_band1, ..., compN_band1, comp1_band2, ...]
    matlab_features(f, :) = reshape(csp_features, 1, []);
end

%% --- compare with ROS output ---
if ~isfile(ros_file)
    warning('ROS output not found: %s', ros_file);
    frame_rate  = sampleRate / chunkSize;
    warmup      = ceil(bufferSize / chunkSize);
    t = (0:n_frames-1) / frame_rate;
    s = first_seq + warmup + 1;
    figure;
    plot(t(s:end), matlab_features(s:end, 1), 'r');
    xlabel('time [s]'); ylabel('feature 1'); title('MATLAB only (no ROS ref)'); grid on;
    return;
end

ros_data = readmatrix(ros_file);   % [n_seq x n_features], row k = seq k-1

n_compare    = min(n_frames, size(ros_data, 1));
ros_feats    = ros_data(1:n_compare, :);
mat_feats    = matlab_features(1:n_compare, :);

frame_rate    = sampleRate / chunkSize;
warmup_frames = ceil(bufferSize / chunkSize);
start_frame   = first_seq + warmup_frames + 1;   % 1-based

t = (0:n_compare-1) / frame_rate;

fprintf('\nAlignment: first_seq=%d, warmup=%d → comparing from seq %d (frame %d)\n', ...
        first_seq, warmup_frames, start_frame-1, start_frame);

%% --- cross-correlation on first feature component to measure acquisition lag ---
MAX_LAG_SEARCH = 5;
valid = start_frame : n_compare;
r_v   = ros_feats(valid, 1);
m_v   = mat_feats(valid, 1);

[xcf, lags] = xcorr(r_v - mean(r_v), m_v - mean(m_v), MAX_LAG_SEARCH, 'normalized');
[~, peak_idx]  = max(xcf);
measured_lag   = lags(peak_idx);   % positive → ROS lags MATLAB by that many frames

fprintf('Cross-correlation peak lag: %+d frame(s)  ', measured_lag);
if measured_lag == 0
    fprintf('[no residual lag]\n');
elseif measured_lag > 0
    fprintf('[ROS lags MATLAB by %d frame(s) – eegdev acquisition pipeline delay]\n', measured_lag);
else
    fprintf('[MATLAB lags ROS by %d frame(s)]\n', -measured_lag);
end

% Apply lag correction (all features, not just comp 1)
if measured_lag > 0
    r_aligned = ros_feats(valid(1 + measured_lag : end), :);
    m_aligned = mat_feats(valid(1             : end - measured_lag), :);
    t_aligned = t(valid(1 : end - measured_lag));
elseif measured_lag < 0
    shift     = -measured_lag;
    r_aligned = ros_feats(valid(1         : end - shift), :);
    m_aligned = mat_feats(valid(1 + shift : end        ), :);
    t_aligned = t(valid(1 : end - shift));
else
    r_aligned = ros_feats(valid, :);
    m_aligned = mat_feats(valid, :);
    t_aligned = t(valid);
end

%% --- restrict window for plots/stats (skip first plot_start_s seconds) ---
if ~isempty(plot_start_s)
    skip_frames    = round(plot_start_s * frame_rate);
    raw_skip       = min(skip_frames, numel(valid));
    aligned_skip   = min(skip_frames, size(r_aligned, 1));
else
    raw_skip     = 0;
    aligned_skip = 0;
end

valid_plot = valid(raw_skip+1 : end);
r_v_plot   = ros_feats(valid_plot, :);
m_v_plot   = mat_feats(valid_plot, :);
t_raw_plot = t(valid_plot);

r_al_plot  = r_aligned(aligned_skip+1 : end, :);
m_al_plot  = m_aligned(aligned_skip+1 : end, :);
t_al_plot  = t_aligned(aligned_skip+1 : end);

% Mismatch metric: mean absolute error on first component (over plot window)
mae_raw     = mean(abs(r_v_plot(:,1)  - m_v_plot(:,1)));
mae_aligned = mean(abs(r_al_plot(:,1) - m_al_plot(:,1)));

if ~isempty(plot_start_s)
    fprintf('Plotting from %.1f s onward (%d raw / %d aligned frames shown)\n', ...
            plot_start_s, numel(valid_plot), size(r_al_plot,1));
end
fprintf('MAE feature-1 (raw)     : %.6f\n', mae_raw);
fprintf('MAE feature-1 (aligned) : %.6f\n', mae_aligned);

%% --- plot: raw ---
componentId = 1;
figure;
subplot(2,1,1);
hold on;
plot(t_raw_plot, r_v_plot(:, componentId), 'b',   'LineWidth', 1.5);
plot(t_raw_plot, m_v_plot(:, componentId), 'r--', 'LineWidth', 1);
legend('ROS node', 'MATLAB simulation');
ylabel('mean power');
title(sprintf('[RAW] %s | bufSize=%d | comp=%d | first\\_seq=%d', ...
    upper(input_mode), bufferSize, componentId, first_seq));
grid on; hold off;
subplot(2,1,2);
bar(t_raw_plot, abs(r_v_plot(:, componentId) - m_v_plot(:, componentId)));
xlabel('time [s]'); ylabel('|diff|');
title(sprintf('Differences (measured lag=%+d)', measured_lag)); grid on;

%% --- plot: lag-corrected ---
figure;
subplot(2,1,1);
hold on;
plot(t_al_plot, r_al_plot(:, componentId), 'b',   'LineWidth', 1.5);
plot(t_al_plot, m_al_plot(:, componentId), 'r--', 'LineWidth', 1);
legend('ROS node', 'MATLAB simulation');
ylabel('mean power');
title(sprintf('[ALIGNED lag=%+d] %s | comp=%d', measured_lag, upper(input_mode), componentId));
grid on; hold off;
subplot(2,1,2);
bar(t_al_plot, abs(r_al_plot(:, componentId) - m_al_plot(:, componentId)));
xlabel('time [s]'); ylabel('|diff|');
title('Differences after alignment'); grid on;
