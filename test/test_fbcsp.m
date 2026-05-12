clc; clear all; close all;

%% process through matlab
datapath = './src/processing_bci/';
filein = [datapath, 'test/rawdata.csv'];
data = readmatrix(filein);
filterOrder = 4;
bands = [8, 10; 10 12; 12 14; 8 14; 14 20];
bufferSize = 500;
sampleRate = 500;
chunkSize = 25;
nsamples = size(data, 1);
nchannels = size(data, 2);

%% Load YAML files
do_ica = false;
do_car = true;

if do_ica
    ica_struct = yaml.ReadYaml([datapath, 'ica.yaml']);
    ica_matrix = cell2mat(ica_struct.ica_matrix);
end

if do_car
    EOG_ch = [1, 32];
    testo = sprintf('do_car true; eog channels: %s', mat2str(EOG_ch));
    disp(testo);
end

% csp_matrices is a cell array of cell arrays of cell arrays
csp_struct = yaml.ReadYaml([datapath, 'cfg/csp/csp_test.yaml']);
ncsp_bands = length(csp_struct.CspCfg.params.csp_matrices);
csp_matrices = cell(ncsp_bands, 1);
for b = 1:ncsp_bands
    csp_matrices{b} = cell2mat(csp_struct.CspCfg.params.csp_matrices{b});
end
ncomponents = size(csp_matrices{1}, 1);

%% Apply the processing (Simulating ROS node)
disp('      [INFO] start FBCSP processing like ros');
anline_ros = 1; % ros miss the first chunk

% 1. ICA
if do_ica
    ica_data = data * ica_matrix';
else
    ica_data = data;
end

% 2. CAR (Common Average Reference)
if do_car
    idxs = 1:32;
    idxs(EOG_ch) = [];
    car_data = ica_data - mean(ica_data(:,idxs), 2);
else
    car_data = ica_data;
end

% 3. Butterworth Filter Setup
% Note: in rosneuro_filters, it cascades LowPass and HighPass
zi_low = cell(ncsp_bands, 1);
zi_high = cell(ncsp_bands, 1);
b_low = cell(ncsp_bands, 1);
a_low = cell(ncsp_bands, 1);
b_high = cell(ncsp_bands, 1);
a_high = cell(ncsp_bands, 1);

for b = 1:ncsp_bands
    [b_low{b}, a_low{b}] = butter(filterOrder, bands(b, 2)*(2/sampleRate), 'low');
    [b_high{b}, a_high{b}] = butter(filterOrder, bands(b, 1)*(2/sampleRate), 'high');
    zi_low{b} = zeros(max(length(a_low{b}), length(b_low{b})) - 1, nchannels);
    zi_high{b} = zeros(max(length(a_high{b}), length(b_high{b})) - 1, nchannels);
end

nchunks = floor(nsamples / chunkSize);
buffers = zeros(bufferSize, nchannels, ncsp_bands);
matlab_features = zeros(nchunks, ncomponents * ncsp_bands);
buffer_idx = 1;

for k = anline_ros:nchunks
    chunk_start = (k-1)*chunkSize + 1;
    chunk_end = k*chunkSize;
    chunk_data = car_data(chunk_start:chunk_end, :);
    
    chunk_features = zeros(1, ncomponents * ncsp_bands);
    
    for b = 1:ncsp_bands
        % Bandpass filter
        [filtered_low, zi_low{b}] = filter(b_low{b}, a_low{b}, chunk_data, zi_low{b});
        [filtered_high, zi_high{b}] = filter(b_high{b}, a_high{b}, filtered_low, zi_high{b});
        
        % Push to buffer
        if buffer_idx + chunkSize - 1 <= bufferSize
            buffers(buffer_idx:buffer_idx+chunkSize-1, :, b) = filtered_high;
        else
            % Shift buffer
            buffers(1:bufferSize-chunkSize, :, b) = buffers(chunkSize+1:end, :, b);
            buffers(bufferSize-chunkSize+1:end, :, b) = filtered_high;
        end
    end
    
    if k >= bufferSize/chunkSize
        % Buffer is full, compute features
        for b = 1:ncsp_bands
            buf_data = buffers(:, :, b);
            % Spatial filter
            csp_data = buf_data * csp_matrices{b}';
            % Variance
            var_data = var(csp_data, 0, 1);
            
            % Store in features row
            start_idx = (b-1)*ncomponents + 1;
            end_idx = b*ncomponents;
            chunk_features(start_idx:end_idx) = var_data;
        end
    end
    matlab_features(k, :) = chunk_features;

    if buffer_idx + chunkSize - 1 <= bufferSize
        buffer_idx = buffer_idx + chunkSize;
    end
end

%% Load file of rosneuro
start_chunk = bufferSize/chunkSize; 
align = bufferSize/chunkSize + 40; % align to remove zero elements before buffer is full
ros_skip_msgs = 0;

file = [datapath 'test/fbcsp_processing.csv'];
disp(['Loading file: ' file])
ros_data = readmatrix(file);

% Align data
ros_data = ros_data(align:end, :);
matlab_data = matlab_features(align+ros_skip_msgs:end, :);

c_title = " FBCSP processed with ros node simulation";

n_eval_samples = min(size(ros_data,1), size(matlab_data,1));
t = (0:n_eval_samples-1) * (chunkSize/sampleRate);

componentId = 1; % Plot the first component

figure;
subplot(2, 1, 1);
hold on;
plot(t, ros_data(1:n_eval_samples, componentId), 'b', 'LineWidth', 1);
plot(t, matlab_data(1:n_eval_samples, componentId), 'r--', 'LineWidth', 1);
legend('rosneuro', 'matlab');
hold off;
grid on;
ylabel('Variance');

subplot(2,1,2)
bar(t, abs(ros_data(1:n_eval_samples, componentId) - matlab_data(1:n_eval_samples, componentId)));
grid on;
xlabel('time [s]');
ylabel('absolute error');
title('Difference')

sgtitle(['Evaluation' c_title])

disp(['max value diff between data_matlab and data_ros: ' num2str(max(matlab_data - ros_data, [],'all'))])
