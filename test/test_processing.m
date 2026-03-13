clc; clear all; close all;

%% process throught matlab
% not modification needed for these informations
datapath = './src/processing_bci/test/';
filein = [datapath ,'rawdata.csv'];
data = readmatrix(filein);
filterOrder = 4;
band = [8, 14];
bufferSize = 500;
sampleRate = 500;
chunkSize = 25;
nsamples = size(data, 1);
nchannels = size(data, 2);

%% apply the processing
disp(['      [INFO] start processing like ros for band ' num2str(band(1)) '-' num2str(band(2))]);
nchunks = nsamples / chunkSize;
buffer = nan(bufferSize, nchannels);

[b_low, a_low] = butter(filterOrder, band(2)*(2/sampleRate),'low');
[b_high, a_high] = butter(filterOrder, band(1)*(2/sampleRate),'high');
zi_low = [];
zi_high = [];

h.EVENT.POS = [];
h.EVENT.DUR = [];
h.SampleRate = 500;

[signal_processed, ~] = processing_onlineROS_CAR_hilbert(data, h, nchannels, bufferSize, filterOrder, band, chunkSize, [1,2,19]);


%% Load file of rosneuro
channelId = 11;
SampleRate = 20;
start = 2 * bufferSize/chunkSize; % start after 2 seconds
align = bufferSize/chunkSize; % to remove the NAN in the matlab data

file = [datapath 'processing.csv'];
disp(['Loading file: ' file])
ros_data = readmatrix(file);
matlab_data = signal_processed;

matlab_data = matlab_data(2:end,:); % ros miss the first message
ros_data = ros_data(align:end, :);
matlab_data = matlab_data(align:end,:);
c_title = "processed with ros node simulation";

nsamples = min(size(ros_data,1), size(matlab_data,1));
t = 0:1/SampleRate:nsamples/SampleRate - 1/SampleRate;


figure;
subplot(2, 1, 1);
hold on;
plot(t(start:end), ros_data(start:size(t,2), channelId), 'b', 'LineWidth', 1);
plot(t(start:end), matlab_data(start:size(t,2), channelId), 'r');
legend('rosneuro', 'matlab');
hold off;
grid on;

subplot(2,1,2)
bar(t(start:end), abs(ros_data(start:size(t,2), channelId)- matlab_data(start:size(t,2), channelId)));
grid on;
xlabel('time [s]');
ylabel('amplitude [uV]');
title('Difference')

sgtitle(['Evaluation' c_title])






