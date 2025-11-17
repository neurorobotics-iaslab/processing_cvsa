clc; clear all; close all;

%% process throught matlab
% not modification needed for these informations
datapath = '/home/paolo/cvsa/ic_cvsa_ws/src/processing_cvsa/test/';
filein = [datapath ,'rawdata.csv'];
data = readmatrix(filein);
filterOrder = 4;
band = [8, 14];
bufferSize = 512;
sampleRate = 512;
frameSize = 32;
nsamples = size(data, 1);
nchannels = size(data, 2);

%% apply the processing
disp(['      [INFO] start processing like ros for band ' num2str(band(1)) '-' num2str(band(2))]);
nchunks = nsamples / frameSize;
buffer = nan(bufferSize, nchannels);

[b_low, a_low] = butter(filterOrder, band(2)*(2/sampleRate),'low');
[b_high, a_high] = butter(filterOrder, band(1)*(2/sampleRate),'high');
zi_low = [];
zi_high = [];

signal_processed = nan(nchunks - bufferSize/frameSize + 1, nchannels);

for i=1:nchunks
    % add
    frame = data((i-1)*frameSize+1:i*frameSize,:);
    buffer(1:end-frameSize,:) = buffer(frameSize+1:end,:);
    buffer(end-frameSize+1:end, :) = frame;

    % check
    if any(isnan(buffer))
        continue;
    end

    % apply low and high pass filters
    [tmp_data, zi_low] = filter(b_low,a_low,buffer,zi_low);
    [tmp_data,zi_high] = filter(b_high,a_high,tmp_data,zi_high);

    % apply power with hilbert
    analytic = hilbert(tmp_data);
    tmp_data = abs(analytic).^2;

    % apply average
    tmp_data = mean(tmp_data, 1);

    signal_processed(i - bufferSize/frameSize + 1,:) = tmp_data;
end

%% Load file of rosneuro
channelId = 10;
SampleRate = 16;
start = 50;

files{1} = [datapath 'class/processing.csv'];
files{2} = [datapath 'node/processing.csv'];

for i=1:length(files)
    file = files{i};
    disp(['Loading file: ' file])
    ros_data = readmatrix(file);
    matlab_data = signal_processed;
    if i == 1
        c_title = "processed with the class";
    else
        c_title = "processed with ros node simulation";
    end
    nsamples = size(matlab_data,1);
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
end





