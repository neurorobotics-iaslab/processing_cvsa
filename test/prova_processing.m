% clc; clear all; close all;
% create the dataset log band power, all channels and freq 8-14 ---> uqual
% to main4d

start = 1;

%% process throught matlab
% not modification needed for these informations
sampleRate = 512;
filein = '/home/paolo/cvsa_ws/src/processing_cvsa/test/rawdata.csv';
data = readmatrix(filein);
filterOrder = 4;
band = [8, 14];
bufferSize = 512;
frameSize = 32;

[s_buffer, s_band, s_pow, s_avg, s_log] = online_rosneuro(data, bufferSize, frameSize, band, filterOrder, sampleRate);


%% Load file of rosneuro
channelId = 1;
SampleRate = 512;

files{1} = '/home/paolo/cvsa_ws/src/processing_cvsa/test/bandpass.csv';
files{2} = '/home/paolo/cvsa_ws/src/processing_cvsa/test/pow.csv';
files{3} = '/home/paolo/cvsa_ws/src/processing_cvsa/test/avg.csv';
files{4} = '/home/paolo/cvsa_ws/src/processing_cvsa/test/log.csv';
files{5} = '/home/paolo/cvsa_ws/src/processing_cvsa/test/buffer.csv';

for i=1:length(files)
    file = files{i};
    disp(['Loading file: ' file])
    rosneuro_data = readmatrix(file);
    if i == 1
        matlab_data = s_band;
        c_title = "butterworth";
        nsamples = size(matlab_data,1);
        t = 0:1/SampleRate:nsamples/SampleRate - 1/SampleRate;
    elseif i == 2
        matlab_data = s_pow;
        c_title = "power";
        nsamples = size(matlab_data,1);
        t = 0:1/SampleRate:nsamples/SampleRate - 1/SampleRate;
    elseif i ==3
        matlab_data = s_avg;
        c_title = "avg";
        nsamples = size(matlab_data,1);
        t = 0:1/SampleRate:nsamples/SampleRate - 1/SampleRate;
    elseif i ==4
        matlab_data = s_log;
        c_title = "log";
        nsamples = size(matlab_data,1);
        t = 0:1/SampleRate:nsamples/SampleRate - 1/SampleRate;
    else
        matlab_data = s_buffer;
        c_title = "buffer";
        nsamples = size(matlab_data,1);
        t = 0:1/SampleRate:nsamples/SampleRate - 1/SampleRate;
    end

    figure;
    subplot(2, 1, 1);
    hold on;
    plot(t(start:end), rosneuro_data(start:end, channelId), 'b', 'LineWidth', 1);
    plot(t(start:end), matlab_data(start:end, channelId), 'r');
    legend('rosneuro', 'matlab');
    hold off;
    grid on;

    subplot(2,1,2)
    bar(t(start:end), abs(rosneuro_data(start:end, channelId)- matlab_data(start:end, channelId)));
    grid on;
    xlabel('time [s]');
    ylabel('amplitude [uV]');
    title('Difference')

    sgtitle(['Evaluation' c_title])
end





