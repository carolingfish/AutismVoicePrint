%==========================================================================
%Decription:
%   data pre-process: energy? voiced? nasalized?
%Input:
%   filename: sound file
%Output:
%   data: processed data
%Usage:
%   data = soundPreProcess('nmFC_0001.wav')
%==========================================================================
function data = soundPreProcess(filename)
%% read data
    [y, fps] = audioread(filename); % read the audio data
    y = y(:, 1);%only check one chanel 
    y = zscore(y);%normalize
    WindowDurationInSeconds = 50.0*10^(-3); %window size: 50ms 
    ShiftDurationInSeconds = 10.0*10^(-3); % 
    WindowFrameCount = fps*WindowDurationInSeconds;
    ShiftFrameCount = fps*ShiftDurationInSeconds;
    
%% energy checking
    filelength = numel(y);
    energies = [];
    passenergies = [];
    data = [];
    for windowIndex = 1 : ShiftFrameCount : filelength 
        windowEndIndex = min(windowIndex + WindowFrameCount, filelength);
        shiftEndIndex = min(windowIndex + ShiftFrameCount, filelength);
        thiswindow = y(windowIndex:windowEndIndex);
        energy = thiswindow' * thiswindow / WindowFrameCount;
        energies = [energies; energy];
        if (energy > 0.5) %threshold
            passenergies = [passenergies; energy];
            data = [data; y(windowIndex:shiftEndIndex)]; 
        end
    end
    figure(1);
    plot([1: numel(energies)], energies);
    xlabel('time');
    ylabel('energe');
    
    figure(2);
    plot([1: numel(passenergies)], passenergies);
    xlabel('time');
    ylabel('passenergies');
    
    audiowrite(['EF_', filename],data,fps); % write a WAVE (.wav) file in the current folder.
    sound(data,fps); %Listen to the audio.
%% voiced checking 
%http://www.mathworks.com/help/signal/ug/find-periodicity-using-autocorrelation.html
%http://dsp.stackexchange.com/questions/15114/cant-find-out-the-period-of-my-signal
    minPitch = 50;
    maxPitch = 200;
    deltaLow = floor(fps / maxPitch);
    deltaHigh = ceil(fps / minPitch);
    
    [r, lags] = xcorr(data);
    %smooth to avoid noise. 
    r = smooth(r(find(lags == deltaLow) : find(lags == deltaHigh)), deltaLow / 2);
    figure(3);
    plot([deltaLow: deltaHigh], r);
    xlabel('Lag');
    ylabel('Autocorrelation');
    [pks,locs]=findpeaks(r);
    figure(4);
    stem(locs, pks);
    tmp = diff(locs);
    tmp = tmp(tmp >= deltaLow & tmp <= deltaHigh);
    assert(~isempty(tmp), 'It is not voiced!');
    delta = min(tmp);

%% nasalized checking


end

