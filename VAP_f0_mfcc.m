addpath('./mfcc_code')

% Variables for F0 tracking
voicingThresh = 0.05;
silenceThresh = 1e-2;
ws = 0.01; % 25ms

% Variables for MFCC
    Tw = 50;                % analysis frame duration (ms)
    Ts = 10;                % analysis frame shift (ms)
    alpha = 0.97;           % preemphasis coefficient
    M = 40;                 % number of filterbank channels 
    C = 24;                 % number of cepstral coefficientsHy
    L = 22;                 % cepstral sine lifter parameter
    LF = 30;               % lower frequency limit (Hz)
    HF = 3700;              % upper frequency limit (Hz)

wav_file = './Samples/Nasalized/ST_0001.wav';  % input audio filename



% Read speech samples, sampling rate and precision from file
[ speech, fs] = audioread( wav_file );

% Check if speech is mono, if not, convert it to mono
if( min(size(speech))~=1 )speech = mean(speech,2);end;


[f0, tt] = track_f0(speech, ws, voicingThresh, silenceThresh);

plot(f0);

% Convert to MFCC
% Feature extraction (feature vectors as columns)
[ MFCCs, FBEs, frames ] = mfcc( speech, fs, Tw, Ts, alpha, @hamming, [LF HF], M, C+1, L );

% Pick out voiced frames
f0_frames = convert_f0_to_frames(f0,tt,ws);

f0_frames = f0_frames(3:end-2); % Need to change this if the frame durations are different...
voiced_frames = find(f0_frames~=0);

voiced_MFCCs = MFCCs(:,voiced_frames);