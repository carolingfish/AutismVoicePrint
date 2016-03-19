addpath('./mfcc_code')

% Variables for F0 tracking
voicingThresh = 0.1;
silenceThresh = 0.05;
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

    
normal_dir = './Vocal_Samples/Normal';
nasal_dir = './Vocal_Samples/Nasal';
normal_files = ls(normal_dir);
normal_m_files = ls(strcat(normal_dir,'/M*'));
normal_f_files = ls(strcat(normal_dir,'/F*'));
nasal_files = ls(nasal_dir);
nasal_m_files = ls(strcat(nasal_dir,'/M*'));
nasal_f_files = ls(strcat(nasal_dir,'/F*'));

normal_M_MFCCs = [];
normal_F_MFCCs = [];
nasal_M_MFCCs = [];
nasal_F_MFCCs = [];

% Male Normal
for i=1:size(normal_m_files,1)
    wav_file = strcat(normal_dir,'/',normal_m_files(i,:));  % input audio filename

    % Read speech samples, sampling rate and precision from file
    [ speech, fs] = audioread( wav_file );

    % Check if speech is mono, if not, convert it to mono
    if( min(size(speech))~=1 )speech = mean(speech,2);end;

    [f0, tt] = track_f0(speech, ws, voicingThresh, silenceThresh);

    % Convert to MFCC
    % Feature extraction (feature vectors as columns)
    [ MFCCs, FBEs, frames ] = mfcc( speech, fs, Tw, Ts, alpha, @hamming, [LF HF], M, C+1, L );

    % Pick out voiced frames
    f0_frames = convert_f0_to_frames(f0,tt,ws);

    f0_frames = f0_frames(3:end-2); % Need to change this if the frame durations are different...
    voiced_frames = find(f0_frames~=0);

    voiced_MFCCs = MFCCs(:,voiced_frames);
    
    ordering = randperm(size(voiced_MFCCs,2));
    voiced_MFCCs = voiced_MFCCs(:,ordering);
    normal_M_MFCCs = horzcat(normal_M_MFCCs,voiced_MFCCs);
end
% Female Normal
for i=1:size(normal_f_files,1)
    wav_file = strcat(normal_dir,'/',normal_f_files(i,:));  % input audio filename

    % Read speech samples, sampling rate and precision from file
    [ speech, fs] = audioread( wav_file );

    % Check if speech is mono, if not, convert it to mono
    if( min(size(speech))~=1 )speech = mean(speech,2);end;

    [f0, tt] = track_f0(speech, ws, voicingThresh, silenceThresh);

    % Convert to MFCC
    % Feature extraction (feature vectors as columns)
    [ MFCCs, FBEs, frames ] = mfcc( speech, fs, Tw, Ts, alpha, @hamming, [LF HF], M, C+1, L );

    % Pick out voiced frames
    f0_frames = convert_f0_to_frames(f0,tt,ws);

    f0_frames = f0_frames(3:end-2); % Need to change this if the frame durations are different...
    voiced_frames = find(f0_frames~=0);

    voiced_MFCCs = MFCCs(:,voiced_frames);
    
    ordering = randperm(size(voiced_MFCCs,2));
    voiced_MFCCs = voiced_MFCCs(:,ordering);
    normal_F_MFCCs = horzcat(normal_F_MFCCs,voiced_MFCCs);
end

% Male Nasal
for i=1:size(nasal_m_files,1)
    wav_file = strcat(nasal_dir,'/',nasal_m_files(i,:));  % input audio filename

    % Read speech samples, sampling rate and precision from file
    [ speech, fs] = audioread( wav_file );

    % Check if speech is mono, if not, convert it to mono
    if( min(size(speech))~=1 )speech = mean(speech,2);end;

    [f0, tt] = track_f0(speech, ws, voicingThresh, silenceThresh);

    % Convert to MFCC
    % Feature extraction (feature vectors as columns)
    [ MFCCs, FBEs, frames ] = mfcc( speech, fs, Tw, Ts, alpha, @hamming, [LF HF], M, C+1, L );

    % Pick out voiced frames
    f0_frames = convert_f0_to_frames(f0,tt,ws);

    f0_frames = f0_frames(3:end-2); % Need to change this if the frame durations are different...
    voiced_frames = find(f0_frames~=0);

    voiced_MFCCs = MFCCs(:,voiced_frames);
    
    ordering = randperm(size(voiced_MFCCs,2));
    voiced_MFCCs = voiced_MFCCs(:,ordering);
    nasal_M_MFCCs = horzcat(nasal_M_MFCCs,voiced_MFCCs);
end
% Female Nasal
for i=1:size(nasal_f_files,1)
    wav_file = strcat(nasal_dir,'/',nasal_f_files(i,:));  % input audio filename

    % Read speech samples, sampling rate and precision from file
    [ speech, fs] = audioread( wav_file );

    % Check if speech is mono, if not, convert it to mono
    if( min(size(speech))~=1 )speech = mean(speech,2);end;

    [f0, tt] = track_f0(speech, ws, voicingThresh, silenceThresh);

    % Convert to MFCC
    % Feature extraction (feature vectors as columns)
    [ MFCCs, FBEs, frames ] = mfcc( speech, fs, Tw, Ts, alpha, @hamming, [LF HF], M, C+1, L );

    % Pick out voiced frames
    f0_frames = convert_f0_to_frames(f0,tt,ws);

    f0_frames = f0_frames(3:end-2); % Need to change this if the frame durations are different...
    voiced_frames = find(f0_frames~=0);

    voiced_MFCCs = MFCCs(:,voiced_frames);
    
    ordering = randperm(size(voiced_MFCCs,2));
    voiced_MFCCs = voiced_MFCCs(:,ordering);
    nasal_F_MFCCs = horzcat(nasal_F_MFCCs,voiced_MFCCs);
end

normal_M_MFCCs = transpose(normal_M_MFCCs);
normal_F_MFCCs = transpose(normal_F_MFCCs);
normal_MFCCs = [normal_M_MFCCs;normal_F_MFCCs];
nasal_M_MFCCs = transpose(nasal_M_MFCCs);
nasal_F_MFCCs = transpose(nasal_F_MFCCs);
nasal_MFCCs = [nasal_M_MFCCs;nasal_F_MFCCs];

csvwrite('normal_M_MFCCs.csv', normal_M_MFCCs);
csvwrite('normal_F_MFCCs.csv', normal_F_MFCCs);
csvwrite('nasal_M_MFCCs.csv', nasal_M_MFCCs);
csvwrite('nasal_F_MFCCs.csv', nasal_F_MFCCs);