%% EEG Preprocessing Pipeline - README
%
% The pipeline covers these main stages:
% 1) Initial preprocessing: electrode setup, filtering, and resampling
% 2) Bad channel detection and interpolation using NEAR toolbox
% 3) Epoching based on event markers
% 4) Bad epoch detection and removal with FASTER plugin
% 5) ICA decomposition (with EEGLAB's runica)
% 6) Automatic artifact component rejection (MARA)
% 7) PSD-based channel cleaning and interpolation
% 8) Export of cleaned data for Lempel-Ziv complexity analysis
%
% Key dependencies:
% - EEGLAB toolbox (tested with version 2023.1)
% - NEAR plugin for bad channel detection
% - FASTER plugin for epoch rejection
% - MARA for artifact component identification
%
% Usage Notes:
% - Update all 'root' and toolbox paths at the start of each section as needed.
% - Ensure all required toolboxes and plugins are installed and added to MATLAB path.
% - The pipeline supports datasets recorded in two sites with specific channel layouts.
% - Scripts automatically handle site-specific electrode and event marker configurations.
% - All processed data are saved stepwise for quality control and reproducibility.

%% EEG Initial Preprocessing – Electrodes, Filtering, Resampling
clear all
close all

% ------------ USER SETTINGS (ADJUST AS NEEDED) -------------
root = '/path/to/project/';
city = 'Wwa'; % Use 'Krk' for Krakow dataset

% Paths (user should update these)
pathEEGLAB = '/path/to/eeglab/';
pathSPM = '/path/to/spm/';
pathElectrodeTemplate = [pathEEGLAB 'plugins/dipfit/standard_BEM/elec/standard_1020.elc'];
pathLoadData = [root 'RAW/' city '/'];
pathSaveData = [root 'I_Prepared/' city '/'];

addpath(genpath(pathEEGLAB))
addpath(genpath(pathSPM))
addpath(genpath(root))

cd(pathEEGLAB)
eeglab nogui

% List input EEG files depending on recording site
if strcmp(city, 'Wwa')
    list = dir([pathLoadData '*.vhdr']);
elseif strcmp(city, 'Krk')
    list = dir([pathLoadData '*.bdf']);
end

for s = 1:length(list)
    filename = strtrim(list(s).name);
    EEG = pop_fileio([pathLoadData filename]);

    % --- Reorder set of electrodes for Warsaw dataset (site-specific) ---
    if strcmp(city, 'Wwa')
        old_cap = [1:31, [1:7, 8,  9:29, 30, 31, 32] + 31, 64:69];
        new_cap = [1:31, [1:7, 32, 8:28, 31, 30, 29] + 31, 64:69];
        [~, old_to_new_order] = sort(new_cap);
        EEG.chanlocs = EEG.chanlocs(old_to_new_order);
    end

    % Load electrode location template
    EEG = pop_chanedit(EEG, 'lookup', pathElectrodeTemplate);

    % Remove unused/bad electrodes based on site
    if strcmp(city, 'Krk')
        EEG = pop_select(EEG, 'nochannel', [71 72 73 74 75 76 77 78 79 80]);
    end
    if strcmp(city, 'Wwa')
        EEG = pop_select(EEG, 'nochannel', [5 10 21 26]);
    elseif strcmp(city, 'Krk')
        EEG = pop_select(EEG, 'nochannel', [24 28 33 48 61]);
    end

    % --- Filtering ---
    if strcmp(city, 'Wwa')
        EEG = pop_eegfiltnew(EEG, [], 1, 3300, 1, [], 0);
        EEG = pop_eegfiltnew(EEG, [], 45, 1100, 0, [], 0);
    elseif strcmp(city, 'Krk')
        EEG = pop_eegfiltnew(EEG, [], 1, 6758, 1, [], 0);
        EEG = pop_eegfiltnew(EEG, [], 45, 2252, 0, [], 0);
    end

    % --- Resample to standard frequency ---
    if strcmp(city, 'Wwa')
        EEG = pop_resample(EEG, 250);
    elseif strcmp(city, 'Krk')
        EEG = pop_resample(EEG, 256);
    end

    % Save preprocessed EEG dataset
    if strcmp(city, 'Wwa')
        EEG = pop_saveset(EEG, 'filename', [filename(1:end-5) '.set'], 'filepath', pathSaveData);
    elseif strcmp(city, 'Krk')
        EEG = pop_saveset(EEG, 'filename', [filename(1:end-4) '.set'], 'filepath', pathSaveData);
    end
end

%% Bad Channel Removal
clear all
close all

% ------------ USER PATHS (ADJUST AS NEEDED) -------------
root = '/path/to/project/';
city = 'Wwa'; % Use 'Krk' for Krakow dataset
pathEEGLAB = '/path/to/eeglab/';
pathLoadData = [root 'I_Prepared/' city '/'];
pathSaveData = [root 'II_Chans_removed/' city '/'];

addpath(genpath(root))
addpath(genpath(pathEEGLAB))
cd(pathEEGLAB)
eeglab nogui

list = dir([pathLoadData '*.set']);

for s = 1:length(list)
    filename = list(s).name;
    SubjID = str2double(filename(1:3)); % Subject code from filename

    EEG = pop_loadset('filename', filename, 'filepath', pathLoadData);

    % --- EOG channel processing (site-specific) ---
    if strcmp(city, 'Krk')
        data_EOG_A12 = EEG.data(60:65, :);
        Initial_Chanlocs = EEG.chanlocs;
        EEG = pop_select(EEG, 'nochannel', [60 61 62 63 64 65]);
        No_EOG_Chanlocs = EEG.chanlocs;
        [EEG, red_chFlat, red_ch, yellow_ch, LOF_vec] = NEAR_getBadChannels(EEG, 1, 5, 0, [], [], 1, 1, [1 45], 1, 0.66, 2, 0);

        removed_channels = [red_chFlat yellow_ch];
        EEG = pop_select(EEG, 'nochannel', removed_channels);
        EEG = pop_interp(EEG, No_EOG_Chanlocs, 'spherical');
        EEG.data(60:65, :) = data_EOG_A12; % Restore EOG channels
        EEG.nbchan = 65;

    elseif strcmp(city, 'Wwa')
        data_EOG_A12 = EEG.data(60:65, :);
        Initial_Chanlocs = EEG.chanlocs;
        EEG = pop_select(EEG, 'nochannel', [60 61 62 63 64 65]);
        No_EOG_Chanlocs = EEG.chanlocs;
        [EEG, red_chFlat, red_ch, yellow_ch, LOF_vec] = NEAR_getBadChannels(EEG, 1, 5, 0, [], [], 1, 1, [1 45], 1, 0.66, 2.2, 0);

        removed_channels = [red_chFlat yellow_ch];
        EEG = pop_select(EEG, 'nochannel', removed_channels);
        EEG = pop_interp(EEG, No_EOG_Chanlocs, 'spherical');
        EEG.data(60:65, :) = data_EOG_A12; % Restore EOG channels
        EEG.nbchan = 65;
    end

    % Restore original electrode layout
    EEG.chanlocs = Initial_Chanlocs;

    % Save cleaned dataset
    EEG = pop_saveset(EEG, 'filename', filename, 'filepath', pathSaveData);

    % Collect information about removed channels
    Removed_chans = {};
    chans_indexes = removed_channels;
    for chan = 1:length(chans_indexes)
        Removed_chans = [Removed_chans, {No_EOG_Chanlocs(chans_indexes(chan)).labels}];
    end
    Chans{s,1} = SubjID;
    Chans{s,2} = Removed_chans;
    Chans{s,3} = length(Removed_chans);
end

% Save information on removed channels for future reference/publication
save([root 'Removed_Chans_' city '.mat'], 'Chans')


%% Epoching
clear all
close all

% ------------ USER SETTINGS (ADJUST AS NEEDED) -------------
root = '/path/to/project/';
city = 'Krk'; % Use 'Wwa' for Warsaw dataset

pathEEGLAB = '/path/to/eeglab/';
pathLoadData = [root 'II_Chans_removed/' city '/'];
pathSaveData = [root 'III_Epoched/' city '/'];
marker_names = {'S 10', 'S 11'}; % Marker names to be used
epoch_duration = 4; % Epoch duration (seconds)
max_seconds_after_S10 = 300;
max_seconds_after_S11 = 300; % Maximum recording length after S11

addpath(genpath(root))
addpath(genpath(pathEEGLAB))
cd(pathEEGLAB)
eeglab nogui

list = dir([pathLoadData '*.set']);

for s = 1:length(list)
    filename = list(s).name;
    SubjID = str2double(filename(1:3)); % Subject identifier

    EEG = pop_loadset('filename', filename, 'filepath', pathLoadData);

    % --- Standardize marker codes depending on recording site ---
    if strcmp(city, 'Krk')
        for i = 1:length(EEG.event)
            if EEG.event(i).type == 65290 || EEG.event(i).type == 10
                EEG.event(i).type = 'S 10';
            elseif EEG.event(i).type == 65291 || EEG.event(i).type == 11
                EEG.event(i).type = 'S 11';
            end
        end
    end

    % Remove unwanted events by type
    EEG = pop_selectevent(EEG, 'type', {'S 10', 'S 11'}, 'deleteevents', 'on', 'deleteepochs', 'off');

    % Find first occurrence of S10 and S11 markers
    event_labels = {EEG.event.type};
    S10_index = find(strcmp(event_labels, 'S 10'), 1);
    S11_index = find(strcmp(event_labels, 'S 11'), 1);

    % Force order: S10 then S11 if needed
    if S10_index > S11_index
        EEG.event(1).type = 'S 10';
        EEG.event(2).type = 'S 11';
    end

    % If markers missing, create them
    if isempty(S10_index) || isempty(S11_index)
        disp('Missing markers "S 10" or "S 11" in data.');
    end
    if isempty(S10_index)
       new_event = struct('type', 'S 10', 'duration', 0.25, 'timestamp', [], 'latency', 1, 'urevent', []);
       EEG.event = [EEG.event, new_event];
       event_labels = {EEG.event.type};
       S10_index = find(strcmp(event_labels, 'S 10'), 1);
    end

    % Keep only first occurrence of unique marker types
    [unique_labels, ~, idx] = unique(event_labels);
    unique_indices = cellfun(@(x) find(strcmp(event_labels, x), 1, 'first'), unique_labels, 'UniformOutput', false);
    unique_indices = cell2mat(unique_indices);
    EEG = pop_selectevent(EEG, 'type', unique_labels, 'deleteevents', 'on', 'deleteepochs', 'off');
    EEG.event = EEG.event(unique_indices);

    % Relabel indices after unique selection
    event_labels = {EEG.event.type};
    S10_index = find(strcmp(event_labels, 'S 10'), 1);
    S11_index = find(strcmp(event_labels, 'S 11'), 1);
    S10_latency = EEG.event(S10_index).latency;
    S11_latency = EEG.event(S11_index).latency;

    epoch_samples = epoch_duration * EEG.srate;

    % --- Add markers every 4 seconds between S10 and S11 ---
    start_sample = S10_latency;
    end_s10 = min(S10_latency + (max_seconds_after_S10 - 4) * EEG.srate, S11_latency);
    while start_sample + epoch_samples <= end_s10
        start_sample = start_sample + epoch_samples;
        if strcmp(city, 'Krk')
            new_event = struct('type', 'S 10', 'duration', 0.25, 'latency', start_sample, 'urevent', []);
        elseif strcmp(city, 'Wwa')
            new_event = struct('type', 'S 10', 'duration', 0.25, 'timestamp', [], 'latency', start_sample, 'urevent', []);
        end
        EEG.event = [EEG.event, new_event];
    end

    % --- Add markers every 4 seconds after S11, up to 300 seconds or end of recording ---
    start_sample = S11_latency;
    end_sample = min(S11_latency + (max_seconds_after_S11 - 4) * EEG.srate, EEG.pnts);
    while start_sample + epoch_samples <= end_sample
        start_sample = start_sample + epoch_samples;
        if strcmp(city, 'Krk')
            new_event = struct('type', 'S 11', 'duration', 0.25, 'latency', start_sample, 'urevent', []);
        elseif strcmp(city, 'Wwa')
            new_event = struct('type', 'S 11', 'duration', 0.25, 'timestamp', [], 'latency', start_sample, 'urevent', []);
        end
        EEG.event = [EEG.event, new_event];
    end

    % Epoch data based on markers
    EEG = pop_epoch(EEG, {'S 10', 'S 11'}, [0 epoch_duration], 'epochinfo', 'yes');

    % Save epoched dataset
    EEG = pop_saveset(EEG, 'filename', [filename(1:end-4) '_epoched.set'], 'filepath', pathSaveData);
end


%% Bad Epoch Detection & Removal (FASTER)
clear all
close all

% ------------ USER SETTINGS (ADJUST AS NEEDED) -------------
root = '/path/to/project/';
city = 'Wwa'; % Use 'Krk' for Krakow dataset

pathEEGLAB = '/path/to/eeglab/';
pathLoadData = [root 'III_Epoched/' city '/'];
pathSaveData = [root 'IV_Epochs_removed/' city '/'];

addpath(genpath(root))
addpath(genpath(pathEEGLAB))
cd(pathEEGLAB)
eeglab nogui

list = dir([pathLoadData '*.set']);

for s = 1:length(list)
    filename = list(s).name;
    SubjID = str2double(filename(1:3));
    EEG = pop_loadset('filename', filename, 'filepath', pathLoadData);

    NrEp(s,1) = SubjID;
    NrEp(s,2) = EEG.trials;

    % -- Remove EOG channels and detect bad epochs using FASTER --
    data_EOG_A12 = EEG.data(60:65,:,:);
    Initial_Chanlocs = EEG.chanlocs;
    EEG = pop_select(EEG, 'nochannel', [60 61 62 63 64 65]);
    epochs_vals = epoch_properties(EEG, 1:size(EEG.data,1));  % FASTER plugin required
    thresholds = mean(epochs_vals) + 2.5 * std(epochs_vals);
    above_threshold = epochs_vals > thresholds;
    to_reject = any(above_threshold, 2);

    % Restore EOG channels
    EEG.data(60:65,:,:) = data_EOG_A12;
    EEG.nbchan = 65;
    EEG.chanlocs = Initial_Chanlocs;

    % Remove flagged epochs
    EEG = pop_rejepoch(EEG, to_reject, 0);

    NrEp(s,3) = EEG.trials;

    % Save cleaned data
    EEG = pop_saveset(EEG, 'filename', filename, 'filepath', pathSaveData);
    clear EpVec
end

save([pathSaveData 'removed_EPs.mat'], 'NrEp')

%% ICA Decomposition
clear all
close all

root = '/path/to/project/';
city = 'Krk'; % Use 'Wwa' for Warsaw dataset

pathEEGLAB = '/path/to/eeglab/';
pathLoadData = [root 'IV_Epochs_removed/' city '/'];
pathSaveData = [root 'V_ICA/' city '/'];

addpath(genpath(root))
addpath(genpath(pathEEGLAB))
cd(pathEEGLAB)
eeglab nogui

list = dir([pathLoadData '*.set']);

for s = 1:length(list)
    filename = list(s).name;
    EEG = pop_loadset('filename', filename, 'filepath', pathLoadData);

    % Run ICA decomposition using EEGLAB
    EEG = pop_runica(EEG, 'extended', 1, 'interupt', 'on', 'pca', 36);

    % Save ICA results
    EEG = pop_saveset(EEG, 'filename', filename, 'filepath', pathSaveData);
    clear EEG filename
end

%% MARA Component Removal
clear all
close all
root = '/path/to/project/';
city = 'Wwa'; % Use 'Krk' for Krakow dataset
pathEEGLAB = '/path/to/eeglab/';
method_list = {'MARA'}; % MARA only

addpath(genpath(root))
addpath(genpath(pathEEGLAB))
cd(pathEEGLAB)
eeglab nogui

for method_idx = 1:numel(method_list)
    method = method_list{method_idx};
    pathLoadData = [root 'V_ICA/' city '/'];
    pathSaveData = [root 'VI_After_' method '/' city '/'];
    list = dir([pathLoadData '*.set']);
    for s = 1:length(list)
        filename = list(s).name;
        SubjID = str2double(filename(1:3));
        ALLEEG = [];
        CURRENTSET = 0;
        EEG = pop_loadset('filename', filename, 'filepath', pathLoadData);
        EEG = eeg_checkset(EEG);

        % MARA processing
        [ALLEEG, EEG] = processMARA(ALLEEG, EEG, CURRENTSET, [0,0,0,0,0]);

        rejComps = find(EEG.reject.gcompreject);
        save([pathSaveData filename(1:end-4)], 'rejComps')

        EEG = pop_subcomp(EEG, rejComps, 0);
        EEG = pop_saveset(EEG, 'filename', filename, 'filepath', pathSaveData);

        Rejected_Comps{s,1} = SubjID;
        Rejected_Comps{s,2} = length(rejComps);
        save([pathSaveData 'Removed_Comps_' method '.mat'], 'Rejected_Comps')
        clear EEG filename rejComps
    end
    clear Rejected_Comps
end

%% PSD-Based Channel Cleaning
clear all
close all

% ------------ USER SETTINGS (ADJUST AS NEEDED) -------------
root = '/path/to/project/';
city = 'Krk'; % Use 'Wwa' for Warsaw dataset

pathEEGLAB = '/path/to/eeglab/';
method_list = {'MARA'}; % Method name to match earlier processing
Ref = 'A1/A2'; % Choose 'A1/A2' or 'Mastoids' as reference
threshold_factor = 4; % Outlier threshold (number of SDs above mean)

addpath(genpath(root))
addpath(genpath(pathEEGLAB))
cd(pathEEGLAB)
eeglab nogui

for method_idx = 1:numel(method_list)
    method = method_list{method_idx};

    % Set folder for loading/saving data (reference scheme included in foldername)
    if strcmp(Ref,'A1/A2')
        pathLoadData = [root 'VI_After_' method '/' city '/'];
        pathSaveData = [root 'VII_Cleaned_' method '_A1A2/' city '/'];
    else
        pathLoadData = [root 'VI_After_' method '/' city '/'];
        pathSaveData = [root 'VII_Cleaned_' method '/' city '/'];
    end
    if ~exist(pathSaveData, 'dir')
        mkdir(pathSaveData);
    end

    list = dir([pathLoadData '*.set']);
    for s = 1:length(list)
        filename = list(s).name;
        SubjID = str2double(filename(1:3));
        ALLEEG = []; CURRENTSET = 0;
        EEG = pop_loadset('filename', filename, 'filepath', pathLoadData);
        EEG = eeg_checkset(EEG);

        % Reference selection and EOG channel cleaning
        if strcmp(city,'Wwa')
            if strcmp(Ref,'A1/A2')
                EEG = pop_select(EEG, 'nochannel',{'LH_EOG', 'PH_EOG', 'V1EOG','V2EOG'});
                if SubjID == 40
                    EEG.data(60, :) = EEG.data(61, :);
                end
                EEG = pop_reref(EEG, [60 61]);
            else
                EEG = pop_select(EEG, 'nochannel',{'LH_EOG', 'PH_EOG', 'V1EOG','V2EOG','A1','A2'});
                EEG = pop_reref(EEG, []);
            end
        elseif strcmp(city,'Krk')
            if strcmp(Ref,'A1/A2')
                EEG = pop_select(EEG, 'nochannel',{'EXG1', 'EXG2','EXG3', 'EXG4'});
                EEG = pop_reref(EEG, [60 61]);
            else
                EEG = pop_select(EEG, 'nochannel',{'EXG1', 'EXG2','EXG3', 'EXG4','EXG5', 'EXG6'});
                EEG = pop_reref(EEG, []);
            end
        end

        % Compute PSD sum per channel and flag outliers
        num_channels = size(EEG.data, 1);
        psd_sum = zeros(num_channels, 1);
        for chan = 1:num_channels
            signal = EEG.data(chan, :);
            [pxx, ~] = pwelch(signal, [], [], [], EEG.srate);
            psd_sum(chan) = sum(pxx);
        end
        mean_psd = mean(psd_sum);
        std_psd = std(psd_sum);
        outliers = find(psd_sum > mean_psd + threshold_factor * std_psd);

        % Remove outlier channels and interpolate
        if ~isempty(outliers)
            disp(['Removed channels for subject ' num2str(SubjID) ': ', EEG.chanlocs(outliers').labels]);
            chanlocs_before = EEG.chanlocs;
            EEG = pop_select(EEG, 'nochannel', outliers);
            EEG = pop_interp(EEG, chanlocs_before, 'spherical');
        end

        % Save cleaned dataset
        EEG = pop_saveset(EEG, 'filename', filename, 'filepath', pathSaveData);

        % Save removed channel info
        Removed_Channels{s, 1} = SubjID;
        Removed_Channels{s, 2} = outliers;
    end

    save([pathSaveData 'Removed_Channels_PSD_' method '.mat'], 'Removed_Channels');
    clear Removed_Channels
end

%% Export Data for Lempel-Ziv (LZ) Analysis
clear all
close all

root = '/path/to/project/';
city = 'Wwa'; % Use 'Krk' for Krakow dataset
pathEEGLAB = '/path/to/eeglab/';
pathLoadData = [root 'VIII_Cleaned_MARA_A1A2/' city '/'];
pathSaveData = [root 'Exported_for_LZ/' city '/'];

addpath(genpath(root))
addpath(genpath(pathEEGLAB))
cd(pathEEGLAB)
eeglab nogui

list = dir([pathLoadData '*.set']);

for s = 1:length(list)
    filename = strtrim(list(s).name);
    SubjID = str2double(filename(1:3));
    EEG = pop_loadset('filename', filename, 'filepath', pathLoadData);
    EEGCond = cell(1,2);

    % Split and save data for each event condition (S 10, S 11)
    for cond = 10:11
        clear epVec
        for ep = 1:EEG.trials
            compare = strcmp(EEG.epoch(ep).eventtype, ['S ' num2str(cond)]);
            epVec(ep) = compare(1);
        end
        EEGcond = pop_select(EEG, 'trial', find(epVec));
        EEGCond{cond}.data      = EEGcond.data;
        EEGCond{cond}.chanlocs  = EEGcond.chanlocs;
        EEGCond{cond}.srate     = EEGcond.srate;
        EEGCond{cond}.EpPosition= find(epVec);
    end

    save([pathSaveData 'Data_' num2str(SubjID) '_mara'], 'EEGCond');
end
