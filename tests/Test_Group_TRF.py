import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import mne
from scipy import signal
from scipy.stats import *
from scipy.signal import *
from scipy.fft import fft
from spyeeg.models.TRF import TRFEstimator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tqdm import tqdm
from pathlib import Path
from CORSICA.artifact_reduction import ci_artifact_reduction

#inspction 
import inspect
from CORSICA.artifact_reduction import ci_artifact_reduction

print(inspect.getsourcefile(ci_artifact_reduction))
print(inspect.getsource(ci_artifact_reduction))

def preprocess_eeg(raw_cleaned, fs_output, corner_freq_l, corner_freq_h):
    """
    Preprocesses an EEG signal by applying a lowpass filter, resampling, and then a highpass filter.

    Input:
    - raw_cleaned: mne.io.Raw object containing the EEG data
    - fs_output: target sampling frequency in Hz
    - corner_freq_l: lower cutoff frequency for the highpass filter in Hz
    - corner_freq_h: upper cutoff frequency for the lowpass filter in Hz

    Returns:
    - raw_cleaned: mne.io.Raw object with the EEG data filtered (bandpass) 
                   and resampled to fs_output
    """
    raw_cleaned.filter(None, corner_freq_h, verbose=False)
    raw_cleaned.resample(fs_output, verbose=False)
    raw_cleaned.filter(corner_freq_l, None, verbose = False)
    return raw_cleaned

def calculate_env(audio, wav_freq, output_freq, corner_freq_l, corner_freq_h):
    """
    Calculates the normalized speech envelope of an audio signal.

    Input:
    - audio: 1D numpy array containing the audio signal
    - wav_freq: sampling frequency of the input audio in Hz
    - output_freq: target sampling frequency after resampling in Hz
    - corner_freq_l: lower cutoff frequency of the bandpass filter in Hz
    - corner_freq_h: upper cutoff frequency of the bandpass filter in Hz

    Returns:
    - env: 1D numpy array containing the filtered, resampled,
           and z-score normalized audio envelope
    """
    # calculate envelope
    env = np.abs(hilbert(audio))
    env = mne.filter.filter_data(env, wav_freq, l_freq=corner_freq_l, h_freq=corner_freq_h, verbose=False)
    down = wav_freq/ output_freq
    env = mne.filter.resample(env, down=down, verbose=False)
    env = zscore(env, nan_policy='raise')
    return env

base_path = "/Users/leonierichter/Documents/2100_Work/2026_Uni_Chair_Senory_in_Neuroengineering/2026.03.27_Code_und_Daten"
base_path = Path(base_path)
SAVE_COEFS = True
PLOT_METHOD = False
CALC_SCORES = True
NULL_MODEL = False

tmin = -0.5
tmax = 0.8
alpha = [1]

#eeg_path = base_path / "simple_prepro/{sub_id}/{sub_id}_Elbenwald_FL_6.set"
eeg_prepro_path = base_path / "simple_prepro"


montage_file = base_path / "CACS-32_NO_REF.bvef"

output_dir = base_path / "output_data_group"
coef_path = output_dir / "coef"
coef_path = Path(coef_path)
coef_path.mkdir(parents=True, exist_ok=True)

sub_ids = list(np.r_[102:115, 116, 118:126, 127:129, 130])

stimuli_trigger_codes = {'Elbenwald_FR_1.wav': '111',
                         'Elbenwald_FL_2.wav': '112',
                         'Elbenwald_FR_3.wav': '113',
                         'Elbenwald_FL_4.wav': '114',
                         'Elbenwald_FR_5.wav': '215',
                         'Elbenwald_FL_6.wav': '216',
                         'Elbenwald_FR_7.wav': '217',
                         'Elbenwald_FL_8.wav': '218',
                         'Elbenwald_FR_9.wav': '219',
                         'Elbenwald_FL_10.wav': '210',
                         'Polarnacht_Focus_FR_1.wav': '121',
                         'Polarnacht_Focus_FL_2.wav': '122',
                         'Polarnacht_Focus_FR_3.wav': '123',
                         'Polarnacht_Focus_FL_4.wav': '124',
                         'Polarnacht_FR_5.wav': '225',
                         'Polarnacht_FL_6.wav': '226',
                         'Polarnacht_FR_7.wav': '227',
                         'Polarnacht_FL_8.wav': '228',
                         'Polarnacht_FR_9.wav': '229',
                         'Polarnacht_FL_10.wav': '220'}

stimulus_names = stimuli_trigger_codes.keys()
stimulus_names = list(stimulus_names)

snr_threshold= 27
fs_eeg=1000

fs_output = 128 # for trf and envelope calculation
corner_freq_l = 1
corner_freq_h = 8


if __name__ == "__main__":
    all_coefs =[] # alle Koeffizienten über alle subjects 
    all_coefs_dis = []

    all_scores = []
    all_scores_dis = []

    all_exclude = []
    all_comps = 0
    all_snr = []

    snr_sub = []
    exclude_sub = []

    coef_sub = []
    coef_sub_dis = []

    scores_sub = []
    scores_sub_dis =[]

    for sub_id in tqdm(sub_ids): #über alle subjecte gehen 
        snr_sub = []
        exclude_sub = []

        coef_sub = []
        coef_sub_dis = []

        scores_sub = []
        scores_sub_dis =[]

        for wav_name in stimulus_names: #über alle Stimuli gehen 
            stimulus_name = wav_name[:-4]  # crop '.wav' ending
            # check if competing or single speaker
            code = stimuli_trigger_codes[wav_name]

            competing = str(code).startswith('2')  # true, if competing speaker scenario

            #für was brauche ich das #NOTE
            #if not competing:
            #    continue
            
            # EEG
            eeg_path = os.path.join(eeg_prepro_path, str(sub_id), (str(sub_id) + "_" + str(stimulus_name) + ".set"))
            raw = mne.io.read_raw_eeglab(eeg_path, preload=True)

            #AUDIO
            aux1 = raw.get_data()[31, :]
            aux2 = raw.get_data()[32, :]

            if '_FR_' in stimulus_name:
                if sub_id == 103:
                    attended = aux1
                    distractor = aux2
                else:
                    # focus side right
                    attended = aux2
                    distractor = aux1
            else:
                if sub_id == 103:
                    attended = aux2
                    distractor = aux1
                else:
                    # focus side left
                    attended = aux1
                    distractor = aux2
            
            
            cleaned_eeg, raw_cleaned, metadata_out = ci_artifact_reduction(raw, sub_id, wav_name, output_dir, fs_eeg, attended,distraction_audio = distractor, snr_threshold=snr_threshold, plot=True, metadata=True)
            
            # calculate envelope
            env = calculate_env(attended, fs_eeg, fs_output, corner_freq_l, corner_freq_h)
            if competing == True:
                distractor_env = calculate_env(distractor, fs_eeg, fs_output, corner_freq_l, corner_freq_h)

            channel_names = raw_cleaned.ch_names
            raw_cleaned.drop_channels([channel_names[31], channel_names[32]])  # Aux1 and Aux2
            # load montage file
            montage = mne.channels.read_custom_montage(montage_file)
            raw_cleaned.set_montage(montage)

            # preprocess eeg
            raw_cleaned = preprocess_eeg(raw_cleaned, fs_output, corner_freq_l,corner_freq_h)
            info = raw_cleaned.info

            eeg = raw_cleaned.get_data()

            #TRF
            trf = TRFEstimator(tmin=tmin, tmax=tmax, srate=fs_output, alpha=alpha, alpha_feat=False)
            trf.fit(np.expand_dims(env, axis=1), eeg.T)
            coefs_attended = trf.get_coef()[:, 0, :, :].T
            coef_sub.append(coefs_attended)

            if competing:
                # additionally calculate TRF for distractor
                trf_dis = TRFEstimator(tmin=tmin, tmax=tmax, srate=fs_output, alpha=alpha, alpha_feat=False)
                trf_dis.fit(np.expand_dims(distractor_env, axis=1), eeg.T)
                coefs_distractor = trf_dis.get_coef()[:, 0, :, :].T
                coef_sub_dis.append(coefs_distractor)

            #snr_sub.append(metadata_out["All SNR values"])
            #exclude_sub.append(metadata_out["Indices of excluded ICs"])
        
        #all_exclude.append(exclude_sub)
        #all_snr.append(snr_sub)
        single_sub_path = os.path.join(base_path, 'single_sub')
        if not os.path.exists(single_sub_path):
            os.makedirs(single_sub_path)

        coefs_sub = np.array(coef_sub)[:, 0, :, :]

        coefs_avg = np.mean(coefs_sub, axis=0)
        evoked = mne.EvokedArray(coefs_avg, info, tmin)
        trf = evoked.plot_joint(title=f'Subject: {sub_id} Correlation Approach (SNR threshold: {snr_threshold}), attended, alpha: {alpha}', show=False)
        trf.savefig(os.path.join(single_sub_path, f'{sub_id}_corr_approach_snr_{int(snr_threshold)}_attended_trf'))
        all_coefs.append(coefs_sub)

        # distractor
        coefs_sub_dis = np.array(coef_sub_dis)[:, 0, :, :]
        coefs_avg_dis = np.mean(coefs_sub_dis, axis=0)
        evoked_dis = mne.EvokedArray(coefs_avg_dis, info, tmin)
        trf_dis = evoked_dis.plot_joint(title=f'Subject: {sub_id} Correlation Approach (SNR threshold: {snr_threshold}), distractor, alpha: {alpha}', show=False)
        trf_dis.savefig(os.path.join(single_sub_path, f'{sub_id}_corr_approach_snr_{int(snr_threshold)}_distractor_trf'))
        all_coefs_dis.append(coefs_sub_dis)


        np.save(os.path.join(coef_path, f'{sub_id}_snr_{snr_threshold}_attended_coefs_sub.npy'), coefs_sub)
        np.save(os.path.join(coef_path, f'{sub_id}_snr_{snr_threshold}_distractor_coefs_sub_sub.npy'), coefs_sub_dis)
        info.save(coef_path / "{sub_id}_raw_cleaned_info_sub.fif", overwrite=True)

    all_coefs = np.array(all_coefs)
    all_coefs_dis = np.array(all_coefs_dis)
    #all_exclude = np.array((all_exclude))
    #all_snr = np.array(all_snr)

    np.save(os.path.join(coef_path, f'snr_{snr_threshold}_attended_all_coefs.npy'), all_coefs)
    np.save(os.path.join(coef_path, f'snr_{snr_threshold}_distractor_all_coefs_dis.npy'), all_coefs_dis)

    info.save(coef_path / "raw_cleaned_info_all.fif", overwrite=True)
    all_coefs_avg = np.mean(all_coefs, axis = 0) # over all sub
    evoked_all = mne.EvokedArray(all_coefs_avg[0], info, tmin)
    trf = evoked_all.plot_joint(title=f'All Subjects, Correlation approach, SNR: {snr_threshold}, attended, alpha: {alpha}')#, exclude=['TP10', 'P8'])
    trf.savefig(os.path.join(base_path, f'corr_approach_snr_{int(snr_threshold)}_attended_trf'))
    time_points_to_plot = [0.024, 0.104, 0.184]
    topo = evoked_all.plot_topomap(times=time_points_to_plot, ch_type='eeg')
    topo.savefig(os.path.join(base_path, f'topoplots_corr_approach_snr_{int(snr_threshold)}_attended.svg'), format='svg')


    all_coefs_avg_dis = np.mean(all_coefs_dis, axis = 0) # over all sub
    evoked_all_dis = mne.EvokedArray(all_coefs_avg_dis[0], info, tmin)
    trf_dis = evoked_all_dis.plot_joint(title=f'All Subjects, Correlation approach, SNR: {snr_threshold}, distractor, alpha: {alpha}')#,exclude=['TP10', 'P8'])
    trf_dis.savefig(os.path.join(base_path, f'corr_approach_snr_{int(snr_threshold)}_distractor_trf'))
    time_points_to_plot = [0.024, 0.104, 0.184]
    topo_dis = evoked_all_dis.plot_topomap(times=time_points_to_plot, ch_type='eeg')
    topo_dis.savefig(os.path.join(base_path, f'topoplots_corr_approach_snr_{int(snr_threshold)}_distractor.svg'), format='svg')


