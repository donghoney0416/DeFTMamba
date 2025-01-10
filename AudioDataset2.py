import os
import glob
import argparse
import configparser as CP
import multiprocessing
import pyroomacoustics as pra
import math
import random
from pydub import AudioSegment
import multiprocessing
from itertools import repeat
import csv
import numpy as np
import librosa
import webrtcvad
import scipy
import scipy.io
import scipy.signal
import soundfile
import pandas
import warnings
import copy
from copy import deepcopy
from collections import namedtuple
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from matplotlib import animation
import webrtcvad
os.environ["CUDA_VISIBLE_DEVICES"] = str('0')
import gpuRIR
import tqdm
import torch

# num_processes = multiprocessing.cpu_count()
# %% Util functions
def acoustic_power(s):
    w = 640  # Window size for silent detection
    o = 320  # Window step for silent detection
    # Window the input signal
    s = np.ascontiguousarray(s)
    sh = (s.size - w + 1, w)
    st = s.strides * 2
    S = np.lib.stride_tricks.as_strided(s, strides=st, shape=sh)[0::o]
    window_power = np.mean(S ** 2, axis=-1)
    th = 0.01 * window_power.max()  # Threshold for silent detection
    return np.mean(window_power[np.nonzero(window_power > th)])

def cart2sph(cart):
    xy2 = cart[:,0]**2 + cart[:,1]**2
    sph = np.zeros_like(cart)
    sph[:,0] = np.sqrt(xy2 + cart[:,2]**2)
    sph[:,1] = np.arctan2(np.sqrt(xy2), cart[:,2]) # Elevation angle defined from Z-axis down
    sph[:,2] = np.arctan2(cart[:,1], cart[:,0])    # Azimuth
    return sph

def sph2cart(spherical_coords):
    radius = spherical_coords[2]
    elevation_deg = spherical_coords[1]
    azimuth_deg = spherical_coords[0]
    # Convert degrees to radians
    elevation = np.radians(elevation_deg)
    azimuth = np.radians(azimuth_deg)
    # Calculate Cartesian coordinates
    x = radius * np.cos(elevation) * np.cos(azimuth)
    y = radius * np.cos(elevation) * np.sin(azimuth)
    z = radius * np.sin(elevation)
    return np.column_stack((x, y, z))

def load_audio(params):
    audio_folder = './0_Preprocessed'
    sub_folder_list = [f.path for f in os.scandir(audio_folder) if f.is_dir()]
    # sub_folder_list = ['./0_Preprocessed/Male_speech', './0_Preprocessed/Female_speech']
    # class_list = ['Telephone', 'Water_tap', 'Musical_instrument', 'Knock', 'Walk_and_footsteps', 'Domestic_sound', 'Laughter', 'Door', 'Male_speech', 'Music', 'Female_speech', 'Bell', 'Clapping']
    class_list = ['Telephone_ASA', 'Water_tap_ASA', 'Musical_instrument_ASA', 'Knock_ASA', 'Walk_and_footsteps_ASA', 'Domestic_sound_ASA', 'Laughter_ASA', 'Door_ASA', 'Male_speech_ASA', 'Music_ASA', 'Female_speech_ASA', 'Bell_ASA', 'Clapping_ASA']
    class_num = random.randint(0, len(class_list)-1)
    sub_folder = sub_folder_list[class_num]

    # only for dev.csv
    audio_files = [os.path.join(root, file) for root, dirs, files in os.walk(sub_folder) for file in files if file.endswith('.wav')]
    audio_files = sorted(audio_files)

    # only for train set
    idx = random.randint(0, len(audio_files)-1)
    audio, sr = librosa.load(audio_files[idx])
    if sr != params['fs']:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=params['fs'])

    # set audio length to 4 seconds
    if len(audio) > params['sec'] * params['fs']:
        start = random.randint(0, len(audio) - int(params['sec'] * params['fs']))
        audio = audio[start:start+int(params['sec'] * params['fs'])]
    if len(audio) < params['sec'] * params['fs']:
        padding = params['sec'] * params['fs'] - len(audio)
        left = random.randint(0, padding)        
        right = padding - left
        audio = np.pad(audio, (left, right), 'constant')

    return audio, class_num

def AAD(s):
    window_length = 1600
    num_windows = 40
    threshold_factor = 0.01
    # Power 계산
    power = np.square(s)
    # Window 단위로 계산된 max power 구하기
    max_power_per_window = np.array([np.max(power[i * window_length: (i + 1) * window_length]) for i in range(num_windows)])
    # Threshold 계산
    threshold = threshold_factor * np.max(max_power_per_window)
    # Activity matrix 생성
    activity_matrix = np.array([True if max_power >= threshold else False for max_power in max_power_per_window])
    return activity_matrix.astype(int)

class RIR_generator(Dataset):
    def __init__(self, params):
        self.nb_points = params['sec'] * 10
        self.noiseDataset = NoiseDataset(noise_path='DataAugmentation/datasets/rir_datasets/source_data/tau/TAU-SNoise_DB')
        self.params = params

    def __getitem__(self, idx):
		# source signals
        values = [2, 3, 4, 5]
        weights = [0.1, 0.2, 0.3, 0.4]
        source_num = random.choices(values, weights)[0]
        source_signals = np.zeros((self.params['sec']*self.params['fs'], source_num))
        source_class = np.zeros(source_num)
        for source_idx in range(source_num):
            source_signals[:, source_idx], source_class[source_idx] = load_audio(self.params)

        # Set RT60
        rt60 = self.params['room_rt60_min'] + np.random.rand() * (self.params['room_rt60_max'] - self.params['room_rt60_min'])
        # Set SNR
        SNR = self.params['SNR_min'] + np.random.rand() * (self.params['SNR_max'] - self.params['SNR_min'])
        # Set room geometry
        width = self.params['room_width_min'] + np.random.rand() * (self.params['room_width_max'] - self.params['room_width_min'])
        length = self.params['room_length_min'] + np.random.rand() * (self.params['room_length_max'] - self.params['room_length_min'])
        height = self.params['room_height_min'] + np.random.rand() * (self.params['room_height_max'] - self.params['room_height_min'])
        room_sz = np.array([width, length, height])

        # Set microphone array location
        axx = width / 2
        ayy = length / 2
        azz = height / 2
        array_center_location = np.array([axx, ayy, azz])
        mic_pos = array_center_location + np.array([sph2cart([45, 35, 0.042]), sph2cart([-45, -35, 0.042]), sph2cart([135, -35, 0.042]), sph2cart([-135, 35, 0.042])])
        orV_rcv = np.array([sph2cart([45, 35, 1]), sph2cart([-45, -35, 1]), sph2cart([135, -35, 1]), sph2cart([-135, 35, 1])])
        
        # noise signal
        noise_signal = self.noiseDataset.get_random_noise(mic_pos)

        # Trajectory points
        src_pos_min = np.array([0, 0, 0]) + np.array([self.params['room_offset_inside'], self.params['room_offset_inside'], self.params['room_offset_inside']])
        src_pos_max = room_sz - np.array([self.params['room_offset_inside'], self.params['room_offset_inside'], self.params['room_offset_inside']])
        timestamps = np.arange(self.nb_points) * self.params['sec'] / self.nb_points

        traj_pts = np.zeros((self.nb_points, 3, source_num))
        SEL_pts = np.zeros((self.nb_points, 2, source_num))
        SED_pts = np.zeros((self.nb_points, source_num))

        for source_idx in range(source_num):
            src_pos_ini = src_pos_min + np.random.random(3) * (src_pos_max - src_pos_min)
            # src_pos_end = src_pos_min + np.random.random(3) * (src_pos_max - src_pos_min)             # Random movement
            src_pos_end = src_pos_ini                                                                   # source is stationary

            Amax = np.min(np.stack((src_pos_ini - src_pos_min, src_pos_max - src_pos_ini, src_pos_end - src_pos_min, src_pos_max - src_pos_end)), axis=0)
            A = np.random.random(3) * np.minimum(Amax, 1)  # Oscilations with 1m as maximum in each axis
            w = 2 * np.pi / self.nb_points * np.random.random(3) * 2  # Between 0 and 2 oscilations in each axis

            traj_pts[:, :, source_idx] = np.array([np.linspace(i, j, self.nb_points) for i, j in zip(src_pos_ini, src_pos_end)]).transpose()
            traj_pts[:, :, source_idx] += A * np.sin(w * np.arange(self.nb_points)[:, np.newaxis])
            if np.random.random(1) < 0.25:
                traj_pts[:, :, source_idx] = np.ones((self.nb_points, 1)) * src_pos_ini
            SEL_pts[:, :, source_idx] = np.rad2deg(cart2sph(traj_pts[:, :, source_idx] - array_center_location)[:, 1:3])

        # Room prameters
        abs_weights = [0.8]*5+[0.5]
        beta = gpuRIR.beta_SabineEstimation(room_sz, rt60, abs_weights=abs_weights)
        Tdiff = gpuRIR.att2t_SabineEstimator(12, rt60)  # Use ISM until the RIRs decay 12dB
        Tmax = gpuRIR.att2t_SabineEstimator(40, rt60)  # Use diffuse model until the RIRs decay 40dB
        nb_img = gpuRIR.t2n(Tdiff, room_sz)

        RIRs_sources = []
        mic_signals_sources = []
        dp_RIRs_sources = []
        dp_mic_signals_sources = []

        for source_idx in range(source_num):
            # Reverberant RIR
            RIRs = gpuRIR.simulateRIR(room_sz, beta, traj_pts[:, :, source_idx], mic_pos, nb_img, Tmax, self.params['fs'], Tdiff=Tdiff, orV_rcv=None, mic_pattern='omni')
            mic_sig = gpuRIR.simulateTrajectory(source_signals[:, source_idx], RIRs, timestamps=timestamps, fs=self.params['fs'])
            mic_sig = mic_sig[:self.params['sec']*self.params['fs'], :]
            # Direct RIR
            dp_RIRs = gpuRIR.simulateRIR(room_sz, beta, traj_pts[:, :, source_idx], mic_pos, [1, 1, 1], 0.1, self.params['fs'], orV_rcv=None, mic_pattern='omni')
            dp_mic_sig = gpuRIR.simulateTrajectory(source_signals[:, source_idx], dp_RIRs, timestamps=timestamps, fs=self.params['fs'])
            dp_mic_sig = dp_mic_sig[:self.params['sec']*self.params['fs'], :]

            RIRs_sources += [RIRs]
            mic_signals_sources += [mic_sig]
            dp_RIRs_sources += [dp_RIRs]
            dp_mic_signals_sources += [dp_mic_sig]
            # SED
            SED_pts[:, source_idx] = AAD(mic_sig[:, 0])

        # RIRs_sources = np.array(RIRs_sources).transpose(1, 2, 3, 0)  # (npoints,nch,nsamples,nsources)
        mic_signals_sources = np.array(mic_signals_sources).transpose(1, 2, 0)  # (nsamples, nch, nsources)
        dp_mic_signals_sources = np.array(dp_mic_signals_sources).transpose(1, 2, 0)

        # Add Noise
        mic_signals = np.sum(mic_signals_sources, axis=2)  # (nsamples, nch)
        dp_mic_signals = np.sum(dp_mic_signals_sources, axis=2)
        ac_pow = np.mean([acoustic_power(dp_mic_signals[:, i]) for i in range(dp_mic_signals_sources.shape[1])])
        ac_pow_noise = np.mean([acoustic_power(noise_signal[:, i]) for i in range(noise_signal.shape[1])])
        noise_signal = np.sqrt(ac_pow / 10 ** (SNR / 10)) / np.sqrt(ac_pow_noise) * noise_signal
        mic_signals += noise_signal[0:self.params['sec']*self.params['fs'], :]

        return mic_signals, mic_signals_sources, dp_mic_signals_sources,  SEL_pts, SED_pts, source_class


class NoiseDataset():
	def __init__(self, T=4, fs=16000, nmic=4, noise_type='diffuse', noise_path='TAU-SNoise_DB', c=343.0):
		self.T = T
		self.fs= fs
		self.nmic = nmic
		self.noise_type = noise_type # ? 'diffuse' and 'real_world' cannot exist at the same time
		# self.mic_pos = mic_pos # valid for 'diffuse'
		self.noie_path = noise_path # valid for 'diffuse' and 'real-world'
		if noise_path != None:
			_, self.path_set = self._exploreCorpus(noise_path, 'wav')
		self.c = c

	def get_random_noise(self, mic_pos=None):
		# noise_type = self.noise_type.getValue()

		if self.noise_type == 'spatial_white':
			noise_signal = self.gen_Gaussian_noise(self.T, self.fs, self.nmic)

		elif self.noise_type == 'diffuse':
			idx = random.randint(0, len(self.path_set)-1)
			noise, fs = soundfile.read(self.path_set[idx])
			noise = noise[:, 0] # single-channel noise
			if fs != self.fs:
				#noise = librosa.resample(noise, orig_sr = fs, target_sr = self.fs)
				noise= scipy.signal.resample_poly(noise, up=self.fs, down=fs)

			nsample_desired = int(self.T * self.fs * self.nmic)
			noise_copy = copy.deepcopy(noise)
			nsample = noise.shape[0]
			while nsample < nsample_desired:
				noise_copy = np.concatenate((noise_copy, noise), axis=0)
				nsample = noise_copy.shape[0]

			st = random.randint(0, nsample - nsample_desired)
			ed = st + nsample_desired
			noise_copy = noise_copy[st:ed]

			noise_signal = self.gen_diffuse_noise(noise_copy, self.T, self.fs, mic_pos, c=self.c)
		elif self.noise_type == 'real_world': # the array topology should be consistent
			idx = random.randint(0, len(self.path_set)-1)
			noise, fs = soundfile.read(self.path_set[idx])
			nmic = noise.shape[-1]
			if nmic != self.nmic:
				raise Exception('Unexpected number of microphone channels')
			if fs != self.fs:
				#noise = librosa.resample(noise.transpose(1,0), orig_sr = fs, target_sr = self.fs).transpose(1,0)
				noise = scipy.signal.resample_poly(noise, up=self.fs, down=fs)
			nsample_desired = int(self.T * self.fs)
			noise_copy = copy.deepcopy(noise)
			nsample = noise.shape[0]
			while nsample < nsample_desired:
				noise_copy = np.concatenate((noise_copy, noise), axis=0)
				nsample = noise_copy.shape[0]

			st = random.randint(0, nsample - nsample_desired)
			ed = st + nsample_desired
			noise_signal = noise_copy[st:ed, :]

		else:
			raise Exception('Unknown noise type specified')

		return noise_signal

	def _exploreCorpus(self, path, file_extension):
		directory_tree = {}
		directory_path = []
		for item in os.listdir(path):
			if os.path.isdir( os.path.join(path, item) ):
				directory_tree[item], directory_path = self._exploreCorpus( os.path.join(path, item), file_extension )
			elif item.split(".")[-1] == file_extension:
				directory_tree[ item.split(".")[0] ] = os.path.join(path, item)
				directory_path += [os.path.join(path, item)]
		return directory_tree, directory_path

	def gen_Gaussian_noise(self, T, fs, nmic):
		noise = np.random.standard_normal((int(T*fs), nmic))

		return noise

	def gen_diffuse_noise(self, noise, T, fs, mic_pos, nfft=256, c=343.0, type_nf='spherical'):
		""" Reference:  E. A. P. Habets, “Arbitrary noise field generator.” https://github.com/ehabets/ANF-Generator
		"""
		M = mic_pos.shape[0]
		L = int(T*fs)

		# Generate M mutually 'independent' input signals
		noise = noise - np.mean(noise)
		noise_M = np.zeros([L, M])
		for m in range(0,M):
			noise_M[:, m] = noise[m*L:(m+1)*L]

		# Generate matrix with desired spatial coherence
		ww = 2*math.pi*self.fs*np.array([i for i in range(nfft//2+1)])/nfft
		DC = np.zeros([M, M, nfft//2+1])
		for p in range(0,M):
			for q in range(0,M):
				if p == q:
					DC[p,q,:] = np.ones([1,1,nfft//2+1])
				else:
					dist = np.linalg.norm(mic_pos[p,:]-mic_pos[q,:])
					if type_nf == 'spherical':
						DC[p,q,:] = np.sinc(ww*dist/(c*math.pi))
					elif type_nf == 'cylindrical':
						DC[p,q,:] = scipy.special(0,ww*dist/c)
					else:
						raise Exception('Unknown noise field')

		# Generate sensor signals with desired spatial coherence
		noise_signal = self.mix_signals(noise_M, DC)

		return noise_signal

	def mix_signals(self, noise, DC, method='cholesky'):
		""" Reference:  E. A. P. Habets, “Arbitrary noise field generator.” https://github.com/ehabets/ANF-Generator
		"""
		M = noise.shape[1] # Number of sensors
		K = (DC.shape[2]-1)*2 # Number of frequency bins

		# Compute short-time Fourier transform (STFT) of all input signals
		noise = np.vstack([np.zeros([K//2,M]), noise, np.zeros([K//2,M])])
		noise = noise.transpose()
		f, t, N = scipy.signal.stft(noise,window='hann', nperseg=K, noverlap=0.75*K, nfft=K)

		# Generate output in the STFT domain for each frequency bin k
		X = np.zeros(N.shape,dtype=complex)
		for k in range(1,K//2+1):
			if method == 'cholesky':
				C = scipy.linalg.cholesky(DC[:,:,k])
			elif method == 'eigen': # Generated cohernce and noise signal are slightly different from MATLAB version
				D, V = np.linalg.eig(DC[:,:,k])
				ind = np.argsort(D)
				D = D[ind]
				D = np.diag(D)
				V = V[:, ind]
				C = np.matmul(np.sqrt(D), V.T)
			else:
				raise Exception('Unknown method specified')

			X[:,k,:] = np.dot(np.squeeze(N[:,k,:]).transpose(),np.conj(C)).transpose()

		# Compute inverse STFT
		F, x = scipy.signal.istft(X,window='hann',nperseg=K,noverlap=0.75*K, nfft=K)
		x = x.transpose()[K//2:-K//2,:]

		return x

def set_seed(seed):
	""" Function: fix random seed.
	"""
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.enabled = False

	np.random.seed(seed)
	random.seed(seed)

def main():
    set_seed(21)
    save_dir = './ASA/dev'
    parser = argparse.ArgumentParser()

    # Configurations: read noisyspeech_synthesizer.cfg and gather inputs
    parser.add_argument('--cfg', default='scene_generator.cfg', help='Read scene_generator.cfg for all the details')
    parser.add_argument('--cfg_str', type=str, default='scene_generator')
    args = parser.parse_args()

    params = dict()
    params['args'] = args
    cfgpath = os.path.join(os.path.dirname(__file__), args.cfg)
    assert os.path.exists(cfgpath), f'No configuration file as [{cfgpath}]'

    cfg = CP.ConfigParser()
    cfg._interpolation = CP.ExtendedInterpolation()
    cfg.read(cfgpath)
    params['cfg'] = cfg._sections[args.cfg_str]
    cfg = params['cfg']

    params['fs'] = int(cfg['sampling_rate'])
    params['sec'] = int(cfg['seconds'])

    params['room_rt60_min'] = float(cfg['room_rt60_min'])
    params['room_rt60_max'] = float(cfg['room_rt60_max'])

    params['SNR_min'] = float(cfg['snr_min'])
    params['SNR_max'] = float(cfg['snr_max'])

    params['room_width_min'] = float(cfg['room_width_min'])
    params['room_width_max'] = float(cfg['room_width_max'])
    params['room_length_min'] = float(cfg['room_length_min'])
    params['room_length_max'] = float(cfg['room_length_max'])
    params['room_height_min'] = float(cfg['room_height_min'])
    params['room_height_max'] = float(cfg['room_height_max'])

    params['room_offset_inside'] = float(cfg['room_offset_inside'])

    params['source_num'] = int(cfg['source_num'])
    params['microphone_num'] = int(cfg['microphone_num'])
    params['microphone_radius'] = float(cfg['microphone_radius'])

    params['array_source_distance_min'] = float(cfg['array_source_distance_min'])
    params['array_source_distance_max'] = float(cfg['array_source_distance_max'])

    params['fileindex_start'] = int(cfg['fileindex_start'])
    params['fileindex_end'] = int(cfg['fileindex_end'])
    params['num_files'] = int(params['fileindex_end']) - int(params['fileindex_start'])

    dataset = RIR_generator(params)
    eps = 1e-12
    for file_idx in tqdm.tqdm(range(params['fileindex_end'])):
        file_idx = file_idx + 49174
        mic_signals, mic_signals_sources, dp_mic_signals_sources, SEL_pts, SED_pts, source_class = dataset[file_idx]
        info_mat = []
        for i in range(mic_signals_sources.shape[2]):
            time_frame = np.nonzero(SED_pts[:, i])[0]
            src_frame = i * np.ones_like(time_frame)
            cls_frame = source_class[i] * np.ones_like(time_frame)
            azi_frame = np.around(SEL_pts[time_frame, 1, i])
            ele_frame = np.around(SEL_pts[time_frame, 0, i])
            info_mat.append(np.stack([time_frame, src_frame, cls_frame, azi_frame, ele_frame], axis=0))
        # sort information matrix
        for i in range(len(info_mat)):
            if i == 0:
                info_arr = np.array(info_mat[i])
            else:
                info_arr = np.hstack((info_arr, np.array(info_mat[i])))
        sorted_indices = np.lexsort((info_arr[1, :], info_arr[0, :]))  # 첫 번째 열이 우선순위, 두 번째 열이 그다음 우선순위
        sorted_info_arr = info_arr[:, sorted_indices].T

        # normalize
        scaling_factor = np.max(np.abs(mic_signals)) + eps
        mic_signals = 32768 * mic_signals / scaling_factor
        mic_signals_sources = 32768 * mic_signals_sources / scaling_factor
        dp_mic_signals_sources = 32768 * dp_mic_signals_sources / scaling_factor
        # save audio
        sig_path = save_dir + '/mixed/' + str(file_idx) + '.wav'
        scipy.io.wavfile.write(sig_path, params['fs'], mic_signals.astype(np.int16))
		# save reverb and anechoic audio
        reverb_path = []
        os.mkdir(f'{save_dir}/reverb/{str(file_idx)}')
        for i in range(mic_signals_sources.shape[2]):
            reverb_path.append(f'{save_dir}/reverb/{str(file_idx)}/{i}.wav')
            reverb = mic_signals_sources[:, :, i] - dp_mic_signals_sources[:, :, i]
            scipy.io.wavfile.write(reverb_path[i], params['fs'], reverb.astype(np.int16))
        anechoic_path = []
        os.mkdir(f'{save_dir}/anechoic/{str(file_idx)}')
        for i in range(dp_mic_signals_sources.shape[2]):
            anechoic_path.append(f'{save_dir}/anechoic/{str(file_idx)}/{i}.wav')
            scipy.io.wavfile.write(anechoic_path[i], params['fs'], dp_mic_signals_sources[:, :, i].astype(np.int16))
        # save csv file
        csv_path = save_dir + '/info/' + str(file_idx) + '.csv'
        with open (csv_path, mode='w', newline='') as csvfile:
            fieldnames = ['time', 'src_idx', 'class', 'azimuth', 'elevation']
            writer = csv.writer(csvfile)
            writer.writerow(fieldnames)
            for info in sorted_info_arr:
                writer.writerow(info)


if __name__ == '__main__':
    main()