import glob
import os
import wfdb
from wfdb import processing
import numpy as np
import heartpy as hp 
#from torchvision import transforms
import matplotlib.pyplot as plt 
 
"""Defines the window_size. 360 values are recorded every second. A window size of 10800 corresponds to a 30 second ECG window. We use a lag of approximately 8 seconds. A lag in slicing the windows ensures that some of the ECG signal is represented multiple times. This well help with detecting any unusual behavior present in the datasets and should improve modul accuracy. """

window_size = 10800
lag = 3000 



def slice_data(inputs):
	"""slice_data creates partially overlapping windoes of the clean ECG data. 
	INPUTS:
	inputs: preprocessed ECG signal
	RETURNS: partially overlapping ECG windows. """	
	
	data = []	
	for i in range(len(inputs)):	
		for j in range(0,len(inputs[0]), window_size - lag):
			if len(inputs[i][j:j+window_size]) != window_size:	
				break		
			else:
				data.append(inputs[i][j:j+window_size])
	return data 

def get_min(file_path):    
	"""In the Physionet dataset the lengths of the ECG signals for each patient can very. In order to standardize the lenght of each signal so each patient 
	is represented in the dataset equally, we need to retrieve the minimum signal lengh after we run our baseline wander and resampling procedues.
	INPUTS:
	file_path: string that contains the the path to the file where the ECG data is stored. 
	RETURNS: the minimum signal length for a patient in the dataset. """	
	
	os.chdir(file_path)
	file_list =[]
	for files in glob.glob("*.dat"):	
		file_list.append(files.replace('.dat',''))	
	clean_signals = []
	fs_target = 200
	
	min_len = []	
	for f in file_list:
		record = wfdb.rdrecord(f).__dict__
		signal =  record['p_signal'][:,0]
		fs = record['fs']			
		ecg= hp.remove_baseline_wander(signal, 200)	
		ecg,_ = np.array(wfdb.processing.resample_sig(ecg,fs,fs_target))	
		min_len.append(len(ecg))
	min_f = min(min_len)
	return min_f	

def get_data(file_path, target_directory, ecg_type):
	"""Input a file path to the given data file. Selects the arguments of the files containing ECG data. Performs preprocessing on the data. Slices the 
	data into partially overlapping windows by calling slice_data. Creates the spectrogram images and saves them into target_directory
	INPUTS:
	file_path: string of the filepath that contains the directory where the ECG data is stored.
	target_directory: string of the target directory where you want the spectrogram images to be stored.
	ecg_type: string of the arrhythmia type used in get_data. Used to name the spectrogram images. """

	os.chdir(file_path)
	file_list =[]
	for files in glob.glob("*.dat"):	
		file_list.append(files.replace('.dat',''))

		
	clean_signals = []
	# Defines the target frequency that we resample all of the data to be in. 	
	fs_target = 200
	min_len = get_min(file_path)	
	for f in file_list:
		record = wfdb.rdrecord(f).__dict__
		signal =  record['p_signal'][:,0]
		fs = record['fs']			
		# Remove baseline_wander, apply a lowpass filter, a highpass filter and finally normalize the bounds 	
		ecg= hp.remove_baseline_wander(signal, 200)
		ecg = hp.filter_signal(ecg, cutoff=0.75, sample_rate=200, filtertype='lowpass')
		ecg = hp.filter_signal(ecg, cutoff=0.75, sample_rate=200, filtertype='highpass')	
		ecg,_ = np.array(wfdb.processing.resample_sig(ecg,fs,fs_target))		
		ecg = wfdb.processing.normalize_bound(ecg)			
		# ECG signal length is different for each patient, so we only create spectograms for the first min_len elements of a singal for a given patient	
		clean_signals.append(ecg[:min_len])		
	clean_signals = np.reshape(clean_signals,(len(file_list), min_len))
	
	# Slice the data into partially overlapping windows, generate the spectrogram and save png files in dir_name 	
	windows = slice_data(clean_signals) 
	for i in range(len(windows)):
		plt.specgram(windows[i], Fs = 1, cmap = 'jet')
		plt.savefig(target_directory + "/" + ecg_type + str(i)+ ".png") 
		if i%100 == 0:
			print("Graph:" + str(i) + "completed")	
	return 

 
#def get_pxl_mat(file_path):
#	os.dir(file_path)
#	for files in glob.glob("*.png"):
#		input_image = Image.open(files)
#		preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
#		preprocess(input_image)

	
	
def main(): 
	 

if __name__ == '__main__':
	main()

