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

def get_labels(file_path):
	# puts all the names of the annotation (.atr) files into file_list so we can access the annotations with wfdb.rdann	
	os.chdir(file_path)
	file_list =[]
	for files in glob.glob('*.atr'):	
		file_list.append(files.replace('.atr',''))	

	# Stores a single annotation file from file list (here I just choose index 10 arbitratily) as a dictionary
	ann_rec = wfdb.rdann(file_list[10],'atr').__dict__ 			
	return ann_rec 

	
	
def main(): 
	ann_rec = get_labels("/gpfs/data/ceickhof/ecg_data/data/ve_data/files") 
	# Prints the annotation record  	
	print(ann_rec)
	# Displays the annotation labels  
	print(wfdb.show_ann_labels())

if __name__ == '__main__':
	main()

