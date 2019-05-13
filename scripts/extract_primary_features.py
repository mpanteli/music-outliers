import os

import numpy as np

import smoothiecore
import MFCC as mfc


DATA_DIR = '../data/'


def extract_melspectrograms(audio_file_list, output_csv_list):
	failed_files = []
	for audio_file, output_csv in zip(audio_file_list, output_csv_list):
		try:
			melspec_extractor = mfc.MFCCs()
			melspec_extractor.load_audiofile(filename=audio_file, segment=True)
			melspec_extractor.mel_spectrogram()
			melspec =  melspec_extractor.melspec
			np.savetxt(output_csv, melspec.T, delimiter=',', fmt='%10.12f')  # write frames as rows
			print "Melspectrogram successfully extracted in %s" % output_csv
		except Exception as e:
			failed_files.append([audio_file, str(e)])
	n_fail_files = len(failed_files)
	n_success_files = len(audio_file_list) - len(failed_files)
	print "Successfully processed %s files, failed %s files" % (n_success_files, n_fail_files)


def extract_chromagrams(audio_file_list, output_csv_list):
	failed_files = []
	for audio_file, output_csv in zip(audio_file_list, output_csv_list):
		try:
			chromagram = smoothiecore.get_smoothie(filename=audio_file, segment=True)
			np.savetxt(output_csv, chromagram.T, delimiter=',', fmt='%10.12f')  # write frames as rows
			print "Chromagram successfully extracted in %s" % output_csv
		except Exception as e:
			failed_files.append([audio_file, str(e)])
	n_fail_files = len(failed_files)
	n_success_files = len(audio_file_list) - len(failed_files)
	print "Successfully processed %s files, failed %s files" % (n_success_files, n_fail_files)


def extract_melodia(audio_file_list, output_csv_list):
	vamp_ext = "_vamp_mtg-melodia_melodia_melody.csv"
	failed_files = []
	for audio_file, output_csv in zip(audio_file_list, output_csv_list):
		try:
			path_to_output_dir = os.path.dirname(output_csv)
			os.system('./sonic-annotator -d vamp:mtg-melodia:melodia:melody %s -w csv --csv-basedir %s' % (audio_file, path_to_output_dir))
			os.system('mv %s %s' % (os.path.join(path_to_output_dir, os.path.splitext(os.path.basename(audio_file))[0] + vamp_ext), output_csv))
			print "Melodia successfully extracted in %s" % output_csv
		except Exception as e:
			failed_files.append([audio_file, str(e)])
	n_fail_files = len(failed_files)
	n_success_files = len(audio_file_list) - len(failed_files)
	print "Successfully processed %s files, failed %s files" % (n_success_files, n_fail_files)


def extract_speech_music_segmentation(audio_file_list, output_csv_list):
	vamp_ext = "_vamp_bbc-vamp-plugins_bbc-speechmusic-segmenter_segmentation.csv"
	failed_files = []
	for audio_file, output_csv in zip(audio_file_list, output_csv_list):
		try:
			path_to_output_dir = os.path.dirname(output_csv)
			os.system('./sonic-annotator -d vamp:bbc-vamp-plugins:bbc-speechmusic-segmenter:segmentation %s -w csv --csv-basedir %s' % (audio_file, path_to_output_dir))
			os.system('mv %s %s' % (os.path.join(path_to_output_dir, os.path.splitext(os.path.basename(audio_file))[0] + vamp_ext), output_csv))
			print "Speech/music segmentation successfully extracted in %s" % output_csv
		except Exception as e:
			failed_files.append([audio_file, str(e)])
	n_fail_files = len(failed_files)
	n_success_files = len(audio_file_list) - len(failed_files)
	print "Successfully processed %s files, failed %s files" % (n_success_files, n_fail_files)


def extract_features(df):
	audio_file_list = DATA_DIR + df['Audio'].get_values()
	melspec_output_csv_list = DATA_DIR + df['Melspec'].get_values()
	chroma_output_csv_list = DATA_DIR + df['Chroma'].get_values()
	melodia_output_csv_list = DATA_DIR + df['Melodia'].get_values()
	speech_output_csv_list = DATA_DIR + df['Speech'].get_values()
	extract_melspectrograms(audio_file_list, melspec_output_csv_list)
	extract_chromagrams(audio_file_list, chroma_output_csv_list)
	extract_melodia(audio_file_list, melodia_output_csv_list)
	extract_speech_music_segmentation(audio_file_list, speech_output_csv_list)
