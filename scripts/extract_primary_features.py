import os

import numpy as np

import smoothiecore
import MFCC as mfc


PATH_TO_SONIC_ANNOTATOR = os.getenv('PATH_TO_SONIC_ANNOTATOR', './sonic-annotator')


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
			print "Melspectrogram failed for %s with error %s" % (output_csv, str(e))
			failed_files.append([audio_file, str(e)])
	n_fail_files = len(failed_files)
	n_success_files = len(audio_file_list) - len(failed_files)
	print "Successfully processed melspectrograms for %s files, failed %s files" % (n_success_files, n_fail_files)


def extract_chromagrams(audio_file_list, output_csv_list):
	failed_files = []
	for audio_file, output_csv in zip(audio_file_list, output_csv_list):
		try:
			chromagram = smoothiecore.get_smoothie(filename=audio_file, segment=True)
			np.savetxt(output_csv, chromagram.T, delimiter=',', fmt='%10.12f')  # write frames as rows
			print "Chromagram successfully extracted in %s" % output_csv
		except Exception as e:
			print "Chromagram failed for %s with error %s" % (output_csv, str(e))
			failed_files.append([audio_file, str(e)])
	n_fail_files = len(failed_files)
	n_success_files = len(audio_file_list) - len(failed_files)
	print "Successfully processed chromagrams for %s files, failed %s files" % (n_success_files, n_fail_files)


def extract_melodia(audio_file_list, output_csv_list):
	vamp_ext = "_vamp_mtg-melodia_melodia_melody.csv"
	failed_files = []
	for audio_file, output_csv in zip(audio_file_list, output_csv_list):
		try:
			path_to_output_dir = os.path.dirname(output_csv)
			os.system('%s -d vamp:mtg-melodia:melodia:melody %s -w csv --csv-basedir %s' % (PATH_TO_SONIC_ANNOTATOR, audio_file, path_to_output_dir))
			os.system('mv %s %s' % (os.path.join(path_to_output_dir, os.path.splitext(os.path.basename(audio_file))[0] + vamp_ext), output_csv))
			_ = open(output_csv, "r").read()  # to check the plugin wrote the file
			print "Melodia successfully extracted in %s" % output_csv
		except Exception as e:
			print "Melodia failed for %s with error %s" % (output_csv, str(e))
			failed_files.append([audio_file, str(e)])
	n_fail_files = len(failed_files)
	n_success_files = len(audio_file_list) - len(failed_files)
	print "Successfully processed melodia for %s files, failed %s files" % (n_success_files, n_fail_files)


def extract_speech_music_segmentation(audio_file_list, output_csv_list):
	vamp_ext = "_vamp_bbc-vamp-plugins_bbc-speechmusic-segmenter_segmentation.csv"
	failed_files = []
	for audio_file, output_csv in zip(audio_file_list, output_csv_list):
		try:
			path_to_output_dir = os.path.dirname(output_csv)
			os.system('%s -d vamp:bbc-vamp-plugins:bbc-speechmusic-segmenter:segmentation %s -w csv --csv-basedir %s' % (PATH_TO_SONIC_ANNOTATOR, audio_file, path_to_output_dir))
			os.system('mv %s %s' % (os.path.join(path_to_output_dir, os.path.splitext(os.path.basename(audio_file))[0] + vamp_ext), output_csv))
			_ = open(output_csv, "r").read()  # to check the plugin wrote the file
			print "Music segmentation successfully extracted in %s" % output_csv
		except Exception as e:
			print "Music segmentation failed for %s with error %s" % (output_csv, str(e))
			failed_files.append([audio_file, str(e)])
	n_fail_files = len(failed_files)
	n_success_files = len(audio_file_list) - len(failed_files)
	print "Successfully processed music segmentation for %s files, failed %s files" % (n_success_files, n_fail_files)


def extract_features(df, melspec=True, chroma=True, melodia=True, speech=True):
	audio_file_list = df['Audio'].get_values()
	if melspec:
		melspec_output_csv_list = df['Melspec'].get_values()
		extract_melspectrograms(audio_file_list, melspec_output_csv_list)
	if chroma:
		chroma_output_csv_list = df['Chroma'].get_values()
		extract_chromagrams(audio_file_list, chroma_output_csv_list)
	if melodia:
		melodia_output_csv_list = df['Melodia'].get_values()
		extract_melodia(audio_file_list, melodia_output_csv_list)
	if speech:
		speech_output_csv_list = df['Speech'].get_values()
		extract_speech_music_segmentation(audio_file_list, speech_output_csv_list)
