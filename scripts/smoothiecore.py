"""Smoothie

A module aimed at melody salience and melody features. It contains

* a frequency transform inspired by constant-Q transforms
* an NMF-based melodic salience function
* methods that transform this salience into melody features (not implemented)

Note: Original code by Matthias Mauch.
"""

import sys
import numpy as np
from scipy.signal import hann
from scipy.sparse import csr_matrix
import librosa

p = {# smoothie q mapping parameters
     'bpo'           :     36,
     'break1'        :    500,
     'break2'        :   4000,
     'f_max'         :   8000,
     'fs'            :  16000,
     # spectrogram parameters
     'step_size'     :    160,
     'use_A_wtg'     :  False,
     'harm_remove'   :  False,
     # NMF and NMF dictionary parameters
     's'             :      0.85, # decay parameter
     'n_harm'        :     50, # I'm gonna do you no harm
     'min_midi'      :     25,
     'max_midi'      :     90.8,
     'bps'           :      5,
     'n_iter'        :     30
     }

def update_nmf_simple(H, W, V):
	"""Update the gain matrix H using the multiplicative NMF update equation.

	Keyword arguments:
	H -- gain matrix
	W -- template matrix
	V -- target data matrix
	"""
	WHV = np.dot(W, H) ** (-1) * V
	H = H * np.dot(W.transpose(), WHV)
	return H

def make_a_weight(f):
	"""Return the values of A-weighting at the input frequencies f."""
	f = np.array(f)
	zaehler = 12200 * 12200* (f**4)
	nenner = (f*f + 20.6*20.6) * np.sqrt((f*f + 107.7*107.7) * (f*f + 737.9*737.9)) * (f*f + 12200*12200)
	return zaehler / nenner

def nextpow2(x):
    """Return the smallest integer n such that 2 ** n > x."""
    return np.ceil(np.log2(x))

def get_smoothie_frequencies(p):
	"""Calculate filter centres and widths for the smooth Q transform."""
	# I think this calculation should be reconsidered in order to have
	# a more principled transition between linear and exponential
	n = 1.0 / (2.0**(1.0/p['bpo']) - 1)
	x = p['break1'] * 1.0 / n

	# first linear stretch
	# f = (np.arange(1, int(np.round(n)) + 1) * x).round().tolist()
	f = (np.arange(1, int(np.round(n)) + 1) * x).tolist()

	# exponential stretch
	while max(f) < p['break2']:
		f.append(max(f) * 2**(1.0/p['bpo']))

	# final linear stretch
	lastdiff = f[-1] - f[-2]
	while max(f) < p['f_max']:
		f.append(max(f) + lastdiff)

	deltaf = np.diff(np.array(f))
	f = f[:-1]

	return f, deltaf

def create_smoothie_kernel(f, deltaf, fs):
	"""Create a sparse matrix that maps the complex DFT to the complex 
	smoothie representation.
	"""
	print >>sys.stdout, "[ SMOOTHIE Q kernel calculation ... ]"
	n_filter = len(f)
	n_fft = 2**nextpow2(np.ceil(fs/min(deltaf)))
	
	thresh = 0.0054
	smoothie_kernel = np.zeros([n_fft, n_filter], np.complex64)

	for i_filter in range(n_filter-1, -1, -1): # descending
		# print i_filter
		Q = f[i_filter] * 1.0 / deltaf[i_filter] # local Q for this filter
		lgth = int(np.ceil(fs * 1.0 / deltaf[i_filter]))
		lgthinv = 1.0 / lgth
		offset = int(n_fft/2 - np.ceil(lgth * 0.5))
		temp = hann(lgth) * lgthinv * \
			np.exp(2j * np.pi * Q * (np.arange(lgth) - offset) * lgthinv)
		# print(sum(hann(lgth)), Q, lgth, offset)
		temp_kernel = np.zeros(n_fft, dtype = np.complex64)
		temp_kernel[np.arange(lgth) + offset] = temp
		spec_kernel = np.fft.fft(temp_kernel)
		spec_kernel[abs(spec_kernel) <= thresh] = 0
		smoothie_kernel[:,i_filter] = spec_kernel
	return csr_matrix(smoothie_kernel.conj()/n_fft)

def smoothie_q_spectrogram(x, p):
	"""Calculate the actual spectrogram with smooth Q frequencies"""
	print >>sys.stdout, "[ SMOOTHIE Q spectrogram ... ]"

	# precalculate smoothie kernel
	f, deltaf = get_smoothie_frequencies(p)
	smoothie_kernel = create_smoothie_kernel(f, deltaf, p['fs'])
	n_fft, n_filter = smoothie_kernel.shape
	
	# some preparations
	n_sample = len(x)
	# n_frame = int(np.floor(n_sample / p['step_size']))
	n_frame = int(np.ceil(n_sample / float(p['step_size']))) # added mp
	t = (np.arange(n_frame) * p['step_size']) * 1.0 / p['fs']
	smoothie_kernelT = smoothie_kernel.T

	# allocate
	s = np.zeros((n_filter, n_frame), dtype = np.complex64)

	# pad (if wanted)
	x = np.concatenate((np.zeros(n_fft/2), x, np.zeros(n_fft/2)))
	
	for i_frame in range(n_frame):
		smpl = p['step_size'] * i_frame
		block = x[smpl + np.arange(n_fft)]
		s[:, i_frame] = smoothie_kernelT.dot(np.fft.fft(block))

	if p['use_A_wtg']:
		a_wtg = make_a_weight(f)
		s = s * a_wtg[:, np.newaxis]

	return s, t

def mel_triangles(input_f):
	"""Make matrix with mel filters at the given frequencies.
	Warning: this is a very coarse mel filterbank.
	"""
	n_linearfilters = 3
	n_logfilters0 = 30 # just something high, will be pruned later
	linear_spacing = 100
	log_spacing = 6.0/4
	n_filter0 = n_linearfilters + n_logfilters0

	freqs = np.zeros(n_filter0+2) # includes one more on either side, hence +2
	freqs[range(n_linearfilters+1)] = \
		np.arange(-1,n_linearfilters) * linear_spacing
	freqs[range(n_linearfilters+1, n_filter0+2)] = \
		freqs[n_linearfilters] * log_spacing ** np.arange(1, n_logfilters0+2)

	centre_freqs = freqs[1:-1]
	lower_freqs  = freqs[0:-2]
	upper_freqs = freqs[2:]

	n_filter = list(np.nonzero(lower_freqs < max(input_f)))[0][-1] + 1

	n_input_f = len(input_f)
	mtr = np.zeros((n_input_f, n_filter))

	for i_filter in range(n_filter):
		for i_freq in range(n_input_f):
			if input_f[i_freq] > lower_freqs[i_filter] \
				and input_f[i_freq] <= upper_freqs[i_filter]:

				if input_f[i_freq] <= centre_freqs[i_filter]:
					mtr[i_freq, i_filter] = \
						(input_f[i_freq] - lower_freqs[i_filter]) * 1.0 / \
						(centre_freqs[i_filter] - lower_freqs[i_filter])
				else:
					mtr[i_freq, i_filter] = \
						1 - (input_f[i_freq] - centre_freqs[i_filter]) / \
						(upper_freqs[i_filter] - centre_freqs[i_filter])
	return mtr

def create_smoothie_nfm_dict(smoothie_kernel, filterf, p):
	"""Create dictionary matrix with note templates."""
	n_note = int((p['max_midi'] - p['min_midi']) * p['bps'] + 1)
	n_fft, n_filter = smoothie_kernel.shape
	t = ((np.arange(n_fft) + 1) - ((n_fft + 1)*0.5))/p['fs']

	mtr = mel_triangles(filterf)
	n_template = n_note + mtr.shape[1]

	w = np.zeros((n_filter, n_template), dtype = np.complex64)
	w[:,n_note:] = mtr
	f0s = []

	smoothie_kernelT = smoothie_kernel.T

	for i_note in range(n_note):
		midi = p['min_midi'] + i_note * 1.0 / p['bps']
		f0 = 440 * 2 ** ((midi-69)*1.0/12)
		f0s.append(f0)
		sig = np.zeros(len(t))
		for i_harm in range(p['n_harm']):
			f = f0 * (i_harm + 1)
			if f > p['fs'] * 0.5:
				continue
			x = np.sin(2*np.pi*f*t) * p['s']**(i_harm)
			sig += x
		w[:, i_note] = smoothie_kernelT.dot(np.fft.fft(sig))

	for i in range(mtr.shape[1]):
		f0s.append(np.nan)

	w = abs(w)
	col_sums = w.sum(axis = 0)
	w = w / col_sums[np.newaxis, :] # normalisation
	return w, np.array(f0s)

def smoothie_salience(x, p, do_tune = False):
	"""Calculate melodic salience."""
	print >>sys.stdout, "[ SMOOTHIE Q salience ... ]"

	# precalculate nmf kernel
	f, deltaf = get_smoothie_frequencies(p)
	smoothie_kernel = create_smoothie_kernel(f, deltaf, p['fs'])
	w, f0s = create_smoothie_nfm_dict(smoothie_kernel, f, p)

	# calculate smoothiegram
	s, t = smoothie_q_spectrogram(x, p)
	s[s==0] = np.spacing(1) # very small number
	s = abs(s)

	# run NMF
	n_frame = len(t)
	print >>sys.stdout, "[ SMOOTHIE Q : NMF updates ... ]"
	sal = np.ones((w.shape[1], n_frame))
	for i_iter in range(p['n_iter']):
		sal = update_nmf_simple(sal, w, s)

	if do_tune:
		error("Tuning isn't yet implemented in the Python version")

	return sal, t, f0s

def qnormalise(x, q, dim):
    nrm = np.sum(x**q, axis=dim, keepdims=True)**(1./float(q))
    nrmmatrix = np.repeat(nrm, x.shape[0], axis=0)
    x = x / nrmmatrix
    return x

def wrap_to_octave(sal, param):
    epsilon = 0.00001
    nsal = qnormalise(sal, 2, 0)
    
    step = 1./float(param['bps'])
    NMFpitch = np.arange(param['min_midi'],param['max_midi']+step,step)
    nNote = len(NMFpitch)
    nsal = nsal[0:nNote, :]

    allsal = nsal
    allsal[nsal<epsilon] = epsilon
    
    # chroma mapping
    n = param['bps']*12
    mmap = np.tile(np.eye(n),(1,5))
    allchroma = mmap.dot(allsal[0:(n*5),:])
    return allchroma
    
def segment_audio(y, sr):
    tracklength = y.shape[0]/float(sr)
    startSample = 0
    endSample = None
    if tracklength > 90:
        startPointSec = (tracklength/2.)-20
        startSample = round(startPointSec*sr)
        endSample = startSample+45*sr
    y = y[startSample:endSample]
    return y

def get_smoothie(filename='test.wav', param=None, segment=True, secondframedecomp=False, hopinsec=None):
    if param is None:
        param = p
        param['fs'] = 44100
        param['step_size'] = 128
    y, sr = librosa.load(filename, sr = param['fs'])
    param['fs'] = sr
    if hopinsec is not None:
        param['step_size'] = int(round(hopinsec*param['fs']))
    if segment:
        y = segment_audio(y,sr)
    # sg, t = smoothie_q_spectrogram(y, param)
    # nmf_dict = create_smoothie_nfm_dict(smoothie_kernel, f, param)
    sal, t, f0s = smoothie_salience(y, param)
    sal = sal[-np.isnan(f0s),:]
    chroma = wrap_to_octave(sal, param)
    
    if secondframedecomp:
        chromasr = param['fs']/float(param['step_size'])
        win2 = int(round(8*chromasr))
        hop2 = int(round(0.5*chromasr))
        nbins, norigframes = chroma.shape
        nframes = int(1+np.floor((norigframes-win2)/float(hop2)))
        avechroma = np.empty((nbins, nframes))
        for i in range(nframes):  # loop over all 8-sec frames
            avechroma[:,i] = np.mean(chroma[:, (i*hop2):(i*hop2+win2)], axis=1, keepdims=True)            
        chroma = avechroma
    return chroma
