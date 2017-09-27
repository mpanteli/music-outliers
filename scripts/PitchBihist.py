# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 22:26:10 2016

@author: mariapanteli
"""
import numpy as np
import os
import scipy.signal


class PitchBihist:
    def __init__(self, win2sec=8):
        self.win2sec = win2sec
        self.hop2sec = 0.5
        self.melody_sr = 44100. / 128.


    def hz_to_cents(self, freq_Hz, ref_Hz=32.703, n_cents=1200):
        """ convert frequency values from Hz to cents
            reference frequency at C1 
        """        
        freq_cents = np.round(n_cents * np.log2(freq_Hz/ref_Hz))
        return freq_cents
    

    def wrap_to_octave(self, cents, octave_length=1200):
        """ wrap to a single octave 0-1200
        """
        octave_cents = cents % octave_length
        return octave_cents


    def get_melody_from_file(self, melodia_file, stop_sec=None):
        if not os.path.exists(melodia_file):
            return []
        data = np.loadtxt(melodia_file, delimiter=',')
        times, freqs = (data[:, 0], data[:, 1])
        if stop_sec is not None:
            stop_idx = np.where(times < stop_sec)[0]
            times, freqs = times[stop_idx], freqs[stop_idx]
        freqs[freqs<=0] = np.nan
        melody = freqs
        return melody


    def get_melody_matrix(self, melody):
        n_bins = 60
        n_frames = len(melody)
        melody_cents = self.hz_to_cents(melody, n_cents=n_bins)
        melody_octave = self.wrap_to_octave(melody_cents, octave_length=n_bins)
        melody_matrix = np.zeros((n_bins, n_frames))
        for time, pitch in enumerate(melody_octave):
            if not np.isnan(pitch):
                melody_matrix[int(pitch), time] = 1
        return melody_matrix


    def bihistogram(self, spec, spec_sr=None, winsec=0.5, align=True):
        if spec_sr is None:
            # assume spec is melody_matrix with default sr
            spec_sr = self.melody_sr
        win = int(round(winsec * spec_sr))
        ker = np.concatenate([np.zeros((win, 1)), np.ones((win+1, 1))], axis=0)
        spec = spec.T  # transpose to have frames as rows in convolution

        # energy threshold
        thr = 0.3*np.max(spec)
        spec[spec < max(thr, 0)] = 0

        # transitions via convolution
        tra = scipy.signal.convolve2d(spec, ker, mode='same')
        tra[spec > 0] = 0

        # multiply with original
        B = np.dot(tra.T, spec)

        # normalize to [0, 1]
        mxB = np.max(B)
        mnB = np.min(B)
        if mxB != mnB:
            B = (B - mnB)/float(mxB-mnB)

        # circshift to highest?
        if align:
            ref = np.argmax(np.sum(spec, axis=0))
            B = np.roll(B, -ref, axis=0)
            B = np.roll(B, -ref, axis=1)
        return B


    def second_frame_decomposition(self, norigframes):
        win2 = int(round(self.win2sec * self.melody_sr))
        hop2 = int(round(self.hop2sec * self.melody_sr))
        print win2, hop2, norigframes
        if norigframes<=win2:
            nframes = 1
            win2 = norigframes
        else:
            nframes = np.int(1+np.floor((norigframes-win2)/float(hop2)))
        return nframes, win2, hop2


    def bihist_from_melodia(self, filename='sample_melodia.csv', secondframedecomp=True, stop_sec=None):
        melody = self.get_melody_from_file(filename, stop_sec=stop_sec)
        if len(melody) == 0:
            return []
        melody_matrix = self.get_melody_matrix(melody)
        bihist = []
        if secondframedecomp:
            nbins, norigframes = melody_matrix.shape
            nframes, win2, hop2 = self.second_frame_decomposition(norigframes)
            bihistframes = np.empty((nbins*nbins, nframes))
            for i in range(nframes):  # loop over all 8-sec frames
                frame = melody_matrix[:, (i*hop2):(i*hop2+win2)]
                bihist = self.bihistogram(frame)
                bihist = np.reshape(bihist, -1)
                bihistframes[:, i] = bihist
            bihist = bihistframes
        else:
            bihist = self.bihistogram(melody_matrix)
        return bihist


if __name__ == '__main__':
    pb = PitchBihist()
    melody = np.concatenate([660 * np.ones(250), 440 * np.ones(500)])
    melody_matrix = pb.get_melody_matrix(melody)
    pb.bihistogram(melody_matrix, align=True)
    