# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 12:58:07 2015

@author: mariapanteli
"""
import librosa
import scipy.signal
import numpy

class MFCCs:
    def __init__(self):
        self.y = None
        self.sr = None
        self.melspec = None
        self.melsr = None
        self.win1 = None
        self.hop1 = None
        self.mfccs = None

    def load_audiofile(self, filename='test.wav', sr=None, segment=True):
        self.y, self.sr = librosa.load(filename, sr=sr)
        if segment:
            tracklength = self.y.shape[0]/float(self.sr)
            startSample = 0
            endSample = None
            if tracklength > 90:
                startPointSec = (tracklength/2.)-20
                startSample = round(startPointSec*self.sr)
                endSample = startSample+45*self.sr
            self.y = self.y[startSample:endSample]
    
    def mel_spectrogram(self, y=None, sr=None):
        if self.y is None:
            self.y = y
        if self.sr is None:
            self.sr = sr
        win1 = int(round(0.04*self.sr))
        hop1 = int(round(win1/8.))
        nfft1 = int(2**numpy.ceil(numpy.log2(win1)))
        nmels = 40
        D = numpy.abs(librosa.stft(self.y, n_fft=nfft1, hop_length=hop1, win_length=win1, window=scipy.signal.hamming))**2
        #melspec = librosa.feature.melspectrogram(S=D, sr=self.sr, n_mels=nmels)  
        melspec = librosa.feature.melspectrogram(S=D, sr=self.sr, n_mels=nmels, fmax=8000)  
        melsr = self.sr/float(hop1)
        self.melspec = melspec
        self.melsr = melsr
        
    def calc_mfccs(self, y=None, sr=None):
        if self.y is None:
            self.y = y
        if self.sr is None:
            self.sr = sr
        # require log-amplitude
        self.mfccs = librosa.feature.mfcc(S=librosa.logamplitude(self.melspec), n_mfcc=21)
        # remove DCT component        
        self.mfccs = self.mfccs[1:,:]
        
    def get_mfccs(self, filename='test.wav', secondframedecomp=False):
        self.load_audiofile(filename=filename)
        self.mel_spectrogram()
        self.calc_mfccs()
        
        if secondframedecomp:
            win2 = int(round(8*self.melsr))
            hop2 = int(round(0.5*self.melsr))
            nbins, norigframes = self.melspec.shape
            nframes = int(1+numpy.floor((norigframes-win2)/float(hop2)))
            avemfccs = numpy.empty((nbins, nframes))
            for i in range(nframes):  # loop over all 8-sec frames
                avemfccs[:,i] = numpy.mean(self.mfccs[:, (i*hop2):(i*hop2+win2)], axis=1, keepdims=True)            
            self.mfccs = avemfccs
        return self.mfccs
    
    def get_mfccs_from_melspec(self, melspec=[], melsr=[]):
        self.melspec = melspec
        self.melsr = melsr
        self.calc_mfccs()
        return self.mfccs

if __name__ == '__main__':
    mfs = MFCCs()
    mfs.get_mfccs()