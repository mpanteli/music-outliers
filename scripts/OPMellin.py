# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 13:37:00 2015

@author: mariapanteli
"""

import numpy as np
import scipy.signal
import librosa


class OPMellin:
    def __init__(self, win2sec=8):
        self.y = None
        self.sr = None
        self.melspec = None
        self.melsr = None
        self.op = None
        self.opmellin = None
        self.win2sec = win2sec
        self.hop2sec = 0.5
        

    def load_audiofile(self, filename, sr=None, segment=True):
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
        nfft1 = int(2**np.ceil(np.log2(win1)))
        nmels = 40
        D = np.abs(librosa.stft(self.y, n_fft=nfft1, hop_length=hop1, win_length=win1, window=scipy.signal.hamming))**2
        melspec = librosa.feature.melspectrogram(S=D, sr=self.sr, n_mels=nmels, fmax=8000)
        melsr = self.sr/float(hop1)
        self.melspec = melspec
        self.melsr = melsr
 

    def post_process_spec(self, melspec=None, log=True, medianfilt=True, sqrt=True, diff=True, subtractmean=True, halfwave=True, maxnormal=True):
        if self.melspec is None:
            self.melspec = melspec
        if log:
            self.melspec = librosa.logamplitude(self.melspec)
        if medianfilt:
            ks = int(0.1 * self.melsr) # 100ms kernel size
            if ks % 2 == 0: # ks must be odd
                ks += 1
            nmels = self.melspec.shape[0]
            for i in range(nmels):
                self.melspec[i,:] = scipy.signal.medfilt(self.melspec[i,:],kernel_size = ks)
        if sqrt:
            self.melspec = self.melspec**.5
        if diff:
            # append one frame before diff to keep number of frames the same
            self.melspec = np.concatenate((self.melspec,self.melspec[:,-1,None]),axis=1)  
            self.melspec = np.diff(self.melspec, n=1, axis=1)
        if subtractmean:
            mean = self.melspec.mean(axis=1)
            mean.shape = (mean.shape[0], 1)
            self.melspec = self.melspec - mean
        if halfwave:
            self.melspec[np.where(self.melspec < 0)] = 0
        if maxnormal:
            if self.melspec.max() != 0:  # avoid division by 0
                self.melspec = self.melspec/self.melspec.max()


    def onset_patterns(self, melspec=None, melsr=None, center=False, bpmrange=True):
        if self.melspec is None:
            self.melspec = melspec
        if self.melsr is None:
            self.melsr = melsr
        win2 = int(round(self.win2sec * self.melsr))
        hop2 = int(round(self.hop2sec * self.melsr))
        nmels, nmelframes = self.melspec.shape
        
        #nfft2 = int(2**np.ceil(np.log2(win2)))
        nfft2 = 2048  # nfft2 does not depend on win2??
        melspectemp = self.melspec
        if ((nfft2 > win2) and (center is False)):
            # in librosa version < 6.0, window is padded to the size of nfft 
            # so if win2<nfft2 the frames returned are less than expected
            # solution: pad the signal by (nfft2-win2)/2 on the edges 
            # then frame decomposition (n_frames) matches the one expected using win2
            melspectemp = np.concatenate([np.zeros((nmels,int((nfft2 - win2) // 2))),self.melspec, np.zeros((nmels,int((nfft2 - win2) // 2)))],axis=1)
        if melspectemp.shape[1]<nfft2:
            # pad with zeros to have at least one 8-sec window
            nzeros = nfft2 - melspectemp.shape[1]
            melspectemp = np.concatenate([np.zeros((nmels,int(np.ceil(nzeros / 2.)))),melspectemp, np.zeros((nmels,int(np.ceil(nzeros / 2.))))],axis=1)
            #melspectemp = np.concatenate([melspectemp, np.zeros((nmels,nzeros))],axis=1)
        # nframes = int(round(np.ceil(round(nmelframes/hop2))))
        ff0 = np.abs(librosa.stft(y=melspectemp[0, :], win_length=win2, hop_length=hop2, n_fft=nfft2, window=scipy.signal.hamming, center=center))
        nframes = ff0.shape[1]
        # nmags, nframes = ff0.shape
        if bpmrange:
            freqresinbpm = float(self.melsr)/float(nfft2/2.)*60.
            minmag = int(np.floor(30./freqresinbpm))  # min tempo 30bpm
            maxmag = int(np.ceil(960./freqresinbpm))  # max tempo 960 bpm
            magsinds = range(minmag, maxmag)
        else:
            magsinds = range(0, 200)
        nmags = len(magsinds)
        fft2 = np.zeros((nmels, nmags, nframes))
        for i in range(nmels):
            fftmags = np.abs(librosa.stft(y=melspectemp[i, :], win_length=win2, hop_length=hop2, n_fft=nfft2, window=scipy.signal.hamming, center=center))
            fftmags = fftmags[magsinds, :]
            fft2[i, :, :] = fftmags
        op = fft2
        self.op = op


    def post_process_op(self, medianfiltOP=True):
        if medianfiltOP:
            hop2 = int(round(0.5*self.melsr))
            ssr = self.melsr/float(hop2)
            ks = int(0.5 * ssr)  # 100ms kernel size
            if ks % 2 == 0:  # ks must be odd
                ks += 1
            nmels, nmags, nframes = self.op.shape
            for i in range(nmels):
                for j in range(nframes):
                    #self.op[i,:,j] = scipy.signal.medfilt(self.op[i,:,j],kernel_size = ks)
                    self.op[i,:,j] = np.convolve(self.op[i,:,j], np.ones(ks)/ks, mode='same')


    def mellin_transform(self, op=None):
        if self.op is None:
            self.op = op
        nmels, nmags, nframes = self.op.shape
        #nmagsout = min(200, nmags)
        nmagsout = 200
        u_max = np.log(nmags)
        delta_c = np.pi / u_max
        c_max = nmagsout
        c = np.arange(delta_c, c_max, delta_c)
        k = range(1, nmags)
        exponent = 0.5 - c*1j

        normMat = 1./(exponent * np.sqrt(2*np.pi))
        normMat.shape = (normMat.shape[0], 1)
        normMat = np.repeat(normMat.T, nmels, axis=0)
        kernelMat = np.asarray([np.power(ki, exponent) for ki in k])
        opmellin = np.zeros((nmels, nmagsout, nframes))
        for i in range(nframes):
            self.op[:, -1, i] = 0
            deltaMat = - np.diff(self.op[:, :, i])
            mellin = np.abs(np.dot(deltaMat, kernelMat) * normMat)
            opmellin[:, :, i] = mellin[:, :nmagsout]
        self.opmellin = opmellin


    def post_process_opmellin(self, opmellin=None, aveFreq=False, normFrame=True):
        if self.opmellin is None:
            self.opmellin = opmellin
        if aveFreq:
            self.opmellin = np.mean(self.opmellin, axis=0)
        else:  # just reshape
            nmels, nmags, nframes = self.opmellin.shape
            self.opmellin = self.opmellin.reshape((nmels*nmags, nframes))
        if normFrame:
            min_opmellin = np.amin(self.opmellin, axis=0, keepdims=True)
            max_opmellin = np.amax(self.opmellin, axis=0, keepdims=True)
            denom = max_opmellin - min_opmellin
            denom[denom==0] = 1 # avoid division by 0 if frame is all 0s-silent
            self.opmellin = (self.opmellin - min_opmellin) / denom


    def get_opmellin(self, filename='test.wav', logfilter=False, center=True, medianfiltspec=False, medianfiltOP=False):
        self.load_audiofile(filename=filename)        
        self.mel_spectrogram()
        self.post_process_spec(log=False, sqrt=True, medianfilt=medianfiltspec)  # sqrt seems to work better
        self.onset_patterns(logfilter=logfilter, center=center)
        self.post_process_op(medianfiltOP=medianfiltOP)
        self.mellin_transform()
        self.post_process_opmellin()
        return self.opmellin


    def get_opmellin_from_melspec(self, melspec=[], melsr=[]):
        self.melspec = melspec
        self.melsr = melsr
        self.post_process_spec(log=False, sqrt=True, medianfilt=True)  # sqrt seems to work better
        self.onset_patterns(center=False)
        self.post_process_op(medianfiltOP=True)
        self.mellin_transform()
        self.post_process_opmellin()
        return self.opmellin


if __name__ == '__main__':
    op = OPMellin()
    op.get_opmellin()
