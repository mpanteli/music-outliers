# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 01:50:57 2017

@author: mariapanteli
"""

import numpy as np
import pandas as pd
import os
from sklearn.decomposition import NMF
import OPMellin as opm
import MFCC as mfc
import PitchBihist as pbi


class FeatureLoader:
    def __init__(self, win2sec=8):
        self.win2sec = float(win2sec)
        self.sr = 44100.
        self.win1 = int(round(0.04*self.sr))
        self.hop1 = int(round(self.win1/8.))
        self.framessr = self.sr/float(self.hop1)
        self.win2 = int(round(self.win2sec*self.framessr))
        self.hop2 = int(round(0.5*self.framessr))
        self.framessr2 = self.framessr/float(self.hop2)
    
    
    def get_op_mfcc_for_file(self, melspec_file=None, scale=True, stop_sec=30.0):
        op = []
        mfc = []
        if not os.path.exists(melspec_file):
            return op, mfc
        print 'extracting onset patterns and mfccs...'
        songframes = pd.read_csv(melspec_file, engine="c", header=None)
        songframes.iloc[np.where(np.isnan(songframes))] = 0
        n_stop = np.int(np.ceil(stop_sec * self.framessr))
        songframes = songframes.iloc[0:min(len(songframes), n_stop), :]
        melspec = songframes.get_values().T
        op = self.get_op_from_melspec(melspec, K=2)
        mfc = self.get_mfcc_from_melspec(melspec)
        if scale:
            # scale all frames by mean and std of recording
            op = (op - np.nanmean(op)) / np.nanstd(op) 
            mfc = (mfc - np.nanmean(mfc)) / np.nanstd(mfc)
        return op, mfc
    
    
    def get_chroma_for_file(self, chroma_file=None, scale=True, stop_sec=30.0):
        ch = []
        if not os.path.exists(chroma_file):
            return ch
        print 'extracting chroma...'
        songframes = pd.read_csv(chroma_file, engine="c", header=None)
        songframes.iloc[np.where(np.isnan(songframes))] = 0
        n_stop = np.int(np.ceil(stop_sec * self.framessr))
        songframes = songframes.iloc[0:min(len(songframes), n_stop), :]
        chroma = songframes.get_values().T
        ch = self.get_ave_chroma(chroma)
        if scale:
            # scale all frames by mean and std of recording
            ch = (ch - np.nanmean(ch)) / np.nanstd(ch)                        
        return ch
    
        
    def get_music_idx_from_bounds(self, bounds, sr=None):
        music_idx = []
        if len(bounds) == 0:  
            # bounds is empty list
            return music_idx
        nbounds = bounds.shape[0]
        if len(np.where(bounds[:,2]=='m')[0])==0:
            # no music segments
            return music_idx
        elif len(np.where(bounds[:,2]=='s')[0])==nbounds:
            # all segments are speech
            return music_idx
        else:
            win2_frames = np.int(np.round(self.win2sec * self.framessr2))
            #half_win_hop = int(round(0.5 * self.win2 / float(self.hop2)))
            music_bounds = np.where(bounds[:, 2] == 'm')[0]
            bounds_in_frames = np.round(np.array(bounds[:, 0], dtype=float) * sr)
            duration_in_frames = np.ceil(np.array(bounds[:, 1], dtype=float) * sr)
            for music_bound in music_bounds:
                #lower_bound = np.max([0, bounds_in_frames[music_bound] - half_win_hop])
                #upper_bound = bounds_in_frames[music_bound] + duration_in_frames[music_bound] - half_win_hop
                lower_bound = bounds_in_frames[music_bound]
                upper_bound = bounds_in_frames[music_bound] + duration_in_frames[music_bound] - win2_frames
                music_idx.append(np.arange(lower_bound, upper_bound, dtype=int))
            if len(music_idx)>0:
                music_idx = np.sort(np.concatenate(music_idx))  # it should be sorted, but just in case segments overlap -- remove duplicates if segments overlap
        return music_idx

    
    def get_music_idx_for_file(self, segmenter_file=None):
        music_idx = []
        if os.path.exists(segmenter_file) and os.path.getsize(segmenter_file)>0:
            print 'loading speech/music segments...'
            bounds = pd.read_csv(segmenter_file, header=None, delimiter='\t').get_values()
            if bounds.shape[1] == 1:  # depends on the computer platform
                bounds = pd.read_csv(segmenter_file, header=None, delimiter=',').get_values()    
            music_idx = self.get_music_idx_from_bounds(bounds, sr=self.framessr2)
        return music_idx
    
    
    def get_features(self, df, stop_sec=30.0, class_label='Country', precomp_melody=True):
        oplist = []
        mflist = []
        chlist = []
        pblist = []        
        clabels = []
        aulabels = []
        n_files = len(df)
        for i in range(n_files):
            if not (os.path.exists(df['Melspec'].iloc[i]) and os.path.exists(df['Chroma'].iloc[i]) and os.path.exists(df['Melodia'].iloc[i])):
                continue
            print 'file ' + str(i) + ' of ' + str(n_files)
            music_idx = self.get_music_idx_for_file(df['Speech'].iloc[i])
            #min_dur_sec=8.0
            #min_n_frames = np.int(np.floor(min_dur_sec * self.framessr2))
            if len(music_idx)==0:  # or len(music_idx)<min_n_frames:
                # no music segments or music segment too short -> skip this file
                continue
            try:
                # allow feature extraction of longer segments (2*stop_sec)
                # because some of it might be speech segments that are filtered out
                stop_sec_feat = 2 * stop_sec  
                op, mfcc = self.get_op_mfcc_for_file(df['Melspec'].iloc[i], stop_sec=stop_sec_feat)
                ch = self.get_chroma_for_file(df['Chroma'].iloc[i], stop_sec=stop_sec_feat)
                pb = self.get_pb_for_file(df['Melodia'].iloc[i], precomp_melody=precomp_melody, stop_sec=stop_sec_feat)
                #if precomp_melody:
                #    pb = self.load_precomputed_pb_from_melodia(df['Melodia'].iloc[i], stop_sec=stop_sec)
                #else:
                #    pb = self.get_pb_from_melodia(df['Melodia'].iloc[i], stop_sec=stop_sec)
            except:
                continue
            n_stop = np.int(np.ceil(stop_sec * self.framessr2))
            print n_stop, len(op), len(mfcc), len(ch), len(pb)
            max_n_frames = np.min([n_stop, len(op), len(mfcc), len(ch), len(pb)])  # ideally, features should have the same number of frames
            if max_n_frames==0:
                # no features extracted -> skip this file
                continue
            # music segment duration must be <= 30sec
            music_idx = music_idx[music_idx<max_n_frames]  
            n_frames = len(music_idx)
            oplist.append(op.iloc[music_idx, :])
            mflist.append(mfcc.iloc[music_idx, :])
            chlist.append(ch.iloc[music_idx, :])
            pblist.append(pb.iloc[music_idx, :])
            clabels.append(pd.DataFrame(np.repeat(df[class_label].iloc[i], n_frames)))
            aulabels.append(pd.DataFrame(np.repeat(df['Audio'].iloc[i], n_frames)))
        print len(oplist), len(mflist), len(chlist), len(pblist), len(clabels), len(aulabels)
        return pd.concat(oplist), pd.concat(mflist), pd.concat(chlist), pd.concat(pblist), pd.concat(clabels), pd.concat(aulabels)

            
    def get_op_from_melspec(self, melspec, K=None):
        op = opm.OPMellin(win2sec=self.win2sec)
        opmellin = op.get_opmellin_from_melspec(melspec=melspec, melsr=self.framessr)
        opmel = pd.DataFrame(opmellin.T)
        if K is not None:
            opmel =  self.mean_K_bands(opmellin.T, K)
            opmel = pd.DataFrame(opmel)
        return opmel


    def get_mfcc_from_melspec(self, melspec, deltamfcc=True, avelocalframes=True, stdlocalframes=True):
        mf = mfc.MFCCs()        
        mfcc = mf.get_mfccs_from_melspec(melspec=melspec, melsr=self.framessr)
        if deltamfcc:
            ff = mfcc
            ffdiff = np.diff(ff, axis=1)
            ffdelta = np.concatenate((ffdiff, ffdiff[:,-1,None]), axis=1)
            frames = np.concatenate([ff,ffdelta], axis=0)                
            mfcc = frames
        if avelocalframes:
            mfcc = self.average_local_frames(mfcc, getstd=stdlocalframes)
        mfcc = pd.DataFrame(mfcc.T)
        return mfcc


    def get_ave_chroma(self, chroma, alignchroma=True, avelocalframes=True, stdlocalframes=True):
        chroma[np.where(np.isnan(chroma))] = 0
        if alignchroma:
            maxind = np.argmax(np.sum(chroma, axis=1))  # bin with max magnitude across time
            chroma = np.roll(chroma, -maxind, axis=0)
        if avelocalframes:
            chroma = self.average_local_frames(chroma, getstd=stdlocalframes)
        chroma = pd.DataFrame(chroma.T)
        return chroma
  
    
    def average_local_frames(self, frames, getstd=False):
        nbins, norigframes = frames.shape
        if norigframes<self.win2:
            nframes = 1
        else:
            nframes = int(1+np.floor((norigframes-self.win2)/float(self.hop2)))
        if getstd:
            aveframes = np.empty((nbins+nbins, nframes))
            for i in range(nframes):  # loop over all 8-sec frames
                meanf = np.nanmean(frames[:, (i*self.hop2):min((i*self.hop2+self.win2),norigframes)], axis=1)
                stdf = np.nanstd(frames[:, (i*self.hop2):min((i*self.hop2+self.win2),norigframes)], axis=1)
                aveframes[:,i] = np.concatenate((meanf,stdf))
        else:
            aveframes = np.empty((nbins, nframes))
            for i in range(nframes):  # loop over all 8-sec frames
                aveframes[:,i] = np.nanmean(frames[:, (i*self.hop2):min((i*self.hop2+self.win2),norigframes)], axis=1)
        return aveframes
    
    
    def mean_K_bands(self, songframes, K=40, nmels=40):
        [F, P] = songframes.shape
        Pproc = int((P/nmels)*K)
        procframes = np.zeros([F, Pproc])
        niters = int(nmels/K)
        nbins = P/nmels  # must be 200 bins
        for k in range(K):
            for j in range(k*niters, (k+1)*niters):
                procframes[:, (k*nbins):((k+1)*nbins)] += songframes[:, (j*nbins):((j+1)*nbins)]
        procframes /= float(niters)
        return procframes
        
        
    def nmfpitchbihist(self, frames):
        nbins, nfr = frames.shape
        npc = 2
        nb = int(np.sqrt(nbins))  # assume structure of pitch bihist is nbins*nbins
        newframes = np.empty(((nb+nb)*npc, nfr))
        for fr in range(nfr):
            pb = np.reshape(frames[:, fr], (nb, nb))
            try:
                nmfmodel = NMF(n_components=npc).fit(pb)
                W = nmfmodel.transform(pb)
                H = nmfmodel.components_.T
                newframes[:, fr, None] = np.concatenate((W, H)).flatten()[:, None]
            except:
                newframes[:, fr, None] = np.zeros(((nb+nb)*npc, 1))
        return newframes    


    def get_pb_for_file(self, melodia_file, precomp_melody=False, nmfpb=True, scale=True, stop_sec=30.0):
        pbihist = []
        if precomp_melody:
            pbihist = self.load_precomp_pb_from_melodia(melodia_file=melodia_file, stop_sec=stop_sec)
        else:
            pbihist = self.extract_pb_from_melodia(melodia_file=melodia_file, stop_sec=stop_sec)
        if len(pbihist) == 0:
            # no file was found
            return pbihist
        if nmfpb is True:
            pbihist = self.nmfpitchbihist(pbihist)
        pbihist = pd.DataFrame(pbihist.T)
        if scale:
            # scale all frames by mean and std of recording
            pbihist = (pbihist - np.nanmean(pbihist)) / np.nanstd(pbihist)
        return pbihist


    def extract_pb_from_melodia(self, melodia_file=None, stop_sec=30.0):
        pbihist = []
        if not os.path.exists(melodia_file):
            return pbihist
        print 'extracting pitch bihist from melodia...'
        pb = pbi.PitchBihist(win2sec=self.win2sec)
        pbihist = pb.bihist_from_melodia(filename=melodia_file, stop_sec=stop_sec)
        return pbihist


    def load_precomp_pb_from_melodia(self, melodia_file=None, stop_sec=30.0):
        pbihist = []
        base = os.path.basename(melodia_file)    
        root = '/import/c4dm-05/mariap/Melodia-melody-'+str(int(self.win2sec))+'sec/'
        root_BL = '/import/c4dm-04/mariap/FeatureCsvs_BL_old/PB-melodia/'
        root_SM = '/import/c4dm-04/mariap/FeatureCsvs/PB-melodia/'
        if 'SampleAudio' in base:
            root = root_SM
        else:
            root = root_BL
            base = base.split('_')[-1].split('.csv')[0]+'_vamp_mtg-melodia_melodia_melody.csv'
        bihist_file = os.path.join(root, base)
        if not os.path.exists(bihist_file):
            return pbihist
        print 'load precomputed pitch bihist', root
        pbihist = np.loadtxt(bihist_file, delimiter=',').T
        n_stop = np.int(np.ceil(stop_sec * self.framessr2))
        pbihist = pbihist[:, :np.min([pbihist.shape[0], n_stop])]
        return pbihist
        
