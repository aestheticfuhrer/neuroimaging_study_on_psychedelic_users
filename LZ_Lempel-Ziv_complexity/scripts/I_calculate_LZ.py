# -*- coding: utf-8 -*-

from numpy import *
from numpy.linalg import *
from scipy import signal
from scipy.signal import butter,lfilter,hilbert
from scipy.stats import ranksums
from scipy.io import savemat, loadmat, whosmat
from pylab import *
import os as os
from os import listdir
from os.path import isfile, join
from random import sample
import mat73


def Pre2(X):
 '''
 Linear-detrend and subtract mean
 '''
 ro,co=shape(X)
 Z=zeros((ro,co))
 for i in range(ro): #maybe divide by std?
  try:
   Z[i,:]=signal.detrend((X[i,:]-mean(X[i,:]))/std(X[i,:]), axis=0)
  except ValueError:
   pass
 return Z

def Ph_rand(original_data):
 '''
 phase randomisation: multi time series in, shuffled time series out - but it has same power spectrum
 '''
 #NOTE WHEN DOING SCHREIBER_SURRO< MAYBE TRY EFFECT OF DIFFERENT PHASE DISTRIBUTIONS ON MEASURES

 surrogates = np.fft.rfft(original_data, axis=1)
 #  Get shapes
 (N, n_time) = original_data.shape
 len_phase = surrogates.shape[1]

 #  Generate random phases uniformly distributed in the
 #  interval [0, 2*Pi]
 phases = np.random.uniform(low=0, high=2 * np.pi, size=(N, len_phase))

 #  Add random phases uniformly distributed in the interval [0, 2*Pi]
 surrogates *= np.exp(1j * phases)

 #  Calculate IFFT and take the real part, the remaining imaginary part
 #  is due to numerical errors.
 return np.ascontiguousarray(np.real(np.fft.irfft(surrogates, n=n_time,axis=1)))

##############
'''
PSpec
'''
##############

def find_closest(A, target):
   '''
   helper function for power spectrum functions (plot and PSpec)
   '''
   #A must be sorted
   idx = A.searchsorted(target)
   idx = np.clip(idx, 1, len(A)-1)
   left = A[idx-1]
   right = A[idx]
   idx -= target - left < right - target
   return idx


def PSpec(X):
 X=Pre2(X)
 fs=250. #sampling rate in Hz

 #data=one dimensional time series

 opt=1
# #n=14 #number of bands size 5 Hz
# F=zeros((n,2))
# for i in range(n):
#  F[i]=[5*i,5*(i+1)]
 de=[1,4]# in Hz
 th=[4,8]
 al=[8,13]
 be=[13,30]
 ga=[30,60]
 hga=[60,120]

 F=[de,th,al,be]#,ga,hga]

 ro,co=shape(X)
 Q=[]

 for i in range(ro):

  v=X[i]
  co=len(v)
  # Number of samplepoints
  N = co
  # sample spacing (denominator in Hz)
  T = 1.0 / fs
  y = v
  yf = fft(y)
  xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
  yff=2.0/N * np.abs(yf[0:N//2])

  bands=zeros(len(F))
  for i in range(len(F)):
   bands[i]=sum(yff[find_closest(xf, F[i][0]):find_closest(xf, F[i][1])])
  bands=bands/sum(bands)
  Q.append(bands)
 return Q



#############
'''
frequency filter
'''
#############

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_highpass(lowcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='highpass')
    return b, a

def butter_highpass_filter(data, lowcut, fs, order):
    b, a = butter_highpass(lowcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def notch_iir(fs,f0,data):
    '''
    fs: Sample frequency (Hz)
    f0: Frequency to be removed from signal (Hz)
    '''

    Q = 10.# 30.0  # Quality factor
    w0 = float(f0)/(fs/2)  # Normalized Frequency
    b, a = signal.iirnotch(w0, Q)
    return lfilter(b, a, data)


##########
'''
LZc - Lempel-Ziv Complexity, column-by-column concatenation

X is continuous multidimensional time series, channels x observations
'''
##########

def cpr(string):
 '''
 Lempel-Ziv-Welch compression of binary input string, e.g. string='0010101'. It outputs the size of the dictionary of binary words.
 '''
 d={}
 w = ''
 i=1
 for c in string:
  wc = w + c
  if wc in d:
   w = wc
  else:
   d[wc]=wc
   w = c
  i+=1
 return len(d)

def str_col(X):
 '''
 Input: Continuous multidimensional time series
 Output: One string being the binarized input matrix concatenated comlumn-by-column
 '''
 ro,co=shape(X)
 TH=zeros(ro)
 M=zeros((ro,co))
 for i in range(ro):
  M[i,:]=abs(hilbert(X[i,:]))
  TH[i]=mean(M[i,:])

 s=''
 for j in range(co):
  for i in range(ro):
   if M[i,j]>TH[i]:
    s+='1'
   else:
    s+='0'

 return s


def LZc(X):
 '''
 Compute LZc and use shuffled result as normalization
 '''
 X=Pre2(X)
 SC=str_col(X)
 M=list(SC)
 shuffle(M)
 w=''
 for i in range(len(M)):
  w+=M[i]
 return cpr(SC)/float(cpr(w))


def LZs(x):

 '''
 Lempel ziv complexity of single timeseries
 '''

 #this differes from Sitt et al as they use 32 bins, not just 2.
 co=len(x)
 x=signal.detrend((x-mean(x))/std(x), axis=0)
 s=''
 r=(abs(hilbert(x)))
 th=mean(r)

 for j in range(co):
  if r[j]>th:
   s+='1'
  else:
   s+='0'

 M=list(s)
 shuffle(M)
 w=''
 for i in range(len(M)):
  w+=M[i]

 return cpr(s)/float(cpr(w))


############
'''
EEG LZ
'''
###########

def EEG_LZ():
    func_dict = {'LZs': LZs}
    conditions = {10, 11}
    city = "Wwa" #Krk
    
    path_dat = '......./Exported_for_LZ/' + city   # path to preprocessed EEG data exported to .mat file 
    path_res = '......./LZs/' + city   # path to save diversity measures files 

    files = os.listdir(path_dat)
    subject_pattern = re.compile(r'Data_(\d+)_mara.mat')
    subjects = sorted([int(subject_pattern.search(f).group(1)) for f in files if subject_pattern.search(f)])
    

    for cond in conditions:
        for subject in subjects:
            data_file = f'{path_dat}/Data_{subject}_mara.mat'
            d = loadmat(data_file)['EEGCond'][0,cond-1]['data']
                
            if isinstance(d, np.ndarray) and d.shape == (1, 1):
                d = d[0, 0]
                
            chs, obs, trs = shape(d)
            print(subject, shape(d))
                
            for measure in func_dict:
                if measure == 'LZs':
                    S = []
                    for t in range(trs):
                        s0 = []
                        for jj in range(chs):
                            s0.append(LZs(d[jj, :, t]))
                        print(cond, subject, measure, t, 'of', trs)
                        S.append(s0)
                    S = array(S)

                if measure == 'LZc':
                    S = []
                    for t in range(trs):
                        S.append(LZc(d[:, :, t]))
                        print(cond, subject, roi, measure, t, 'of', trs)
                    S = array(S)
            
            
                savetxt(f'{path_res}/{measure}_Sub_{subject}_Cond_{cond}.txt', S)
                
                
EEG_LZ()