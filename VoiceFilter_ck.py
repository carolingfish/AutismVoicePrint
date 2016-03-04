import numpy as np
import soundfile as sf
from soundfile import SEEK_END
import math

def GetSoundFile(filePath):
    s = sf.SoundFile(filePath, 'r')
    sdata = s.read()
    s.close()
    # only look at left channel for now, index 0
    if len(sdata.shape) == 2:
        return sdata[:,0]
    else:
        return sdata

def GetStatistic(sdata):
    return max(sdata), min(sdata), np.mean(sdata), np.std(sdata)

def FilterLowEnergy(sdata, fps = 44100, WDIS=50.0*10**-3, SDIS = 10.0*10**-3):
    WindowFrameCount = fps*WDIS
    ShiftFrameCount = fps*SDIS
    
    smax, smin, smean, sstd = GetStatistic(sdata)
    filelength = sdata.shape[0]
    
    energies = []
    
    for windowIndex in range(0, filelength,int(ShiftFrameCount)):
        windowStartIndex = windowIndex
        windowEndIndex = min((int)(windowIndex + WindowFrameCount), filelength)
        thiswindow = sdata[windowStartIndex:windowEndIndex]
        energy = np.dot(thiswindow,thiswindow) / (windowEndIndex-windowStartIndex)
        energies.append(energy)
        
    emax, emin, emean, estd = GetStatistic(energies)
    
    sdataEF = np.zeros(0)
    for windowIndex in range(0, filelength,int(ShiftFrameCount)):
        windowStartIndex = windowIndex
        windowEndIndex = min((int)(windowIndex + WindowFrameCount), filelength)
        middleIndex = (windowEndIndex - windowStartIndex) / 2.0 + windowStartIndex
        shiftStartIndex = min((int)(middleIndex - ShiftFrameCount / 2.0) , filelength)
        shiftEndIndex = min((int)(middleIndex + ShiftFrameCount / 2.0), filelength)
        thiswindow = sdata[windowStartIndex:windowEndIndex]
        energy = np.dot(thiswindow,thiswindow) / (windowEndIndex-windowStartIndex)
        if ((energy - emin)/estd > 0.05):
            sdataEF = np.insert(sdataEF,len(sdataEF),sdata[shiftStartIndex:shiftEndIndex])
    
    return sdataEF

def GetMaxDiff(nparr):
    nparr = np.sort(nparr)
    ds = np.diff(nparr)
    return np.max(ds)

def PitchFilter(sdata, fps = 44100, WDIS = 50.0*10**-3, SDIS = 10.0*10**-3):
    WindowFrameCount = fps*WDIS
    ShiftFrameCount = fps*SDIS
    
    corrResults = []
    filelength = len(sdata)
    for windowIndex in range(0, filelength,int(ShiftFrameCount)):
        windowStartIndex = windowIndex
        windowEndIndex = min((int)(windowIndex + WindowFrameCount), filelength)
        shiftEndIndex = min((int)(windowIndex + ShiftFrameCount), filelength)
        thiswindow = sdata[windowStartIndex:windowEndIndex]
        result = np.correlate(thiswindow, thiswindow, mode='full')
        autoCorr = result[result.size/2:]
        corrResults.append(autoCorr)
        
    fd2 = []
    for ys in corrResults:
        fd = np.zeros(ys.shape)
        for i in range(0, len(ys)):
            fd[i] = np.max(ys[i:len(ys)])
        fd2.append(fd)
        
    fd3 = []
    for ys in fd2:
        fd3.append(np.diff(ys))
        
    fd4 = []
    sc = 20
    for ys in fd3:
        fd = np.zeros(ys.shape)
        for i in range(0, len(ys) - sc):
            fd[i] = np.max(ys[i:i+sc])
        fd4.append(fd)
        
    fd5 = []
    for ys in fd4:
        fd5.append(np.diff(ys))
    
    final = fd5
        
    s3data= np.zeros(0)
    i = 0
    finalll = []
    for windowIndex in range(0, filelength,int(ShiftFrameCount)):
        windowStartIndex = windowIndex
        windowEndIndex = min(windowIndex + WindowFrameCount, filelength)
        middleIndex = (windowEndIndex - windowStartIndex) / 2.0 + windowStartIndex
        shiftStartIndex = min((int)(middleIndex - ShiftFrameCount / 2.0) , filelength)
        shiftEndIndex = min((int)(middleIndex + ShiftFrameCount / 2.0), filelength)
        inxes = final[i].argsort()[-2:][::-1]
        dur = GetMaxDiff(inxes)
        freq = fps*1.0/dur
        if freq > 50.0 and freq < 500.0:
            s3data = np.insert(s3data,len(s3data),sdata[shiftStartIndex:shiftEndIndex])
        i+=1
        
    return s3data

