import AMFM_decompy.amfm_decompy.pYAAPT as pYAAPT
import AMFM_decompy.amfm_decompy.basic_tools as basic
import scipy
import librosa
from librosa import display
import numpy as np
from matplotlib import pyplot as plt
import Fonksiyonlar as Fonk
from brian2 import *
from brian2hears import *
from gammatone_master.gammatone.gtgram import gtgram_xe, Gtgram



parametre = {'bp_forder': 150, 'bp_low': 50.0, 'bp_high': 1500.0,
             'dec_factor': 1}




signal = basic.SignalObj('uzun.wav')

plot_a = plt.subplot(211)
plot_a.plot(signal.data)
plot_a.set_xlabel('sample rate * time')
plot_a.set_ylabel('energy')
plt.show()


## pitch kısmı



pitch = pYAAPT.yaapt(signal, **{'f0_min' : 150.0, 'frame_length' : 15.0, 'frame_space' : 5.0})
pitch_interp=np.copy(pitch.samp_interp)
pitch_values=np.copy(pitch.samp_values)


plt.plot(pitch.samp_values, label='samp_values', color='blue')
plt.xlabel('frames', fontsize=18)
plt.ylabel('pitch (Hz)', fontsize=18)
plt.show()



segmentler=np.zeros((30,2))
sayac=0
tak=0  ## dizinin sıfırdan buyuklugunu test eder

sayac = Fonk.segmentBulma(segmentler,sayac,pitch_values,tak)
segmentler= segmentler[0:sayac]




r_matrixPitch=np.zeros((sayac,sayac))
Fonk.benzerlikBulma(r_matrixPitch,segmentler,sayac,pitch_interp)




print("\n")
print("pitch benzerlik oranı: \n")
print(r_matrixPitch)

## mfcc kısmı

lmfcc=librosa.feature.mfcc(signal.data,signal.fs);

fig, ax = plt.subplots()
img = librosa.display.specshow(lmfcc, x_axis='time', ax=ax,fmin=150)
fig.colorbar(img, ax=ax)
ax.set(title='MFCC')
plt.show()

##mel spect

S = librosa.feature.melspectrogram(y=signal.data, sr=signal.fs,n_fft=8192, n_mels=128,
                                    fmax=8000)
fig, ax = plt.subplots()
S_dB = librosa.power_to_db(S, ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='time',
                         y_axis='mel', sr=signal.fs,
                         fmax=8000, ax=ax,fmin=150)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Mel-frequency spectrogram')
plt.show()





##rms

S, phase = librosa.magphase(librosa.stft(signal.data))
rms = librosa.feature.rms(S=S)

fig, ax = plt.subplots(nrows=2, sharex=True)
times = librosa.times_like(rms)
ax[0].semilogy(times, rms[0], label='RMS Energy')
ax[0].set(xticks=[])
ax[0].legend()
ax[0].label_outer()
librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                         y_axis='log', x_axis='time', ax=ax[1])
ax[1].set(title='log Power spectrogram')
plt.show()





##chroma ile

chroma=librosa.feature.chroma_cqt(y=signal.data, sr=signal.fs,fmin=150,threshold=0.75)

chromaDeger=np.sum(chroma,axis=0)



fig, ax = plt.subplots()
img = librosa.display.specshow(chroma, x_axis='time', ax=ax)
fig.colorbar(img, ax=ax)
ax.set(title='CHROMA')
plt.show()


segmentler=np.zeros((30,2))
sayac=0
tak=0

sayac=Fonk.segmentBulma(segmentler,sayac,chromaDeger,tak)
segmentler= segmentler[0:sayac]

r_matrixChroma=np.zeros((sayac,sayac))

Fonk.benzerlikBulma(r_matrixChroma,segmentler,sayac,chromaDeger)

print("\n")
print("chroma benzerlik oranı: \n")
print(r_matrixChroma)







##Gelismis benzerlik chroma


segmentler=np.zeros((30,2))
sayac=0
tak=0

sayac=Fonk.segmentBulma(segmentler,sayac,chromaDeger,tak)
segmentler= segmentler[0:sayac]
r_matrixChromaG=np.zeros((sayac,sayac))

r_matrixChromaG=Fonk.gelismisBenzerlikBulma(r_matrixChromaG,segmentler,sayac,chroma)


print("\n")
print("chroma Gelismis benzerlik oranı: \n")
print(r_matrixChromaG)





##gammotene kısmı Gelismis benzerlik


gfcc=Gtgram(signal.data,signal.fs,0.030,0.015,channels=12,f_min=150)

gfcc=gfcc*1000


gfccDeger=np.sum(gfcc,axis=0)

gfccKontrol=np.zeros(len(gfccDeger))

for i in range(len(gfccDeger)):

    if (gfccDeger[i]>1):
        gfccKontrol[i]=gfccDeger[i]
    else :
        gfccKontrol[i]=0





fig, ax = plt.subplots()
img=librosa.display.specshow(gfcc, fmin=150,sr=signal.fs, fmax=8000, ax=ax,x_axis='time')
fig.colorbar(img, ax=ax)
ax.set(title='GFCC')
plt.show()


segmentler=np.zeros((30,2))
sayac=0
tak=0

sayac=Fonk.segmentBulma(segmentler,sayac,gfccKontrol,tak)
segmentler= segmentler[0:sayac]

r_matrixGfcc=np.zeros((sayac,sayac))



r_matrixGfcc=Fonk.gelismisBenzerlikBulma(r_matrixGfcc,segmentler,sayac,gfcc)

print("\n")
print("GFCC benzerlik oranı: \n")
print(r_matrixGfcc)


##DFT coefficients

frame_size = int(np.fix(15 * signal.fs / 1000))
frame_jump = int(np.fix(5* signal.fs / 1000))

fir_filter = pYAAPT.BandpassFilter(signal.fs,parametre)
signal.filtered_version(fir_filter)

nframe=Fonk.enframe(signal.filtered,frame_size,frame_jump)

specData=np.fft.fft(nframe,8192)



frame_energy = np.abs(specData[:, int(60 - 1):int(400)]).sum(axis=1)


frame_mean=np.mean(frame_energy)
frame_energy=frame_energy/frame_mean


frame_energyKontrol=np.zeros(len(frame_energy))

for i in range(len(frame_energy)):

    if (frame_energy[i]>1):
        frame_energyKontrol[i]=frame_energy[i]
    else :
        frame_energyKontrol[i]=0


plot_a = plt.subplot(211)
plot_a.plot(frame_energyKontrol)
plot_a.set_xlabel('sample rate * time')
plot_a.set_ylabel('energy')
plt.show()


segmentler=np.zeros((30,2))
sayac=0
tak=0

sayac=Fonk.segmentBulma(segmentler,sayac, frame_energyKontrol,tak)
segmentler= segmentler[0:sayac]

r_matrixDft=np.zeros((sayac,sayac))

Fonk.benzerlikBulma(r_matrixDft,segmentler,sayac,frame_energyKontrol)

print("\n")
print("DFTbenzerlik oranı: \n")
print(r_matrixDft)



## c4.5 based detection

r_matrix4G=r_matrixPitch+r_matrixChromaG+r_matrixGfcc+r_matrixDft
r_matrix4G=r_matrix4G/4.0

print("\n")
print("c4.5 based detection benzerlik oranı: \n")
print(r_matrix4G)
print("\n")

i=sayac-1
while(i>=0):

    for j in range(i):

        if (r_matrix4G[i][j]>=0.98 and i!=j):

            print(i,".segment ile ",j,"segment birbirinden kopyalanmistir.\n")


    i=i-1














