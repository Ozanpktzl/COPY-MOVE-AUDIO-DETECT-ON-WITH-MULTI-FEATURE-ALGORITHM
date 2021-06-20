import scipy
import numpy as np
from scipy import signal

def enframe(sinyal, nw, inc):

    sinyal_length=len(sinyal) #Toplam sinyal uzunluğu
    if sinyal_length<=nw:
        nf=1
    else: #Aksi takdirde, çerçevenin toplam uzunluğunu hesaplayın
        nf=int(np.ceil((1.0*sinyal_length-nw+inc)/inc))
    pad_length=int((nf-1)*inc+nw) #Tüm çerçeveler toplam düzleştirilmiş uzunluğa kadar toplanır
    zeros=np.zeros((pad_length-sinyal_length,))
    pad_signal=np.concatenate((sinyal,zeros))
    indices=np.tile(np.arange(0,nw),(nf,1))+np.tile(np.arange(0,nf*inc,inc),(nw,1)).T  #nf*nw uzunluğunda bir matris elde etmek için tüm karelerin zaman noktalarını çıkarmaya eşdeğerdir
    indices=np.array(indices,dtype=np.int32)
    frames=pad_signal[indices] #çerçeve sinyalini al

    window= signal.windows.hann(nw+2)[1:-1] #hann pencereleme


    return frames*window









def karsılaştırma(kard,karb,deger2x):

    minx=min(kard[1],karb[1])
    r=-999999999

    x=deger2x[int(kard[0]):int(kard[0]+kard[1])]
    y=deger2x[int(karb[0]):int(karb[0]+kard[1])]

    sum1 = np.sum(x)
    sum2 = np.sum(y)

    if (sum1 == 0 and sum2 == 0):
        return 1
    elif (sum1 == 0 and sum2 != 0):
        return 0
    elif (sum1 != 0 and sum2 == 0):
        return 0

    for i in range(int(abs(kard[1]-karb[1]))):

        y=deger2x[int(karb[0]+i):int(karb[0]+kard[1]+i)]
        r=max(r, scipy.stats.pearsonr(x,y)[0])

    r = max(r, scipy.stats.pearsonr(x, y)[0])
    return r



def gelismisBenzerlikBulma(r_matrix ,segmentler,sayac,temp):


    r_matrix0=np.zeros((sayac,sayac))
    r_matrix1=np.zeros((sayac,sayac))
    r_matrix2=np.zeros((sayac,sayac))
    r_matrix3=np.zeros((sayac,sayac))
    r_matrix4=np.zeros((sayac,sayac))
    r_matrix5=np.zeros((sayac,sayac))
    r_matrix6=np.zeros((sayac,sayac))
    r_matrix7=np.zeros((sayac,sayac))
    r_matrix8=np.zeros((sayac,sayac))
    r_matrix9=np.zeros((sayac,sayac))
    r_matrix10=np.zeros((sayac,sayac))
    r_matrix11=np.zeros((sayac,sayac))


    for k in range(12):

        deger=temp[k]

        for i in range(sayac):

            for j in range(sayac):

                if segmentler[i][1]>=segmentler[j][1]:

                    if(k==0):

                        r_matrix0[i][j]=karsılaştırma(segmentler[j],segmentler[i],deger)
                    elif(k==1):
                        r_matrix1[i][j]=karsılaştırma(segmentler[j],segmentler[i],deger)
                    elif(k==2):
                        r_matrix2[i][j] = karsılaştırma(segmentler[j], segmentler[i], deger)
                    elif(k==3):
                        r_matrix3[i][j] = karsılaştırma(segmentler[j], segmentler[i], deger)
                    elif (k == 4):
                        r_matrix4[i][j] = karsılaştırma(segmentler[j], segmentler[i], deger)
                    elif (k == 5):
                        r_matrix5[i][j] = karsılaştırma(segmentler[j], segmentler[i], deger)
                    elif (k == 6):
                        r_matrix6[i][j] = karsılaştırma(segmentler[j], segmentler[i], deger)
                    elif (k == 7):
                        r_matrix7[i][j] = karsılaştırma(segmentler[j], segmentler[i], deger)
                    elif (k == 8):
                        r_matrix8[i][j] = karsılaştırma(segmentler[j], segmentler[i], deger)
                    elif (k == 9):
                        r_matrix9[i][j] = karsılaştırma(segmentler[j], segmentler[i], deger)
                    elif (k == 10):
                        r_matrix10[i][j] = karsılaştırma(segmentler[j],segmentler[i], deger)
                    elif (k == 11):
                        r_matrix11[i][j] = karsılaştırma(segmentler[j],segmentler[i], deger)



                else:
                    if(k==0):
                     r_matrix0[i][j] = karsılaştırma(segmentler[i], segmentler[j], deger)
                    elif(k==1):
                        r_matrix1[i][j] = karsılaştırma(segmentler[i], segmentler[j], deger)

                    elif(k==2):
                        r_matrix2[i][j] = karsılaştırma(segmentler[i], segmentler[j], deger)
                    elif(k==3):
                        r_matrix3[i][j] = karsılaştırma(segmentler[i], segmentler[j], deger)
                    elif (k == 4):
                        r_matrix4[i][j] = karsılaştırma(segmentler[i], segmentler[j], deger)

                    elif (k == 5):
                        r_matrix5[i][j] = karsılaştırma(segmentler[i], segmentler[j], deger)
                    elif (k == 6):
                        r_matrix6[i][j] = karsılaştırma(segmentler[i], segmentler[j], deger)
                    elif (k == 7):
                        r_matrix7[i][j] = karsılaştırma(segmentler[i], segmentler[j], deger)

                    elif (k == 8):
                        r_matrix8[i][j] = karsılaştırma(segmentler[i], segmentler[j], deger)
                    elif (k == 9):
                        r_matrix9[i][j] = karsılaştırma(segmentler[i],segmentler[j], deger)
                    elif (k == 10):
                        r_matrix10[i][j] = karsılaştırma(segmentler[i], segmentler[j], deger)

                    elif (k == 11):
                        r_matrix11[i][j] = karsılaştırma(segmentler[i], segmentler[j], deger)





    r_matrix=r_matrix0+r_matrix1+r_matrix2+r_matrix3+r_matrix4+r_matrix5+r_matrix6+r_matrix7+r_matrix8
    r_matrix=r_matrix+r_matrix9+r_matrix10+r_matrix11

    r_matrix = r_matrix/12.0

    return r_matrix








def benzerlikBulma(r_matrix,segmentler,sayac,deger):
    for i in range(sayac):

        for j in range(sayac):

            if segmentler[i][1] >= segmentler[j][1]:
                r_matrix[i][j] = karsılaştırma(segmentler[j], segmentler[i], deger)

            else:
                r_matrix[i][j] = karsılaştırma(segmentler[i], segmentler[j], deger)





def segmentBulma(segmentler,sayac, kontrol, tak):

    for i in range(len(kontrol)):

        if i > 0:
            tak = 1
        if kontrol[i - 1] == 0 and kontrol[i] != 0 and tak == 1:
            j = i
            while (kontrol[j] != 0 and j < len(kontrol)):
                j += 1

            if j - i >= 10:
                segmentler[sayac][0] = i + 3
                segmentler[sayac][1] = j - i - 4

                sayac += 1
                i = i + j
    return sayac