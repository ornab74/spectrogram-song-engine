import numpy as np, soundfile as sf, scipy.signal as sig, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, librosa, json
p='/mnt/data/Punchier Pre-Chorus and Chorus (Extended) (1) (1) (1).wav'
y,sr=sf.read(p,dtype='float32',always_2d=True)
mono=y.mean(axis=1)
fs=12000
x=sig.resample_poly(mono,1,4)
hop=256;nfft=2048
freq,tim,z=sig.stft(x,fs=fs,nperseg=nfft,noverlap=nfft-hop,boundary=None)
mag=np.abs(z)
db=20*np.log10(np.maximum(mag,1e-7))
# compute spectral flux and RMS via framewise STFT
flux=np.maximum(np.diff(mag,axis=1),0).mean(axis=0)
frms=np.sqrt(np.mean(mag**2,axis=0))
# interval stats by 8 sec
sections=[]
for s in np.arange(0,len(x)/fs,8):
    mask=(tim>=s)&(tim<s+8)
    if not mask.any():continue
    sections.append(dict(start=round(float(s),1),end=round(float(min(s+8,len(x)/fs)),1),rms=round(float(np.sqrt(np.mean(x[int(s*fs):int(min(s+8,len(x)/fs)*fs)]**2))),4),spectral_centroid=round(float(np.sum(freq[:,None]*mag[:,mask])/np.sum(mag[:,mask])),1),flux=round(float(np.mean(flux[np.maximum(np.where(mask)[0]-1,0)])),7)))
# calculate onset, tempo on 12k array with librosa (one-time)
onset=librosa.onset.onset_strength(y=x,sr=fs,hop_length=hop)
tempo=librosa.feature.tempo(onset_envelope=onset,sr=fs,hop_length=hop,aggregate=np.median)
beat_tempo,beat_idx=librosa.beat.beat_track(onset_envelope=onset,sr=fs,hop_length=hop,trim=False)
beat_times=librosa.frames_to_time(beat_idx,sr=fs,hop_length=hop)
# chroma STFT quick map: resample mag to semitone bins, ignore subbass, key profile correlation
chroma=librosa.feature.chroma_stft(S=(mag*1000)**2,sr=fs,hop_length=hop,n_fft=nfft)
ch=np.mean(chroma,axis=1)
keys=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
maj=np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
minr=np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
ranked=[]
for root in range(12):
    for mode,prof in [('major',maj),('minor',minr)]:
        correlation=float(np.corrcoef(ch,np.roll(prof,root))[0,1])
        ranked.append((keys[root],mode,round(correlation,3)))
ranked.sort(key=lambda p:-p[2])
# plotting: spectrogram time vs freq, 0-7k and RMS / flux
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(3,1,figsize=(16,9),sharex=True,gridspec_kw={'height_ratios':[3,1,1]},layout='constrained')
S=db-db.max(); vmax=0;vmin=-72
pc=axs[0].pcolormesh(tim,freq/1000,S,shading='auto',cmap='magma',vmin=vmin,vmax=vmax,rasterized=True)
axs[0].set_ylim(0,6);axs[0].set_ylabel('Frequency (kHz)');axs[0].set_title('Track spectrogram | Relative intensity (dB)',loc='left',fontweight='bold',fontsize=16)
fig.colorbar(pc,ax=axs[0],label='Relative level (dB)',shrink=.8)
axs[1].plot(tim,20*np.log10(frms+1e-7),color='#2768b2',lw=.9)
axs[1].set_ylabel('Broadband level (dB)');axs[1].grid(alpha=.2)
axs[2].plot(librosa.times_like(onset,sr=fs,hop_length=hop),onset,color='#b44666',lw=.8)
axs[2].set_ylabel('Onset energy');axs[2].set_xlabel('Track time (seconds)');axs[2].set_xlim(0,len(x)/fs);axs[2].grid(alpha=.2)
fig.savefig('/mnt/data/track_spectrogram.png',dpi=150,bbox_inches='tight')
# waveform envelope dynamics
plt.close(fig)
# get top section boundary changes via energy deltas and novelty
sec=np.asarray([r['rms'] for r in sections]);diff=np.diff(sec)
boundaries=sorted([(round(float(abs(v)),4),float(sections[i+1]['start']), 'increase' if v>0 else 'decrease') for i,v in enumerate(diff)],reverse=True)[:12]
res={'duration':len(x)/fs,'channels':2,'sample_rate':sr,'global_rms':float(np.sqrt(np.mean(x*x))),'peak':float(np.max(abs(x))),'bpm_tempo_est':np.atleast_1d(tempo).tolist(),'bpm_beat_tracker':np.atleast_1d(beat_tempo).tolist(),'beat_times_first':beat_times[:25].round(2).tolist(),'beat_count':len(beat_times),'chroma':{keys[i]:round(float(v),3) for i,v in enumerate(ch)},'estimated_keys':ranked[:8],'sections_8s':sections,'energy_changes':boundaries}
with open('/mnt/data/track_analysis.json','w') as f:json.dump(res,f,indent=2)
print(json.dumps({k:v for k,v in res.items() if k!='sections_8s'},indent=2))
print('SECTIONS')
for sec in sections:print(sec)
