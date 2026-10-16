"""Audio-conditioned RGB 3-qubit statevector entropy explorer (NumPy only for circuit).

Inspired by PennyLane's gate-and-measurement workflow; does not import or use PennyLane.
The quantum state/probabilities are simulated classically. Samples are from NumPy's
seeded pseudo-random number generator, NOT quantum-hardware randomness.

Usage: python quantum_rgb_entropy.py input.wav
Output: band energy ratios, gate angles, probabilities, Shannon entropy, and
cohesive imagery/rhyme prompts (never explicitly inserted in generated lyrics).
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

I=np.eye(2,dtype=np.complex128)
H=np.array([[1,1],[1,-1]],dtype=np.complex128)/np.sqrt(2)

def ry(a):
    return np.array([[np.cos(a/2),-np.sin(a/2)],[np.sin(a/2),np.cos(a/2)]],dtype=complex)

def rz(a):
    return np.array([[np.exp(-.5j*a),0],[0,np.exp(.5j*a)]],dtype=complex)

def on(q,gate):
    """Full 8x8 matrix with q0 as the most significant qubit."""
    mats=[gate if i==q else I for i in range(3)]
    return np.kron(np.kron(mats[0],mats[1]),mats[2])

def cnot(control,target):
    mat=np.zeros((8,8),dtype=np.complex128)
    cbit=1 << (2-control)
    tbit=1 << (2-target)
    for state in range(8):
        dest=(state^tbit) if state&cbit else state
        mat[dest,state]=1
    return mat

def audio_rgb(wav):
    """R=low (40–250Hz), G=mid (250–2000Hz), B=high (2–6000Hz)."""
    audio,sr=sf.read(wav,always_2d=True,dtype='float32')
    mono=audio.mean(axis=1)
    # reproducible audio-conditioned RNG seed, not cryptographic randomness
    seed=int.from_bytes(hashlib.sha256(mono[::257].tobytes()).digest()[:8],'little')
    # 12 kHz suffices for <= 6kHz; small 8192-sample FFTs drawn across the song
    if sr != 12000:
        from math import gcd
        d=gcd(sr,12000)
        mono=resample_poly(mono,12000//d,sr//d)
        sr=12000
    nfft=8192; num=128
    pos=np.linspace(0,max(0,len(mono)-nfft),num).astype(int)
    window=np.hanning(nfft)
    psd=np.zeros(nfft//2+1)
    for start in pos:
        frame=mono[start:start+nfft]
        if len(frame)<nfft: frame=np.pad(frame,(0,nfft-len(frame)))
        psd+=np.abs(np.fft.rfft(frame*window))**2
    freqs=np.fft.rfftfreq(nfft,d=1/sr)
    bounds=[(40,250),(250,2000),(2000,6000)]
    energies=np.array([psd[(freqs>=lo)&(freqs<hi)].sum() for lo,hi in bounds])
    rgb=energies/energies.sum()
    return rgb,seed

def circuit(rgb):
    """Two layers of single-qubit rotations + cyclic CNOT entanglement."""
    angles=np.pi*(.2+1.6*np.sqrt(rgb))
    state=np.zeros(8,dtype=complex);state[0]=1
    for q in range(3):
        state=on(q,H)@state
        state=on(q,ry(angles[q]))@state
        state=on(q,rz(np.pi*rgb[(q+1)%3]))@state
    for c,t in [(0,1),(1,2),(2,0)]:
        state=cnot(c,t)@state
    for q in range(3):
        state=on(q,ry(angles[(q+1)%3]*.47))@state
        state=on(q,rz(angles[(q+2)%3]*.37))@state
    return state,angles

def select_ideas(prob, seed):
    rng=np.random.default_rng(seed)
    shots=rng.choice(8,4096,p=prob)
    # Theme: return + release. Motifs are intentionally drawn from one cohesive world.
    images=['streetlights','rain on glass','match flame','neon skyline',
            'engine hum','open doorway','rising smoke','first light']
    rhyme_families=[['night','light','bright'],['rain','again','remain'],
       ['fire','higher','wire'],['glow','slow','go'],
       ['drive','alive','survive'],['door','more','before'],
       ['smoke','broke','woke'],['day','away','stay']]
    # Select one principal motif, one complement; don't jump images every line.
    counts=np.bincount(shots,minlength=8)
    top=np.argsort(-counts)
    first=int(top[0]);second=int(next(i for i in top if i!=first))
    # use sampled outcome to choose a secondary emotional verb, not the story itself
    verb=['release','rise','ignite','return','move','hold','breathe','begin'][int(shots[-1])]
    return {'anchor_motif':images[first], 'secondary_motif':images[second],
            'anchor_rhymes':rhyme_families[first],
            'secondary_rhymes':rhyme_families[second],
            'emotional_action':verb, 'top_outcomes':[format(int(i),'03b') for i in top[:4]],
            'observed_top_counts':[int(counts[i]) for i in top[:4]]}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('wav',type=Path)
    ap.add_argument('--json',type=Path,help='Optional JSON output path')
    args=ap.parse_args()
    rgb,seed=audio_rgb(args.wav)
    state,angles=circuit(rgb)
    p=np.abs(state)**2
    p=p/p.sum()
    Hbits=float(-np.sum(p[p>0]*np.log2(p[p>0])))
    result={'frequency_bands_hz':{'R':[40,250],'G':[250,2000],'B':[2000,6000]},
            'relative_audio_band_energy':dict(zip(['R','G','B'],[round(float(i),6) for i in rgb])),
            'seed_from_audio':int(seed),
            'ry_angles_radians':[round(float(i),6) for i in angles],
            'measurement_probabilities':{format(i,'03b'):round(float(v),6) for i,v in enumerate(p)},
            'shannon_entropy_bits':round(Hbits,6),
            'normalized_entropy':round(Hbits/3,6),
            'cohesion_constraints':{'story_arc':'trapped -> resistance -> release -> renewal',
             'motif_limit':2,'keep_hook_identical_each_repeat':True},
            'lyric_idea_selections':select_ideas(p,seed)}
    print(json.dumps(result,indent=2))
    if args.json:args.json.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
