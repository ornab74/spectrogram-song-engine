import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
base=Path(__file__).parent
j=json.load(open('/mnt/data/quantum_rgb_entropy_results.json'))
probs=list(j['measurement_probabilities'].values())
keys=list(j['measurement_probabilities'].keys())
labels=['streetlights','rain on glass','match flame','neon skyline','engine hum','open doorway','rising smoke','first light']
fig,ax=plt.subplots(figsize=(11.7,4.3),layout='constrained')
for i,p in enumerate(probs):
  ax.bar(i,p, width=.7, color=('#284e8b' if i in (0,3) else '#919db0'))
  ax.text(i,p+0.012,f'{p:.3f}',ha='center',fontsize=9)
ax.set_xticks(range(8), [f'{k}\n{lab}' for k,lab in zip(keys,labels)],fontsize=8)
ax.set_ylabel('Measurement probability')
ax.set_ylim(0,.77)
ax.set_title('Audio-conditioned statevector: probability for each 3-bit outcome',loc='left',fontweight='bold')
ax.spines[['top','right']].set_visible(False)
ax.grid(axis='y',alpha=.2)
fig.savefig(base/'measurement_probabilities.png',dpi=180)
plt.close(fig)

fig,ax=plt.subplots(figsize=(11.5,3.4),layout='constrained')
ax.set_xlim(-.4,14.65);ax.set_ylim(-.7,3.0);ax.axis('off')
xs=[0,1.2,2.35,3.6,5,6.1,7.2,9,10.25,11.55,13.0]
for q in range(3):
 y=2.3-q*.9
 ax.plot([0,14.4],[y,y],color='#24364b',lw=1.5,zorder=0.5)
 ax.text(-.35,y,f'q{q}',ha='right',va='center',fontsize=10,weight='bold')
for q in range(3):
 y=2.3-q*.9
 for x,text in zip(xs[1:4],['H','RY','RZ']):
  ax.add_patch(FancyBboxPatch((x-.35,y-.28),.7,.56,boxstyle='round,pad=0.04',fc='#e6ecf6',ec='#5071a1',lw=1.1,zorder=3))
  ax.text(x,y,text,ha='center',va='center',fontsize=9,weight='bold',zorder=4)
 for x,text in zip(xs[8:10],['RY','RZ']):
  ax.add_patch(FancyBboxPatch((x-.35,y-.28),.7,.56,boxstyle='round,pad=0.04',fc='#eee8f5',ec='#77568a',lw=1.1,zorder=3))
  ax.text(x,y,text,ha='center',va='center',fontsize=9,weight='bold',zorder=4)
# cyclic CNOT gates sequentially
for xc,c,t in [(5,0,1),(6.1,1,2),(7.2,2,0)]:
 yc=2.3-c*.9;yt=2.3-t*.9
 ax.plot([xc,xc],[yc,yt],color='#a54343',lw=1.8)
 ax.scatter([xc],[yc],c='#a54343',s=30,zorder=5)
 circ=plt.Circle((xc,yt),.18,fc='white',ec='#a54343',lw=1.8,zorder=6)
 ax.add_patch(circ)
 ax.plot([xc-.13,xc+.13],[yt,yt],color='#a54343',lw=1.5,zorder=7)
 ax.plot([xc,xc],[yt-.13,yt+.13],color='#a54343',lw=1.5,zorder=7)
for q in range(3):
 y=2.3-q*.9; x=13.0
 ax.add_patch(FancyBboxPatch((x-.43,y-.25),.86,.5,boxstyle='round,pad=0.03',fc='#eaf0e2',ec='#67824d',zorder=3))
 ax.text(x,y,'p(b)',ha='center',va='center',fontsize=9,zorder=4)
ax.text(2.4,2.85,'Audio-conditioned local rotations',ha='center',fontsize=10)
ax.text(6.05,2.85,'Cyclic CNOT layer',ha='center',fontsize=10)
ax.text(10.3,2.85,'Second mixing layer',ha='center',fontsize=10)
fig.savefig(base/'circuit_schematic.png',dpi=180,bbox_inches='tight')
plt.close(fig)
