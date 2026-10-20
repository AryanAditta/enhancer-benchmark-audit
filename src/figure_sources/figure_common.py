from pathlib import Path
import os, json, hashlib, datetime
WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
OUT = WORK / 'staging/Figures'
QA = WORK / 'qa'
for key in ['TMP','TEMP','TMPDIR','MPLCONFIGDIR','XDG_CACHE_HOME']:
    p=WORK/'runtime'/key.lower();p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle, Arc
import numpy as np
import pandas as pd
from PIL import Image
plt.rcParams.update({'font.family':'Arial','font.size':6.2,'axes.titlesize':6.8,'axes.titleweight':'bold',
 'axes.labelsize':6.5,'xtick.labelsize':6,'ytick.labelsize':6,'legend.fontsize':5.8,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,
 'xtick.major.width':.6,'ytick.major.width':.6,'xtick.major.size':2.5,'ytick.major.size':2.5,
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white',
 'figure.facecolor':'white','axes.unicode_minus':False})
BLACK='#222222'; GREY='#727B83'; BLUE='#306B9A'; TEAL='#277D79'; ORANGE='#B9632D'
BL='#DCE9F4'; TL='#DCEFEB'; OL='#F8EBDD'; LIGHT='#F1F3F5'; EDGE='#48525C'
MODEL_COLORS={'dnabert2_selected':BLUE,'legnet_selected':ORANGE,'mprann_selected':TEAL,'spaced6_4_ridge':GREY}
ORDER=['gc_logreg','k3_logreg','spaced6_4_logreg','small_cnn','frozen_dnabert_legacy','frozen_dnabert2','frozen_gena_lm','finetuned_dnabert_legacy','finetuned_dnabert2']
LABELS=['GC/CpG logistic regression','3-mer logistic regression','Spaced 6/4-mer logistic regression','Small CNN','Frozen DNABERT','Frozen DNABERT-2','Frozen GENA-LM','Fine-tuned DNABERT','Fine-tuned DNABERT-2']
INPUTS={}; MANIFEST=[]; POINTS=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(path):
    INPUTS[path]=sha(ROOT/path)
    return pd.read_csv(ROOT/path,float_precision='round_trip')
def status(stage,**extra):
    p=WORK/'status.json';d=json.loads(p.read_text()) if p.exists() else {}
    d.update(stage=stage,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra)
    p.write_text(json.dumps(d,indent=2),encoding='utf-8')
def save(fig,stem,figure,panel=None,sources=()):
    fig.canvas.draw(); r=fig.canvas.get_renderer();outside=[]
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text():continue
        # Tick locators retain off-range labels which are not drawn.
        if t.axes is not None and t.get_transform()==t.axes.get_xaxis_transform():continue
        b=t.get_window_extent(r)
        if b.x0 < -1 or b.y0 < -1 or b.x1 > fig.bbox.width+1 or b.y1 > fig.bbox.height+1:
            outside.append(t.get_text())
    fig.savefig(OUT/(stem+'.pdf'),metadata={'Title':stem,'Author':''})
    fig.savefig(OUT/(stem+'.svg'))
    fig.savefig(OUT/(stem+'.png'),dpi=600)
    with Image.open(OUT/(stem+'.png')) as im:rgb=im.convert('RGB')
    rgb.save(OUT/(stem+'.png'),dpi=(600,600))
    rgb.save(OUT/(stem+'.tiff'),dpi=(600,600),compression='tiff_lzw')
    rgb.thumbnail((1800,1800),Image.Resampling.LANCZOS);rgb.save(OUT/(stem+'_preview.png'))
    MANIFEST.append(dict(stem=stem,figure=figure,panel=panel,width_mm=fig.get_figwidth()*25.4,
                         height_mm=fig.get_figheight()*25.4,sources=list(sources),outside_text_candidates=outside))
    plt.close(fig)
def plotfig(h=78,left=.13,bottom=.18,right=.97,top=.86):
    fig=plt.figure(figsize=(180/25.4,h/25.4));ax=fig.add_axes([left,bottom,right-left,top-bottom]);return fig,ax
def plabel(fig,letter,title):
    fig.text(.014,.957,letter,fontsize=8,fontweight='bold',va='top')
    fig.text(.05,.957,title,fontsize=6.8,fontweight='bold',va='top')
def finish():
    (QA/'figure_inputs.json').write_text(json.dumps(INPUTS,indent=2),encoding='utf-8')
    (WORK/'figure_manifest.json').write_text(json.dumps(MANIFEST,indent=2),encoding='utf-8')
    pd.DataFrame(POINTS).to_csv(QA/'plotted_values.csv',index=False,float_format='%.17g')

class Diagram:
    def __init__(self,h):
        self.h=h;self.fig=plt.figure(figsize=(180/25.4,h/25.4));self.ax=self.fig.add_axes([0,0,1,1]);self.ax.set(xlim=(0,180),ylim=(h,0));self.ax.axis('off');self.texts=[]
    def t(self,x,y,s,fs=6.1,bold=False,ha='center',va='center'):
        a=self.ax.text(x,y,s,fontsize=fs,fontweight='bold' if bold else 'normal',ha=ha,va=va,color=BLACK,linespacing=1.18,zorder=5);self.texts.append(a);return a
    def box(self,x,y,w,h,txt='',fill=LIGHT,fs=6.1,dashed=False):
        self.ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=.7',facecolor=fill,edgecolor=EDGE,linewidth=.6,linestyle=(0,(3,2)) if dashed else '-'))
        if txt:self.t(x+w/2,y+h/2,txt,fs)
    def rect(self,x,y,w,h,fill=LIGHT):self.ax.add_patch(Rectangle((x,y),w,h,facecolor=fill,edgecolor=EDGE,linewidth=.4))
    def line(self,pts,lw=.65):self.ax.plot(*zip(*pts),color=EDGE,lw=lw,zorder=2)
    def arrow(self,x,y,xx,yy):self.ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=5,shrinkA=0,shrinkB=0,linewidth=.65,color=EDGE,zorder=3))
    def panel(self,a,title,y):self.t(2,y,a,8,True,ha='left');self.t(8,y,title,7,True,ha='left')
    def seq(self,x,y,w=22,letters=True):
        for i,n in enumerate('AACGTAGC'):
            self.rect(x+i*w/8,y,w/8,3,[BL,'white','#BCD3E6','#E7EFF7']['ACGT'.index(n)])
            if letters:self.t(x+(i+.5)*w/8,y+1.5,n,5.2)
    def matrix(self,x,y,w=15,h=10,rows=4,cols=7,fill=BL):
        for i in range(rows):
            for j in range(cols):self.rect(x+j*w/cols,y+i*h/rows,w/cols,h/rows,fill if (i+j)%3 else 'white')
    def pooling(self,x,y):
        self.matrix(x,y,12,10);self.arrow(x+13,y+5,x+17,y+5);self.matrix(x+18,y+3,2.5,4,rows=4,cols=1,fill=TL)
    def check(self,stem,num,panel=None,sources=()):
        self.fig.canvas.draw();r=self.fig.canvas.get_renderer();over=[]
        for i,a in enumerate(self.texts):
            for b in self.texts[i+1:]:
                if a.get_window_extent(r).overlaps(b.get_window_extent(r)):over.append([a.get_text(),b.get_text()])
        save(self.fig,stem,num,panel,sources)
        MANIFEST[-1]['text_bbox_overlaps']=over
