import sys,json
from pathlib import Path
from PIL import Image,ImageChops
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analysis_tools import final_synthesis as s
from analysis_tools import final_presentation as p
s.verify_inputs();p.phase43();rows=[]
for r in s.read('final_figures/figure_index.csv').itertuples():
    a=Image.open(s.PACKAGE/'inputs'/r.png).convert('RGB');b=Image.open(s.ROOT/r.png).convert('RGB');assert a.size==b.size
    assert ImageChops.difference(a,b).getbbox() is None,r.figure
    rows.append(dict(figure=r.figure,pixel_identical=True))
s.dump('figure_validation.json',dict(passed=True,figures=len(rows),checks=rows,meaning='All 13 PNG pixel arrays exactly reproduce in the recorded environment. PDF/SVG metadata bytes can vary.'))
print('All 13 scientific figures reproduced pixel-identically.')
