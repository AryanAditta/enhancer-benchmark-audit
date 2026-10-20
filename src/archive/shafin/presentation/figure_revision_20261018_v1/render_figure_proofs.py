from figure_common import *
from pypdf import PdfReader, PdfWriter
from PIL import ImageDraw, ImageFont
import subprocess
folder=QA/'vector_render_proofs';folder.mkdir(exist_ok=True)
m=sorted(json.loads((WORK/'figure_manifest.json').read_text()),key=lambda x:(x['figure'],x.get('panel') or ''))
w=PdfWriter()
for x in m:w.add_page(PdfReader(OUT/(x['stem']+'.pdf')).pages[0])
proof=folder/'Vector_figure_proofs.pdf'
with proof.open('wb') as stream:w.write(stream)
cmd=[r'C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe','-r','150','-png',str(proof),str(folder/'figure')]
subprocess.run(cmd,cwd=ROOT,env=os.environ.copy(),check=True,creationflags=subprocess.CREATE_NO_WINDOW)
pages=sorted(folder.glob('figure-*.png'))
assert len(pages)==14
font=ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf',20)
for group in range(2):
    canvas=Image.new('RGB',(1500,2240),'#E4E7EA');draw=ImageDraw.Draw(canvas)
    for j,index in enumerate(range(group*7,min(group*7+7,len(pages)))):
        col,row=j%2,j//2;x=col*750;y=row*560
        im=Image.open(pages[index]).convert('RGB');im.thumbnail((720,515),Image.Resampling.LANCZOS)
        canvas.paste(im,(x+(750-im.width)//2,y+35+(515-im.height)//2))
        draw.text((x+15,y+7),m[index]['stem'],font=font,fill='black')
    canvas.save(folder/f'proof_sheet_{group+1}.png')
print('Rendered14 vector masters; proof sheets prepared')
