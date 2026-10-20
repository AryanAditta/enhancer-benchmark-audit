"""Read-only validation and QA contact sheets for new finalization artifacts."""
from pathlib import Path
import sys,os,json,zipfile,xml.etree.ElementTree as E,math,subprocess
from analysis_tools import final_synthesis as s
from analysis_tools.final_documents import runtime,DEPS
runtime();ROOT=s.ROOT

def workbook():
    payload=s.jd('phase44_outputs/workbook_payload.json');p=ROOT/'final_supplement/Supplementary_Tables.xlsx'
    ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    checked=0;max_error=0.;rows=[]
    with zipfile.ZipFile(p) as z:
        assert z.testzip() is None
        strings=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            strings=[''.join(n.itertext()) for n in E.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',ns)]
        def parse(sheet):
            cells={}
            for c in E.fromstring(z.read(sheet)).findall('.//s:sheetData/s:row/s:c',ns):
                kind=c.attrib.get('t');v=c.find('s:v',ns)
                if kind=='inlineStr':value=''.join(c.find('s:is',ns).itertext())
                elif v is None:value=None
                elif kind=='s':value=strings[int(v.text)]
                elif kind=='b':value=v.text=='1'
                elif kind in ['str','e']:value=v.text
                else:value=float(v.text)
                cells[c.attrib['r']]=value
            return cells
        def letters(n):
            out=''
            while n:n,k=divmod(n-1,26);out=chr(65+k)+out
            return out
        for number,t in enumerate(payload['tables'],2):
            cells=parse(f'xl/worksheets/sheet{number}.xml')
            for i,row in enumerate([t['columns']]+t['data'],1):
                for j,value in enumerate(row,1):
                    got=cells.get(letters(j)+str(i))
                    if value is None:assert got is None,(t['name'],i,j,got)
                    elif isinstance(value,(float,int)) and not isinstance(value,bool):
                        err=abs(got-value);assert err<=1e-12*max(1,abs(value)),(t['name'],i,j,got,value)
                        max_error=max(max_error,err)
                    else:assert got==value,(t['name'],i,j,got,value)
                    checked+=1
            rows.append(dict(sheet=t['name'],data_rows=len(t['data']),columns=len(t['columns']),all_values_checked=True))
    s.dump('phase44_outputs/workbook_value_validation.json',dict(passed=True,all_data_cells_checked=checked,maximum_absolute_numeric_roundtrip_error=max_error,sheets=rows,workbook_sha256=s.sha(p),precision='XLSX stores up to15 significant digits; full round-trip CSVs retained',formula_policy='Static scientific result export; no invented Excel calculations'))
    print('Workbook values validated:',checked,'cells')

def render_pdf(path,folder,dpi=130):
    output=ROOT/folder;output.mkdir(parents=True,exist_ok=True)
    executable=DEPS/'native/poppler/Library/bin/pdftoppm.exe'
    subprocess.run([str(executable),'-r',str(dpi),'-png',str(ROOT/path),str(output/'page')],cwd=ROOT,env=os.environ.copy(),check=True,creationflags=subprocess.CREATE_NO_WINDOW)

def contact_sheets(folder,prefix,per_page=6,width=1800,height=1800):
    from PIL import Image,ImageDraw
    folder=ROOT/folder
    paths=sorted(p for p in folder.glob('*.png') if not p.name.startswith('contact_'))
    for batch in range(math.ceil(len(paths)/per_page)):
        canvas=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(canvas)
        for j,p in enumerate(paths[batch*per_page:(batch+1)*per_page]):
            im=Image.open(p).convert('RGB');im.thumbnail((width//2-25,height//3-45))
            x=(j%2)*(width//2);y=(j//2)*(height//3)
            draw.text((x+10,y+8),p.name,fill='black');canvas.paste(im,(x+10,y+30))
        canvas.save(folder/f'contact_{prefix}_{batch+1:02}.png')
    print('Contact sheets:',len(paths),'previews',folder)

if __name__=='__main__':
    workbook()
    render_pdf('final_supplement/Supplementary_Information.pdf','phase44_outputs/PDF_QA')
    contact_sheets('phase44_outputs/PDF_QA','pdf')
    contact_sheets('phase44_outputs/workbook_QA','xlsx')
