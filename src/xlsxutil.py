"""
Pure-stdlib xlsx reader (fallback when pandas is unavailable)
- parse workbook.xml to obtain the sheet names
- parse sharedStrings.xml
- stream-parse sheetN.xml and return list[list[str]]
"""
import zipfile, re
from xml.etree import ElementTree as ET

NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
RNS = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
PKGREL = '{http://schemas.openxmlformats.org/package/2006/relationships}'


def _col_idx(ref):
    m = re.match(r'([A-Z]+)', ref or '')
    if not m:
        return 0
    s = m.group(1)
    n = 0
    for ch in s:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def read_xlsx(path, max_rows=None):
    """Return (sheet_names:list[str], data_by_sheet: dict[name, list[list[str]]])"""
    z = zipfile.ZipFile(path)
    names = z.namelist()

    # shared strings
    shared = []
    if 'xl/sharedStrings.xml' in names:
        root = ET.fromstring(z.read('xl/sharedStrings.xml'))
        for si in root.findall(NS + 'si'):
            shared.append(''.join(t.text or '' for t in si.iter(NS + 't')))

    # workbook -> sheets + rels
    wb = ET.fromstring(z.read('xl/workbook.xml'))
    rels = {}
    if 'xl/_rels/workbook.xml.rels' in names:
        rr = ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
        for rel in rr:
            rels[rel.get('Id')] = rel.get('Target')

    sheets = []
    for sh in wb.find(NS + 'sheets'):
        rid = sh.get(RNS + 'id')
        tgt = rels.get(rid, '')
        tgt = tgt.lstrip('/')
        if not tgt.startswith('xl/'):
            tgt = 'xl/' + tgt
        sheets.append((sh.get('name'), tgt))

    out = {}
    sheet_names = []
    for sname, tgt in sheets:
        sheet_names.append(sname)
        if tgt not in names:
            out[sname] = []
            continue
        rows = []
        with z.open(tgt) as fh:
            for ev, el in ET.iterparse(fh, events=('end',)):
                if el.tag == NS + 'row':
                    vals = {}
                    maxc = -1
                    for c in el.findall(NS + 'c'):
                        r = c.get('r')
                        ci = _col_idx(r)
                        t = c.get('t')
                        v = c.find(NS + 'v')
                        isel = c.find(NS + 'is')
                        if t == 'inlineStr' and isel is not None:
                            val = ''.join(x.text or '' for x in isel.iter(NS + 't'))
                        elif v is None:
                            val = ''
                        elif t == 's':
                            try:
                                val = shared[int(v.text)]
                            except Exception:
                                val = v.text or ''
                        elif t == 'b':
                            val = 'TRUE' if v.text == '1' else 'FALSE'
                        else:
                            val = v.text or ''
                        vals[ci] = val
                        if ci > maxc:
                            maxc = ci
                    if maxc >= 0:
                        rows.append([vals.get(i, '') for i in range(maxc + 1)])
                    else:
                        rows.append([])
                    el.clear()
                    if max_rows and len(rows) >= max_rows:
                        break
        out[sname] = rows
    return sheet_names, out


def to_float(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == '' or s.lower() in ('na', 'nan', 'none', 'null', '-'):
        return None
    try:
        return float(s)
    except Exception:
        return None
