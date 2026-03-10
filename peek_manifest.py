import json, pathlib
m = pathlib.Path('data/manifest.json')
data = json.loads(m.read_text())
for k in list(data.keys())[:10]:
    print(repr(k), '->', data[k].get('date_range'))
