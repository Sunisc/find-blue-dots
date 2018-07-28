from bottle import route, request, static_file, run
import Args
import pandas as pd
import numpy as np
from PIL import Image
import base64
from io import BytesIO

args = Args.Parse(
    blobs=str,
    outblobs=str
)

data = pd.read_pickle(args.blobs)
data['colFromIndex'] = data.index
data = data.sort_values(['isdot', 'colFromIndex'])


perRow = 10
NROWS = 100


def toImage(img, title):
    tpl = '<image title="{title}" src="data:image/png;base64,{data}" />'
    pil_img = Image.fromarray(img)
    buff = BytesIO()
    pil_img.save(buff, format="png")
    data = base64.b64encode(buff.getvalue()).decode("utf-8")
    t = tpl.format(data=data, title=title)
    return t


def makeCell(row):
    img = toImage(row.pixels)
    return '<td>{}</td>'.format(img)


page = '''
<html>
  <head>
    <title>
      Edit Labels
    </title>
    <link rel="stylesheet" href="/static/EditLabels.css" />
  </head>
  <body>
    <form method="post" action="/">
      {table}
      <input type="hidden" name="start" value="{start}" />
      <button type="submit" name="back" value="1">Back</button>
      <button type="submit" name="next" value="1">Next</button>
      <button type="submit" name="save" value="1">Save</button>
    </form>
  </body>
</html>
'''


def drawForm(start):
    end = min(start + NROWS, len(data))
    nrows = end - start

    trs = []
    for i, rows in data.iloc[start:end].groupby(np.arange(nrows) // perRow):
        tr = []
        for row in rows.itertuples():
            img = toImage(row.pixels, row.Index)
            check = '''<label>
                <input type="checkbox" name="D{key}" {check} />
                {image}
            </label>'''.format(image=img, key=row.Index,
                               check='checked' if row.isdot else '')
            td = '<td>{}</td>'.format(check)
            tr.append(td)
        trs.append('<tr>{}</tr>'.format('\n'.join(tr)))
    table = '<table>{}</table>'.format('\n'.join(trs))
    return page.format(table=table, start=start)


@route('/')
def editor():
    start = int(request.query.get('start', '0'))
    return drawForm(start)


@route('/', method='post')
def update():
    start = int(request.forms.get('start'))
    end = min(start + NROWS, len(data))

    backPressed = request.forms.get('back')
    nextPressed = request.forms.get('next')
    savePressed = request.forms.get('save')
    rows = data.iloc[start:end]
    for row in rows.itertuples():
        name = 'D{key}'.format(key=row.Index)
        data.at[row.Index, 'isdot'] = 1 if request.forms.get(name) else 0

    if backPressed:
        start = max(0, start - NROWS)
    elif nextPressed:
        start = min(len(data), start + NROWS)
    elif savePressed:
        data.to_pickle(args.outblobs)

    return drawForm(start)


@route('/static/<filename:path>')
def send_static(filename):
    return static_file(filename, root='.')


if __name__ == '__main__':
    run(host='localhost', port=8080, debug=True, reloader=True)
