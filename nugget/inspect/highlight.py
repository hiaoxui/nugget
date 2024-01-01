import html
from typing import *

from nugget.utils.types import NuggetInspect


html_head = '''<!DOCTYPE html>
<html>
<head>
<style>
.highlight {background-color: red;}
.highlight_red {background-color: red;}
.highlight_green {background-color: green;}
</style>
</head>
<body>
BODY_PLACEHOLDER
</body>
</html>'''


def highlight_sent(indices, tokens):
    ret = list()
    colors = ['red', 'green']
    color_idx = 0
    for i, tok in enumerate(tokens):
        if tok == '<pad>':
            continue
        tok = html.escape(tok)
        space = ' ' if tok.startswith(' ') else ''
        tok = tok.replace(' ', '')
        if i in indices:
            tok = f'<span class="highlight_{colors[color_idx]}">{tok}</span>'
            color_idx = (color_idx + 1) % len(colors)
        tok = space + tok
        ret.append(tok)
    return ''.join(ret)


def gen_highlight(nuggets: List[NuggetInspect], save_path: str):
    highlighted = [highlight_sent(set(nug.index), nug.tokens) for nug in nuggets]
    with open(save_path, 'w') as fp:
        fp.write(html_head.replace('BODY_PLACEHOLDER', '<br><hr>'.join(highlighted)))


