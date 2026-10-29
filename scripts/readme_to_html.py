"""Minimal Markdown -> HTML renderer (GitHub-like look) for the report screenshots.

python scripts/readme_to_html.py README.md out.html ["## Heading to start from"]
"""

import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _data_uri(rel: str) -> str:
    import base64

    p = ROOT / rel
    mime = "image/svg+xml" if p.suffix == ".svg" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()


def inline(s: str) -> str:
    s = html.escape(s, quote=False)
    s = re.sub(
        r"!\[([^\]]*)\]\(([^)]+)\)",
        lambda m: f'<img alt="{m.group(1)}" src="{_data_uri(m.group(2))}">',
        s,
    )
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![*\w])\*([^*]+)\*(?!\w)", r"<em>\1</em>", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    return s


def convert(md: str) -> str:
    out: list[str] = []
    lines, i = md.splitlines(), 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("<!--"):
            i += 1
            continue
        if ln.startswith("```"):
            j = i + 1
            while not lines[j].startswith("```"):
                j += 1
            out.append("<pre><code>" + html.escape("\n".join(lines[i + 1 : j])) + "</code></pre>")
            i = j + 1
            continue
        m = re.match(r"(#{1,6}) (.*)", ln)
        if m:
            n = len(m.group(1))
            out.append(f"<h{n}>{inline(m.group(2))}</h{n}>")
            i += 1
            continue
        if ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip("|").split("|")])
                i += 1
            head, body = rows[0], [r for r in rows[2:]]
            t = (
                "<table><thead><tr>"
                + "".join(f"<th>{inline(c)}</th>" for c in head)
                + "</tr></thead><tbody>"
            )
            t += "".join(
                "<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body
            )
            out.append(t + "</tbody></table>")
            continue
        if re.match(r"\s*([*-]|\d+\.) ", ln):
            tag = "ol" if re.match(r"\s*\d+\.", ln) else "ul"
            items = []
            while i < len(lines) and lines[i].strip():
                if re.match(r"\s*([*-]|\d+\.) ", lines[i]):
                    items.append(re.sub(r"^\s*([*-]|\d+\.) ", "", lines[i]))
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            out.append(f"<{tag}>" + "".join(f"<li>{inline(x)}</li>" for x in items) + f"</{tag}>")
            continue
        if ln.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].startswith(">"):
                buf.append(lines[i].lstrip("> "))
                i += 1
            out.append("<blockquote>" + inline(" ".join(buf)) + "</blockquote>")
            continue
        if ln.strip():
            buf = []
            while (
                i < len(lines)
                and lines[i].strip()
                and not re.match(r"(#|\||```|>|\s*[*-] )", lines[i])
            ):
                buf.append(lines[i])
                i += 1
            out.append("<p>" + inline(" ".join(buf)) + "</p>")
            continue
        i += 1
    return "\n".join(out)


CSS = """
body{background:#fff;color:#1f2328;
font:15px/1.55 -apple-system,'Segoe UI',Helvetica,Arial,sans-serif;
max-width:980px;margin:24px auto;padding:0 32px}
h1,h2{border-bottom:1px solid #d1d9e0;padding-bottom:.3em} h4{margin-bottom:.4em}
table{border-collapse:collapse;margin:8px 0 16px;display:block;overflow:auto;font-size:13.5px}
th,td{border:1px solid #d1d9e0;padding:5px 11px} tr:nth-child(2n){background:#f6f8fa}
code{background:#eff1f3;padding:.15em .35em;border-radius:5px;font-size:85%}
pre{background:#f6f8fa;padding:14px;border-radius:6px;overflow:auto} pre code{background:none}
img{max-width:100%} blockquote{color:#59636e;border-left:4px solid #d1d9e0;margin:0;padding:0 14px}
a{color:#0969da;text-decoration:none}
"""

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
md = src.read_text(encoding="utf-8")
if len(sys.argv) > 3:
    md = md[md.index(sys.argv[3]) :]
body = convert(md)
dst.write_text(
    f"<!doctype html><html><head><meta charset='utf-8'><title>README</title><style>{CSS}</style>"
    f"</head><body>{body}</body></html>",
    encoding="utf-8",
)
