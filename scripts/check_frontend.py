"""Extract inline JavaScript from the HTML and ask Node to syntax-check it."""
from html.parser import HTMLParser
from pathlib import Path
import subprocess
import tempfile


class InlineScriptParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_script = False
        self.current = []
        self.scripts = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "script" and not dict(attrs).get("src"):
            self.in_script = True
            self.current = []

    def handle_data(self, data):
        if self.in_script:
            self.current.append(data)

    def handle_endtag(self, tag):
        if tag.lower() == "script" and self.in_script:
            self.scripts.append("".join(self.current))
            self.in_script = False


html_path = Path(__file__).resolve().parents[1] / "frontend" / "index.html"
parser = InlineScriptParser()
parser.feed(html_path.read_text(encoding="utf-8"))
if not parser.scripts:
    raise SystemExit("No inline JavaScript found in frontend/index.html")

with tempfile.NamedTemporaryFile(mode="w", suffix=".js", encoding="utf-8") as temp:
    temp.write("\n".join(parser.scripts))
    temp.flush()
    subprocess.run(["node", "--check", temp.name], check=True)
print("Frontend inline JavaScript syntax OK.")
