"""Branch coverage for rag.html_parse."""

from rag.html_parse import HTMLText
from rag.parse import parse_html


class _AlwaysIn(list):
    def __contains__(self, item):
        return True


def test_html_stack_covers_tables_skips_pre_and_empty_flushes():
    html = """
    <script><div>hidden</div></style></script>
    <style>css</style>
    <pre></pre>
    <pre>code block</pre>
    <h1></h1>
    <h2>Title</h2>
    <p></p>
    <p>Paragraph</p>
    <div>Loose</div>
    <ul><li>item</li></ul>
    <img alt="">
    <img alt="Caption">
    <img ALT="Upper">
    <table>
      stray
      <tr><td><li>in table</li></td><th>H</th></tr>
      <tr></tr>
      <tr><td></td></tr>
      <tr><th>A</th><td>1</td></tr>
    </table>
    <table></table>
    <table><tr><td>hi</tr></td></table>
    <table><tr><td>x</td></table></tr>
    <b><i>nested</b></i>
    <div></span>
    """
    blocks = parse_html(html, "page.html")
    kinds = {block.kind for block in blocks}
    assert "code" in kinds
    assert "table" in kinds
    assert "list" in kinds
    assert "caption" in kinds
    assert any(block.heading_path == "Title" for block in blocks)
    assert any("Paragraph" in block.text for block in blocks)

    empty = HTMLText()
    empty.feed("")
    empty.close()
    assert empty.blocks == []

    skipped = HTMLText()
    skipped.skip = 1
    skipped.handle_starttag("div", [])
    skipped.handle_endtag("span")
    assert skipped.stack == []
    assert skipped.skip == 1

    mismatched = HTMLText()
    mismatched.stack = _AlwaysIn()
    mismatched.handle_endtag("span")
    assert mismatched.stack == []
