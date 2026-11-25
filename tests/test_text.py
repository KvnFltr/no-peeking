from nopeeking.data.text import anonymize, extract_8k_items, html_to_text, name_variants, strip_sgml_header

HTML = b"""<html><head><title>x</title><style>p{}</style></head><body>
<div style="display:none"><ix:header>hidden xbrl 2024</ix:header></div>
<p>Revenue rose to <b>$94.9</b> billion.</p>
<table><tr><td>EPS</td><td>$</td><td>0.97</td></tr><tr><td>Margin</td><td>46.2</td><td>%</td></tr></table>
</body></html>"""


def test_html_to_text_keeps_visible_text_and_table_rows():
    text = html_to_text(HTML)
    assert "hidden xbrl" not in text
    assert "Revenue rose to $94.9 billion." in text
    assert "EPS | 0.97" in text
    assert "Margin | 46.2" in text


def test_strip_sgml_header():
    raw = "EX-99.1\n2\na8-kex991q42024.htm\nEX-99.1\n\nExhibit 99.1\nApple reports"
    assert strip_sgml_header(raw).startswith("Exhibit 99.1")


def test_extract_8k_items_cuts_cover_and_signatures():
    text = (
        "FORM 8-K\nCheck the box...\nItem 2.02 Results of Operations\nWe announced results.\n"
        "Item 9.01 Financial Statements and Exhibits\n99.1 Press release\nSIGNATURES\nBy: CFO"
    )
    body = extract_8k_items(text)
    assert body.startswith("Item 2.02")
    assert "We announced results." in body
    assert "SIGNATURES" not in body and "Press release" not in body


def test_name_variants_strip_suffixes():
    v = name_variants(["APPLE INC", "Apple Inc.", "Procter & Gamble Co /OH/"])
    assert "Apple" in v or "APPLE" in v
    assert "Procter & Gamble" in v
    assert all(len(x) >= 4 for x in v)


def test_anonymize_masks_names_tickers_dates_urls():
    text = (
        "CUPERTINO, Apple Inc. (Nasdaq: AAPL) today announced results for the quarter ended "
        "September 28, 2024. Apple's revenue grew in fiscal 2024 and Q3'24. AAPL. See www.apple.com/ir "
        "or a8-kex991q42024.htm. Contact ir@apple.com on Thursday, 10/31/2024."
    )
    out = anonymize(text, ["Apple Inc.", "APPLE INC"], ["AAPL"])
    for leaked in ("Apple", "AAPL", "2024", "September", "apple.com", "Thursday", "10/31", "a8-kex991"):
        assert leaked not in out, (leaked, out)
    assert "[COMPANY]" in out and "[TICKER]" in out and "[DATE]" in out and "[YEAR]" in out


def test_anonymize_keeps_short_tickers_in_words():
    out = anonymize("IT services grew. A strong quarter.", ["Gartner Inc"], ["IT", "A"])
    assert out == "IT services grew. A strong quarter."


def test_fix_c1_repairs_cp1252_punctuation():
    from nopeeking.data.text import fix_c1

    assert fix_c1("the \x93Board\x94 \x96 Veeva\x92s") == "the “Board” – Veeva’s"


def test_strip_boilerplate_keeps_facts():
    from nopeeking.data.text import strip_boilerplate

    legal = (
        "The information in this Item 2.02 and Exhibit 99.1 shall not be deemed filed for purposes of "
        "Section 18 of the Securities Exchange Act of 1934, nor incorporated by reference in any filing."
    )
    fls = (
        "This release contains forward-looking statements that involve risks and uncertainties; the Company "
        "undertakes no obligation to update them. Actual results may differ materially."
    )
    text = f"Revenue rose 8% to $1.2 billion.\n{legal}\nGuidance raised.\n{fls}\nShall not be deemed."
    out = strip_boilerplate(text)
    assert out.splitlines() == [
        "Revenue rose 8% to $1.2 billion.",
        "Guidance raised.",
        "Shall not be deemed.",
    ]
