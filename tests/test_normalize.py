from rag.normalize import drop_page_chrome, normalize_text


def test_nfkc_and_soft_hyphen():
    assert normalize_text("Ａ\u00adB\x00") == "AB"


def test_joins_line_break_hyphenation():
    assert normalize_text("inter-\nnational") == "international"
    assert normalize_text("inter-\n national") == "international"


def test_keeps_a_real_hyphen_and_the_sku():
    assert normalize_text("inter-\nNational") == "inter-\nNational"
    assert normalize_text("SKU-7842-XL") == "SKU-7842-XL"
    assert normalize_text("Hello") == "Hello"


def test_collapses_spaces_and_keeps_paragraphs():
    assert normalize_text("a   b\n\n\n\nc") == "a b\n\nc"


def test_code_is_not_dehyphenated():
    assert normalize_text("foo-\nbar", code=True) == "foo-\nbar"


def test_drops_repeated_short_header():
    pages = [
        "Secret Manual\nDose is 5 mg.",
        "Secret Manual\nOpen the valve.",
        "Secret Manual\nClose the latch.",
    ]
    cleaned = drop_page_chrome(pages)
    assert all("Secret Manual" not in page for page in cleaned)
    assert "5 mg" in cleaned[0]
    assert "valve" in cleaned[1]


def test_one_page_keeps_its_header():
    assert drop_page_chrome(["Secret Manual\nDose is 5 mg."])[0].startswith("Secret Manual")
