from src.provenance_tracker import classify_licence


def test_open_licence_url():
    assert classify_licence("https://creativecommons.org/licenses/by/4.0/") == "OPEN"

def test_open_licence_text():
    assert classify_licence("Creative Commons Attribution license (reuse allowed)") == "OPEN"

def test_open_licence_reuse():
    assert classify_licence("reuse allowed") == "OPEN"

def test_restricted_licence():
    assert classify_licence("https://creativecommons.org/licenses/by-nc/4.0/") == "RESTRICTED"

def test_proprietary_licence():
    assert classify_licence("youtube.com/t/terms") == "PROPRIETARY"

def test_unknown_licence_empty():
    assert classify_licence("") == "UNKNOWN"

def test_unknown_licence_none():
    assert classify_licence(None) == "UNKNOWN"

def test_cc0():
    assert classify_licence("CC0") == "OPEN"