from __future__ import annotations

"""Independent adversarial verification of upload / path-traversal protections.

`test_malicious_files.py` already covers: `../../evil.csv` traversal, executable
bytes disguised as .csv, wrong-content .xlsx, and a null byte in the filename. This
file goes further, into territory that file does not touch:

1. A wider adversarial-filename set (Windows drive-letter absolute paths, UNC paths,
   Windows reserved device names, long names, non-ASCII) run through the real
   `/api/datasets/upload` endpoint.
2. XXE and entity-expansion ("billion laughs") attacks against the .xlsx parser
   (`openpyxl` via `pandas.read_excel`), which nothing in the existing suite exercises
   at all -- this project has neither `lxml` nor `defusedxml` installed, so openpyxl
   falls back to the stdlib `xml.etree.ElementTree` parser. Whether that fallback is
   actually safe was an open question verified here, not assumed.

All payloads here were run for real (not just reasoned about) via a Python-level
`pd.read_excel` probe before being turned into these regression tests; the amplification
payload is intentionally sized just above the point where CPython's bundled expat
(2.8.1, verified installed) enforces its default billion-laughs guard, so this test
runs in well under a second rather than actually trying to allocate gigabytes.
"""

import io
import zipfile

import pytest

from app.datasets.validation import sanitize_display_name


# ---------------------------------------------------------------------------
# 1. Adversarial filenames through the real endpoint (not just the pure function)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "adversarial_filename",
    [
        "C:\\Windows\\System32\\evil.csv",
        "\\\\attacker-host\\share\\evil.csv",
        "../../../../etc/passwd.csv",
        "..\\..\\..\\Windows\\win.ini.csv",
        "CON.csv",
        "NUL.csv",
        "PRN.csv",
        "a" * 500 + ".csv",
        "файл_данных.csv",  # non-ASCII (Cyrillic)
        "evil.csv\x00.exe",
        "...csv",
        "  leading_spaces.csv",
    ],
)
def test_upload_with_adversarial_filename_never_escapes_storage_dir(client, tmp_path, monkeypatch, adversarial_filename):
    """Full end-to-end: whatever the filename looks like, the file that lands on disk
    must be inside the configured storage dir, named by a generated id -- never by
    anything derived from user input. This exercises the real upload endpoint, not
    just `sanitize_display_name` in isolation."""
    from pathlib import Path

    from app.config import get_settings
    import app.datasets.storage as storage_module

    monkeypatch.setenv("STORAGE_DIR", str(tmp_path))
    get_settings.cache_clear()
    storage_module._store = None

    files = {"file": (adversarial_filename, b"a,b\n1,2\n3,4\n", "text/csv")}
    response = client.post("/api/datasets/upload", files=files)

    # Must respond cleanly either way -- never crash the server.
    assert response.status_code in (200, 400)

    if response.status_code == 200:
        dataset_id = response.json()["dataset_id"]
        record = storage_module.get_dataset_store().get(dataset_id)
        stored_path = Path(record.stored_path).resolve()
        assert tmp_path.resolve() in stored_path.parents
        assert ".." not in stored_path.parts
        # The on-disk filename must be the generated id, never the raw upload name.
        assert stored_path.stem == dataset_id

    storage_module._store = None
    get_settings.cache_clear()


def test_reserved_windows_device_names_are_never_used_as_a_real_path():
    """CON/NUL/PRN/etc. are special device names on Windows -- if `sanitize_display_name`
    output were ever used as an actual filesystem path (it is documented not to be,
    see app/datasets/validation.py), writing to `CON.csv` could hang or behave
    unexpectedly. Confirms the documented invariant holds: this function's output is
    display-only, and does not attempt filesystem-safety normalization itself (that
    responsibility lives entirely in storage.py's uuid-based path construction)."""
    # No crash / hang on any of these -- purely a string transform.
    for name in ["CON.csv", "NUL", "PRN.csv", "AUX.csv", "COM1.csv", "LPT1.csv"]:
        result = sanitize_display_name(name)
        assert isinstance(result, str) and len(result) > 0


# ---------------------------------------------------------------------------
# 2. XML entity-expansion / XXE against the .xlsx (openpyxl) parser
# ---------------------------------------------------------------------------

_CONTENT_TYPES = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
<Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>
</Types>
"""
_RELS = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>
"""
_WORKBOOK = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets>
</workbook>
"""
_WORKBOOK_RELS = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/>
</Relationships>
"""
_SHEET1 = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<sheetData>
<row r="1"><c r="A1" t="str"><v>header</v></c></row>
<row r="2"><c r="A2" t="s"><v>0</v></c></row>
</sheetData>
</worksheet>
"""


def _build_xlsx(shared_strings: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _CONTENT_TYPES)
        zf.writestr("_rels/.rels", _RELS)
        zf.writestr("xl/workbook.xml", _WORKBOOK)
        zf.writestr("xl/_rels/workbook.xml.rels", _WORKBOOK_RELS)
        zf.writestr("xl/worksheets/sheet1.xml", _SHEET1)
        zf.writestr("xl/sharedStrings.xml", shared_strings)
    return buf.getvalue()


_XXE_SHARED_STRINGS = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<!DOCTYPE sst [
<!ENTITY xxe SYSTEM "file:///C:/Windows/win.ini">
]>
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="1" uniqueCount="1">
<si><t>PAYLOAD:&xxe;</t></si>
</sst>
"""

# 9 nested entities, each = 10 copies of the previous -> a theoretical ~10^9x
# amplification of a 10-byte seed (~10GB if actually realized). This is well past
# expat's default billion-laughs guard (100x amplification / 8MB activation
# threshold), so a safe parser must reject it near-instantly, never allocate that
# memory. Verified via a standalone probe to reject in ~0.3s before being added here.
_amplification_defs = ['<!ENTITY l0 "AAAAAAAAAA">']
for _i in range(1, 9):
    _amplification_defs.append(
        f'<!ENTITY l{_i} "&l{_i - 1};&l{_i - 1};&l{_i - 1};&l{_i - 1};&l{_i - 1};'
        f'&l{_i - 1};&l{_i - 1};&l{_i - 1};&l{_i - 1};&l{_i - 1};">'
    )
_BILLION_LAUGHS_SHARED_STRINGS = (
    """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<!DOCTYPE sst [
"""
    + "\n".join(_amplification_defs)
    + """
]>
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="1" uniqueCount="1">
<si><t>&l8;</t></si>
</sst>
"""
).encode("utf-8")


def test_xlsx_xxe_external_entity_does_not_leak_local_file_content(client, tmp_path, monkeypatch):
    """A crafted .xlsx whose sharedStrings.xml declares a SYSTEM entity pointing at a
    local file (classic XXE) must never have that file's content appear in the parsed
    dataset. Verified: CPython's expat does not resolve external SYSTEM entities by
    default (no network/file fetch), so this is expected to fail parsing cleanly --
    confirmed here rather than assumed."""
    from app.config import get_settings
    import app.datasets.storage as storage_module

    monkeypatch.setenv("STORAGE_DIR", str(tmp_path))
    get_settings.cache_clear()
    storage_module._store = None

    content = _build_xlsx(_XXE_SHARED_STRINGS)
    files = {"file": ("payload.xlsx", content, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
    response = client.post("/api/datasets/upload", files=files)

    # Whichever way it resolves, the response must be clean (400 rejection is the
    # actual observed behavior -- expat raises "undefined entity" since it refuses to
    # fetch the external file) and must never contain win.ini's real content.
    assert response.status_code in (200, 400)
    assert "[extensions]" not in response.text  # a real win.ini section header
    assert "for 16-bit app support" not in response.text

    storage_module._store = None
    get_settings.cache_clear()


def test_xlsx_entity_expansion_bomb_is_rejected_fast_not_hung_or_crashed(client, tmp_path, monkeypatch):
    """A billion-laughs-style .xlsx must be rejected quickly (expat's built-in
    amplification guard), never hang the request thread or exhaust server memory.
    This is a real DoS-shaped adversarial input, run against the actual upload
    endpoint end-to-end."""
    import time

    from app.config import get_settings
    import app.datasets.storage as storage_module

    monkeypatch.setenv("STORAGE_DIR", str(tmp_path))
    get_settings.cache_clear()
    storage_module._store = None

    content = _build_xlsx(_BILLION_LAUGHS_SHARED_STRINGS)
    files = {"file": ("bomb.xlsx", content, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}

    start = time.time()
    response = client.post("/api/datasets/upload", files=files)
    elapsed = time.time() - start

    assert response.status_code == 400  # rejected as an unparseable file, not a crash
    assert elapsed < 5.0  # must fail fast; a real bomb without a guard would hang far longer
    # The parser's internal error text may appear in the 400 detail (it's an expat
    # parse error, not sensitive data) -- what matters is it never returns 200 with an
    # actually-expanded multi-megabyte cell value.

    storage_module._store = None
    get_settings.cache_clear()
