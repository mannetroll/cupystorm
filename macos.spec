# macos.spec
# Build:
#   rm -rf build dist
#   uv pip install pyinstaller
#   uv run pyinstaller macos.spec
#   ./dist/cupystorm.app/Contents/MacOS/cupystorm
#   open -n ./dist/cupystorm.app
#

a = Analysis(
    ["cupystorm/turbo_main.py"],
    pathex=["."],
    binaries=[],
    datas=[],
    hiddenimports=[],
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="cupystorm",
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="cupystorm",
)

app = BUNDLE(
    coll,
    name="cupystorm.app",
    icon="cupystorm/cupystorm.icns",
    bundle_identifier="se.mannetroll.cupystorm",
)