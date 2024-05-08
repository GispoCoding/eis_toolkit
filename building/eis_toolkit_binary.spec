import glob, os
from PyInstaller.utils.hooks import collect_dynamic_libs


a = Analysis(
    ['../eis_toolkit/__main__.py'],
    pathex=['c:/hostedtoolcache/windows/python/3.9.13/x64/lib/site-packages'],
    binaries=[],
    datas=[],
    hiddenimports=['encodings', 'encodings.*'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    exclude_binaries=False,
    name='eis_toolkit_binary',
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

