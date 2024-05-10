# -*- mode: python ; coding: utf-8 -*-

import glob, os
rasterio_imports_paths = glob.glob(r'/home/niko/code/eis_toolkit_venv/lib/python3.10/site-packages/rasterio/*.py')
rasterio_imports = ['rasterio.sample']

for item in rasterio_imports_paths:
    current_module_filename = os.path.split(item)[-1]
    current_module_filename = 'rasterio.'+current_module_filename.replace('.py', '')
    rasterio_imports.append(current_module_filename)

a = Analysis(
    ['../eis_toolkit/__main__.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=rasterio_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='eis_toolkit',
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
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='eis_toolkit',
)
