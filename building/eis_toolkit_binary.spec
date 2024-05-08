import glob, os
from PyInstaller.utils.hooks import collect_dynamic_libs

rasterio_imports_paths = glob.glob(r'/home/niko/code/eis_toolkit_venv/lib/python3.10/site-packages/rasterio/*.py')
rasterio_imports = ['rasterio.sample']
rasterio_binaries = collect_dynamic_libs('rasterio')

for item in rasterio_imports_paths:
    current_module_filename = os.path.split(item)[-1]
    current_module_filename = 'rasterio.' + current_module_filename.replace('.py', '')
    rasterio_imports.append(current_module_filename)

rasterio_imports.append('encodings')
rasterio_imports.append('encodings.*')

a = Analysis(
    ['../eis_toolkit/__main__.py'],
    pathex=['/home/niko/code/eis_toolkit_venv/lib/python3.10/site-packages'],
    binaries=rasterio_binaries,
    datas=[],
    hiddenimports=rasterio_imports,
    hookspath=['./hooks'],
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
    debug=False,  # Consider setting to False for production
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

