from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

hiddenimports = collect_submodules("rasterio")
datas = collect_data_files("rasterio", subdir=None)
binaries = collect_dynamic_libs("rasterio")
