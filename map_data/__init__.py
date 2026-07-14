from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("map_data")
except PackageNotFoundError:
    __version__ = "0.0.0"  # fallback for uninstalled development
