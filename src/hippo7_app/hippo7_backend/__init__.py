from pathlib import Path


def get_asset_folder():
    return Path(__file__).parent.parent.parent.parent / "assets"
