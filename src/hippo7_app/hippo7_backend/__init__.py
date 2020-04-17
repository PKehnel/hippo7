from pathlib import Path

required_folders = ["networks", "videos", "audio"]


def get_asset_folder():
    return Path(__file__).parent.parent.parent.parent / "assets"


def check_environment():
    path_asset_folder = get_asset_folder()
    for folder in required_folders:
        if not Path.exists(path_asset_folder / folder):
            Path(path_asset_folder / folder).mkdir(parents=True)
