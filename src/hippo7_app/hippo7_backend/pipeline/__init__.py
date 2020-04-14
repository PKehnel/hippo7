from hippo7_app.hippo7_backend import get_asset_folder

geometry_folder = get_asset_folder() / "meshes"
meshes = list(geometry_folder.glob("*.ply"))
mesh_dict = {}
for mesh in meshes:
    mesh_name = str(mesh).rpartition("/")[2].rstrip(".ply")
    mesh_dict[mesh_name] = mesh

audio_folder = get_asset_folder() / "audio"
songs = list(audio_folder.glob("*.mp3"))
song_dict = {}
for song in songs:
    song_name = str(song).rpartition("/")[2].replace(".mp3", "")
    song_dict[song_name] = song
