import shutil
import os
from moviepy.editor import VideoFileClip, CompositeVideoClip, TextClip, concatenate_videoclips

def safe_clear_pycache(start_path='.'):
    for root, dirs, files in os.walk(start_path):
        for d in dirs:
            if d == '__pycache__':
                full_path = os.path.join(root, d)
                try:
                    shutil.rmtree(full_path)
                    print(f"Deleted {full_path}")
                except PermissionError as e:
                    print(f"Permission denied to delete {full_path}: {e}")

safe_clear_pycache()


import shutil
from pathlib import Path
def delete_path(path):
    try:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.is_file():
            path.unlink()
        print(f"✅ Deleted: {path}")
    except Exception as e:
        print(f"❌ Error deleting {path}: {e}")

# Cache directories
cache_dirs = [
    Path.home() / "Library" / "Caches",
    Path("/Library/Caches"),
]

# Log directories
log_dirs = [
    Path.home() / "Library" / "Logs",
    Path("/Library/Logs")
]





dirs_to_clean = cache_dirs + log_dirs 
print("🧹 Cleaning up...")

for d in dirs_to_clean:
    if d.exists():
        for item in d.iterdir():
            delete_path(item)
    else:
        print(f"⚠️ Directory not found: {d}")

print("✅ Done cleaning up!")