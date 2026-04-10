# safe_replace.py
from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx
import subprocess
import os

# ========================================================
# SAFE REENCODE
# ========================================================
def reencode_clip_safe(path, temp_folder="temp_reencoded"):
    """Re-encode clip to avoid ffmpeg/MoviePy decode errors."""
    if path is None:
        return None

    if not os.path.isfile(path):
        print(f"❌ File does not exist: {path}")
        return None

    os.makedirs(temp_folder, exist_ok=True)
    base_name = os.path.basename(path)
    out_path = os.path.join(temp_folder, base_name)

    cmd = [
        "ffmpeg", "-y",
        "-i", path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "128k",
        out_path
    ]

    try:
        subprocess.run(cmd, check=True)
        return out_path
    except subprocess.CalledProcessError:
        print(f"⚠ Failed to re-encode: {path}")
        return None


# ========================================================
# SAFE REVERSE CLIP
# ========================================================
import uuid
from moviepy.editor import VideoFileClip, vfx

def replace_with_reverse_safe(c1, c2, min_duration=1.0, temp_folder="temp_reencoded"):
    """
    Robust reverse transition:
      - Ensures c1 and c2 are readable (re-encodes if needed)
      - Performs reverse on a fresh reopened file (so MoviePy readers don't get closed)
      - Matches duration to c2
      - Returns a VideoClip that is safe for composition/export

    On irrecoverable failures it returns a safe fallback (usually c2).
    """

    def is_readable_clip(clip):
        try:
            clip.get_frame(0)
            return True
        except Exception:
            return False

    def reload_path_or_none(path):
        """Reencode (if necessary) and return a valid path we can open, or None."""
        # If input path is already in temp_folder and works, return it
        try:
            # try open directly first
            test = VideoFileClip(path)
            test.get_frame(0)
            test.close()
            return path
        except Exception:
            pass

        # try re-encoding
        new = reencode_clip_safe(path, temp_folder=temp_folder)
        if not new:
            return None

        # verify new file is readable
        try:
            test2 = VideoFileClip(new)
            test2.get_frame(0)
            test2.close()
            return new
        except Exception:
            return None

    def open_safe(path):
        """Open and return a fresh VideoFileClip or None."""
        try:
            clip = VideoFileClip(path)
            clip.get_frame(0)
            return clip
        except Exception:
            return None

    # -------------------------
    # 1) Ensure we have file paths to reopen (prefer reopening from .filename)
    # -------------------------
    # If c1/c2 are VideoFileClip objects with .filename attribute, use those paths.
    c1_path = getattr(c1, "filename", None)
    c2_path = getattr(c2, "filename", None)

    # If path missing, try to create a temporary re-encode from the clip object
    if not c1_path:
        # write a temp re-encoded file from the in-memory clip
        tmp_name = os.path.join(temp_folder, f"tmp_c1_{uuid.uuid4().hex}.mp4")
        os.makedirs(temp_folder, exist_ok=True)
        try:
            c1.write_videofile(tmp_name, codec="libx264", audio_codec="aac", verbose=False, logger=None)
            c1_path = tmp_name
        except Exception:
            c1_path = None

    if not c2_path:
        tmp_name = os.path.join(temp_folder, f"tmp_c2_{uuid.uuid4().hex}.mp4")
        os.makedirs(temp_folder, exist_ok=True)
        try:
            c2.write_videofile(tmp_name, codec="libx264", audio_codec="aac", verbose=False, logger=None)
            c2_path = tmp_name
        except Exception:
            c2_path = None

    # If still no path, attempt to read frames directly from provided clip objects for minimal checks
    if c1_path is None and not is_readable_clip(c1):
        print("⚠ c1 not readable and has no filename → returning c2")
        return c2

    if c2_path is None and not is_readable_clip(c2):
        print("⚠ c2 not readable and has no filename → returning c1")
        return c1

    # -------------------------
    # 2) Re-open fresh validated files
    # -------------------------
    # Prefer validated filesystem paths for robust reopening
    if c1_path:
        c1_valid_path = reload_path_or_none(c1_path)
        if not c1_valid_path:
            print(f"❌ Unable to obtain readable version of c1 ({c1_path}) → returning c2")
            return c2
        c1_safe = open_safe(c1_valid_path)
        if c1_safe is None:
            print(f"❌ Failed to open reencoded c1 ({c1_valid_path}) → returning c2")
            return c2
    else:
        # c1 had no path but was readable as clip object
        c1_safe = c1

    if c2_path:
        c2_valid_path = reload_path_or_none(c2_path)
        if not c2_valid_path:
            print(f"❌ Unable to obtain readable version of c2 ({c2_path}) → returning c1")
            # Close opened c1_safe if we opened it
            try:
                if getattr(c1_safe, "close", None):
                    c1_safe.close()
            except:
                pass
            return c1
        c2_safe = open_safe(c2_valid_path)
        if c2_safe is None:
            print(f"❌ Failed to open reencoded c2 ({c2_valid_path}) → returning c1")
            try:
                if getattr(c1_safe, "close", None):
                    c1_safe.close()
            except:
                pass
            return c1
    else:
        c2_safe = c2

    # -------------------------
    # 3) Safety: duration checks
    # -------------------------
    if getattr(c1_safe, "duration", 0) is None or c1_safe.duration < min_duration:
        print("⚠ c1 duration too small → skipping reverse and returning c2")
        try:
            c1_safe.close()
        except:
            pass
        return c2_safe

    if getattr(c2_safe, "duration", 0) is None or c2_safe.duration < min_duration:
        print("⚠ c2 duration too small → returning c1")
        try:
            c1_safe.close()
        except:
            pass
        return c1_safe

    # -------------------------
    # 4) Attempt reverse on a fresh clip (never the original)
    # -------------------------
    try:
        rev = c1_safe.fx(vfx.time_mirror)
    except Exception as e:
        print("⚠ reverse fx on c1 failed:", e)
        # try re-encode c1 path and reopen
        if c1_path:
            re_path = reload_path_or_none(c1_path)
            if not re_path:
                print("⛔ reencode retry failed → fallback to c2")
                try:
                    c1_safe.close()
                    c2_safe.close()
                except:
                    pass
                return c2_safe
            try:
                reopened = open_safe(re_path)
                rev = reopened.fx(vfx.time_mirror)
            except Exception:
                print("⛔ reverse still failing → fallback to c2")
                try:
                    c1_safe.close()
                    c2_safe.close()
                except:
                    pass
                return c2_safe

    # -------------------------
    # 5) Match durations (use speedx or subclip)
    # -------------------------
    try:
        if c2_safe.duration > rev.duration and rev.duration > 0:
            factor = rev.duration / c2_safe.duration
            # speedx expects factor >0; if factor ~0 fallback
            if factor <= 0:
                pass
            else:
                rev = rev.fx(vfx.speedx, factor=factor)
        else:
            rev = rev.subclip(0, min(rev.duration, c2_safe.duration))
    except Exception as e:
        print("⚠ duration matching failed:", e)
        try:
            c1_safe.close()
            c2_safe.close()
        except:
            pass
        return c2_safe

    # -------------------------
    # 6) Final read test
    # -------------------------
    try:
        rev.get_frame(0)
    except Exception as e:
        print("❌ final reversed clip unreadable:", e)
        try:
            c1_safe.close()
            c2_safe.close()
        except:
            pass
        return c2_safe

    # Close the helper opened clips (their readers are independent from rev)
    try:
        if getattr(c1_safe, "close", None):
            c1_safe.close()
        if getattr(c2_safe, "close", None):
            c2_safe.close()
    except:
        pass

    # Ensure returned rev has the same duration as c2_safe (safety)
    return rev.set_duration(c2_safe.duration)


# ========================================================
# MAIN TEST
# ========================================================
def test_forward_backward(clips_paths, output_path="forward_backward_test.mp4", min_duration=1.0):
    clips = []

    for p in clips_paths:
        safe_path = reencode_clip_safe(p)
        if not safe_path:
            print(f"⚠ Skipping unreadable file: {p}")
            continue

        try:
            clip = VideoFileClip(safe_path)
            clip.get_frame(0)  # validate
            print(f"Loaded OK: {p} (duration {clip.duration:.2f}s)")
            clips.append(clip)
        except Exception:
            print(f"❌ Still unreadable after re-encode: {p}")
            continue

    if len(clips) < 2:
        print("❌ Not enough valid clips.")
        return

    final_clips = []

    i = 0
    while i < len(clips) - 1:
        c1 = clips[i]
        c2 = clips[i + 1]

        new_c2 = replace_with_reverse_safe(c1, c2, min_duration)
        final_clips.append(c1)
        final_clips.append(new_c2)

        i += 2

    if len(clips) % 2 == 1:
        final_clips.append(clips[-1])

    final_video = concatenate_videoclips(final_clips, method="compose")
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")



# ========================================================
# RUN
# ========================================================
if __name__ == "__main__":
    clip_files = [
        "/Users/uday/Downloads/fictic/firstclip/scene_01_94320_96320.mp4",
        "/Users/uday/Downloads/fictic/firstclip/scene_02_156280_157880.mp4",
        "firstclip/scene_05_224480_228680.mp4",  # this is the broken file
        "/Users/uday/Downloads/fictic/firstclip/scene_04_194160_195960.mp4",
        "/Users/uday/Downloads/fictic/firstclip/scene_13_166080_168440.mp4",
        "/Users/uday/Downloads/fictic/firstclip/scene_19_214800_217120.mp4",
        "/Users/uday/Downloads/fictic/firstclip/scene_04_194160_195960.mp4"
    ]

    test_forward_backward(clip_files, output_path="forward_backward_test.mp4", min_duration=1.0)
