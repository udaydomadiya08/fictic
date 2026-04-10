from moviepy.editor import ColorClip, TextClip, CompositeVideoClip, concatenate_videoclips

# --- Create dummy base video (5 sec, black background, 1080x1920) ---
base_clip = ColorClip(size=(1080, 1920), color=(0, 0, 0), duration=5)

# --- Watermark ---
watermark = TextClip(
    "Uday_editZ",
    fontsize=40,
    color="white",
    font="Arial",
    method="label"
).set_position(("center", "top")).set_opacity(0.4).set_duration(base_clip.duration)

video_with_watermark = CompositeVideoClip([base_clip, watermark])

# --- Donation screen (5 sec, transparent background) ---
donation = TextClip(
    "If you like my videos\nplease like, comment and subscribe!\n\nThank you :)\nUday_editZ",
    fontsize=62,
    color="white",
    font="Arial",
    method="label",
    size=(1080, 1920),
    transparent=True
).set_duration(5).set_position("center").on_color(size=(1080, 1920), color=(0,0,0), col_opacity=0)

# --- Concatenate base video + donation screen ---
final_clip = concatenate_videoclips([video_with_watermark, donation])

# --- Export ---
final_clip.write_videofile("test_transparent_output.mp4", codec="libx264", fps=30, audio=False)

# --- Cleanup ---
final_clip.close()
base_clip.close()