from moviepy.editor import TextClip, CompositeVideoClip

# Output file
output_file = "donation_clip_9x16.mp4"

# Create a 1080x1920 TextClip
txt_clip = TextClip(
    "If you like my videos \nplease show support \nby donating at:\n\nudaydomadiya08-1@oksbi",
    fontsize=90,
    color='white',
    font='Arial',
    size=(1080, 1920),  # vertical 9:16
    method='label',
    align='center'
)

# Duration of the clip
txt_clip = txt_clip.set_duration(5)

# Add background color (black)
final_clip = txt_clip.on_color(
    color=(0, 0, 0),    # black background
    col_opacity=1.0
)

# Export
final_clip.write_videofile(
    output_file,
    fps=30,
    codec="libx264",
    audio=False,
    ffmpeg_params=["-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p"]
)
