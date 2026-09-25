"""
theme.py

Look and feel for the Gradio app: a lupine-based Gradio theme (light mode
only), the page CSS (src/ui/app.css), and the <head> tags that load the
display font.

    demo.launch(theme=THEME, css=CSS, head=HEAD)
"""

from pathlib import Path

import gradio as gr


LUPINE = gr.themes.Color(
    c50="#F6F3FB", c100="#ECE6F7", c200="#D9CDEF", c300="#BFABE3",
    c400="#9F86D2", c500="#8468BF", c600="#6C51A8", c700="#58418C",
    c800="#45346D", c900="#32264F", c950="#211934",
    name="lupine",
)

# Greys with a slight lupine bias, so neutrals match the brand instead of
# reading as default grey.
LUPINE_NEUTRAL = gr.themes.Color(
    c50="#F8F7FA", c100="#EFEDF3", c200="#E1DEE8", c300="#C9C4D3",
    c400="#9D97A9", c500="#736D80", c600="#5A5466", c700="#433E4D",
    c800="#2E2A35", c900="#1F1C24", c950="#131117",
    name="lupine_neutral",
)

THEME = gr.themes.Base(
    primary_hue=LUPINE,
    secondary_hue=LUPINE,
    neutral_hue=LUPINE_NEUTRAL,
    font=[gr.themes.GoogleFont("DM Sans"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    radius_size=gr.themes.sizes.radius_md,
)

# Light mode only: every *_dark variable takes its light value, so the page
# looks the same when Gradio switches to dark (e.g. the OS prefers dark).
THEME.set(**{name: getattr(THEME, name.removesuffix("_dark"))
             for name in vars(THEME) if name.endswith("_dark")})

CSS = (Path(__file__).parent / "app.css").read_text()

# Display face for headings; body fonts come from THEME.
HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Urbanist:wght@500;700;800&display=swap">
"""
