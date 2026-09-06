"""Figure-sizing constants, matching analysis/plot.py's LaTeX \\textwidth
convention so every figure here lines up at the same physical size when
placed with ``\\includegraphics{width=\\textwidth}``.

Each plotting function fixes its total width to FIG_WIDTH and derives its
height from whatever aspect ratio suits its own layout (see FIGSIZE for the
single-panel default) -- multiply/divide FIG_WIDTH and FIG_HEIGHT rather than
hardcoding new absolute sizes.
"""

TEXTWIDTH    = 6.30045   # inches -- LaTeX \textwidth
ASPECT_RATIO = 3/4
SCALE        = 1.0
FIG_WIDTH    = TEXTWIDTH * SCALE
FIG_HEIGHT   = FIG_WIDTH * ASPECT_RATIO
FIGSIZE      = (FIG_WIDTH, FIG_HEIGHT)
