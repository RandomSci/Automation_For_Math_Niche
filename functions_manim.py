from manim import *
import math

BRAND_WHITE = "#F5F7FA"
BRAND_GREEN = "#38D996"
BRAND_RED   = "#FF4D4D"
BRAND_GOLD  = "#FFD166"
BRAND_GRAY  = "#8A94A6"
BRAND_PANEL = "#111A24"
BRAND_BG    = "#0B111A"


def fm_glow_around(mobject, color=None, n_layers=3):
    """Wraps a mobject in semi-transparent glow layers.
    Returns a VGroup(glow_layers, original). Add the returned group to the scene."""
    if color is None:
        color = BRAND_GOLD
    layers = VGroup()
    for i in range(n_layers, 0, -1):
        c = mobject.copy()
        c.scale(1 + i * 0.055)
        c.set_opacity(max(0.18 - i * 0.04, 0.03))
        try:
            c.set_stroke(color, width=2.5 * i, opacity=max(0.25 - i * 0.05, 0.04))
        except Exception:
            pass
        layers.add(c)
    return VGroup(layers, mobject)


def fm_card(label_text, value_text, accent_color=BRAND_GOLD,
             panel_color=BRAND_PANEL, text_color=BRAND_WHITE,
             label_size=36, value_size=90, buff=0.45):
    """Auto-sized card: SurroundingRectangle wraps content so the box always
    fits exactly — no text overflow, no empty oversized box.
    Returns a VGroup. Position with .move_to() then FadeIn yourself."""
    val = Text(value_text, font_size=value_size, color=text_color, weight=BOLD)
    lbl = Text(label_text, font_size=label_size, color=accent_color)
    content = VGroup(lbl, val).arrange(DOWN, buff=0.18)
    box = SurroundingRectangle(
        content, buff=buff,
        color=accent_color,
        fill_color=panel_color, fill_opacity=0.88,
        corner_radius=0.18,
    )
    return VGroup(box, content)


def fm_two_cards(left_label, left_val, left_color,
                  right_label, right_val, right_color,
                  panel_color=BRAND_PANEL, text_color=BRAND_WHITE,
                  label_size=34, value_size=80, spacing=1.1):
    """Two side-by-side cards with distinct accent colors, centered at ORIGIN.
    Returns a VGroup. FadeIn yourself."""
    left  = fm_card(left_label,  left_val,  left_color,  panel_color, text_color, label_size, value_size)
    right = fm_card(right_label, right_val, right_color, panel_color, text_color, label_size, value_size)
    return VGroup(left, right).arrange(RIGHT, buff=spacing)


def fm_stacked_cards(items, panel_color=BRAND_PANEL, text_color=BRAND_WHITE,
                      label_size=30, value_size=68, spacing=0.24):
    """Vertical stack of bill/expense cards.
    items = list of (label_str, value_str, accent_color_hex).
    Returns a VGroup. Animate entry yourself."""
    cards = VGroup()
    for label, value, color in items:
        c = fm_card(label, value, color, panel_color, text_color, label_size, value_size, buff=0.32)
        cards.add(c)
    return cards.arrange(DOWN, buff=spacing)


def fm_animate_counter(scene, start_val, end_val, label_text,
                        accent_color=BRAND_GOLD, prefix="$", suffix="",
                        duration=3.0, position=None, value_size=130, label_size=38):
    """Counting number using ValueTracker + always_redraw (zero LaTeX).
    Handles all self.play()/self.wait(). Returns (tracker, counter_mob, label_mob)."""
    if position is None:
        position = ORIGIN
    tracker = ValueTracker(float(start_val))
    end_f   = float(end_val)
    use_decimal = isinstance(end_val, float) or abs(end_val) < 100

    def _num():
        v = tracker.get_value()
        s = f"{prefix}{v:,.1f}{suffix}" if use_decimal else f"{prefix}{int(v):,}{suffix}"
        return Text(s, font_size=value_size, color=BRAND_WHITE, weight=BOLD).move_to(position)

    counter = always_redraw(_num)
    lbl = Text(label_text, font_size=label_size, color=accent_color)
    lbl.next_to(position, DOWN, buff=0.55)
    scene.add(fm_glow_around(counter, accent_color), lbl)

    anim_t = max(min(duration * 0.78, duration - 0.25), 0.1)
    hold_t = max(duration - anim_t, 0.05)
    scene.play(tracker.animate.set_value(end_f), run_time=anim_t, rate_func=smooth)
    scene.wait(hold_t)
    return tracker, counter, lbl


def fm_animate_bar_chart(scene, values, names, colors=None,
                          duration=3.5, title_text=""):
    """Professional BarChart using Manim's built-in (real tick marks + axes).
    Auto-ranges y-axis. Handles all animation."""
    if colors is None:
        colors = [BRAND_GREEN, BRAND_GOLD, BRAND_RED, BRAND_WHITE]
    bar_colors = [colors[i % len(colors)] for i in range(len(values))]

    max_v   = max(abs(v) for v in values) if values else 1
    y_step  = max(max_v / 4, 1)
    y_top   = max_v * 1.28

    chart = BarChart(
        values=values,
        bar_names=names,
        y_range=[0, y_top, y_step],
        bar_colors=bar_colors,
        y_length=4.6,
        x_length=min(len(values) * 1.9 + 0.8, 11.5),
        bar_width=0.68,
    )
    chart.move_to(ORIGIN + UP * 0.25)

    if title_text:
        title = Text(title_text, font_size=30, color=BRAND_GRAY)
        title.next_to(chart, UP, buff=0.28)
        scene.add(title)

    val_labels = VGroup()
    for bar, v, c in zip(chart.bars, values, bar_colors):
        lbl = Text(
            f"{int(v):,}" if isinstance(v, int) else f"{v:.1f}",
            font_size=26, color=c, weight=BOLD,
        )
        lbl.next_to(bar, UP, buff=0.1)
        val_labels.add(lbl)

    grow_t = max(min(duration * 0.72, duration - 0.35), 0.1)
    hold_t = max(duration - grow_t, 0.1)
    scene.play(Create(chart), run_time=grow_t * 0.45, rate_func=smooth)
    scene.play(Write(val_labels), run_time=grow_t * 0.55, rate_func=smooth)
    scene.wait(hold_t)
    return chart, val_labels


def fm_animate_gauge(scene, value, max_val, label_text,
                      accent_color=BRAND_GREEN, duration=3.0,
                      position=None, radius=2.0):
    """Arc gauge: gray full-circle track + colored fill arc.
    ValueTracker drives the fill. Handles all animation."""
    if position is None:
        position = ORIGIN

    fill_ratio  = max(0.0, min(1.0, float(value) / float(max_val or 1)))
    start_angle = PI + PI * 0.12
    sweep_total = PI - PI * 0.24

    track = Arc(radius=radius, start_angle=start_angle, angle=sweep_total)
    track.set_stroke(color=BRAND_GRAY, width=16, opacity=0.32)
    track.move_to(position)

    tracker = ValueTracker(0.0)

    def _arc():
        frac = tracker.get_value()
        if frac < 1e-6:
            return VMobject()
        a = Arc(radius=radius, start_angle=start_angle, angle=sweep_total * frac)
        a.set_stroke(color=accent_color, width=16, opacity=1.0)
        a.move_to(position)
        return a

    fill_arc = always_redraw(_arc)
    val_str  = f"{int(value)}" if isinstance(value, int) or float(value) == int(value) else f"{value:.1f}"
    val_lbl  = Text(val_str, font_size=100, color=BRAND_WHITE, weight=BOLD)
    val_lbl.move_to(position + UP * 0.2)
    cat_lbl = Text(label_text, font_size=34, color=accent_color)
    cat_lbl.next_to(track, DOWN, buff=0.32)

    scene.add(track, fill_arc, val_lbl, cat_lbl)
    anim_t = max(min(duration * 0.72, duration - 0.25), 0.1)
    hold_t = max(duration - anim_t, 0.05)
    scene.play(tracker.animate.set_value(fill_ratio), run_time=anim_t, rate_func=smooth)
    scene.wait(hold_t)
    return tracker, val_lbl, cat_lbl


def fm_animate_donut(scene, percentage, label_text,
                      accent_color=BRAND_GREEN, duration=3.0,
                      position=None, radius=1.85, thickness=0.52):
    """Donut ring: gray Annulus track + colored Arc fill + pct hero text inside.
    Handles all animation."""
    if position is None:
        position = ORIGIN

    pct        = max(0.0, min(100.0, float(percentage)))
    fill_angle = (pct / 100.0) * TAU
    inner_r    = max(radius - thickness, 0.05)

    track = Annulus(inner_radius=inner_r, outer_radius=radius,
                     color=BRAND_GRAY, fill_opacity=0.22, stroke_width=0)
    track.move_to(position)

    tracker = ValueTracker(0.0)

    def _fill():
        angle = tracker.get_value()
        if angle < 1e-6:
            return VMobject()
        arc = Arc(
            radius=inner_r + thickness / 2,
            start_angle=PI / 2,
            angle=-angle,
            stroke_width=int(thickness * 105),
        )
        arc.set_stroke(color=accent_color, opacity=1.0)
        arc.move_to(position)
        return arc

    fill     = always_redraw(_fill)
    pct_lbl  = Text(f"{pct:.0f}%", font_size=90, color=BRAND_WHITE, weight=BOLD)
    pct_lbl.move_to(position)
    cat_lbl  = Text(label_text, font_size=34, color=accent_color)
    cat_lbl.next_to(track, DOWN, buff=0.4)

    scene.add(track, fill, pct_lbl, cat_lbl)
    anim_t = max(min(duration * 0.72, duration - 0.25), 0.1)
    hold_t = max(duration - anim_t, 0.05)
    scene.play(tracker.animate.set_value(fill_angle), run_time=anim_t, rate_func=smooth)
    scene.wait(hold_t)
    return tracker, pct_lbl, cat_lbl


def fm_animate_line_chart(scene, y_values, end_value_label,
                           accent_color=BRAND_GREEN, x_labels=None,
                           duration=3.5, title_text=""):
    """Axes-based trend line with gradient fill region under the curve.
    y_values: list of numbers (uniform x-spacing). Handles all animation."""
    n = len(y_values)
    if n < 2:
        return None, None, None

    min_y   = min(y_values)
    max_y   = max(y_values)
    y_span  = max(max_y - min_y, 1.0)
    y_pad   = y_span * 0.22
    y_lo    = max(0.0, min_y - y_pad)
    y_hi    = max_y + y_pad
    y_step  = max((y_hi - y_lo) / 4, 0.01)
    x_step  = max((n - 1) // 5, 1)

    axes = Axes(
        x_range=[0, n - 1, x_step],
        y_range=[y_lo, y_hi, y_step],
        x_length=10.5,
        y_length=5.2,
        axis_config={
            "color": BRAND_GRAY,
            "stroke_opacity": 0.45,
            "include_tip": False,
            "include_numbers": False,
        },
    )
    axes.move_to(ORIGIN + DOWN * 0.15)

    pts        = [axes.c2p(i, y_values[i]) for i in range(n)]
    line       = VMobject()
    line.set_points_as_corners(pts)
    line.set_stroke(color=accent_color, width=4.5, opacity=0.95)

    baseline_y = y_lo
    fill_pts   = pts + [axes.c2p(n - 1, baseline_y), axes.c2p(0, baseline_y)]
    fill_region = Polygon(*fill_pts, fill_opacity=0.20, stroke_width=0)
    fill_region.set_color_by_gradient(accent_color, BRAND_BG)

    end_dot = Dot(axes.c2p(n - 1, y_values[-1]), color=accent_color, radius=0.13)
    end_lbl = Text(end_value_label, font_size=38, color=accent_color, weight=BOLD)
    end_lbl.next_to(end_dot, UR, buff=0.15)

    if title_text:
        ttl = Text(title_text, font_size=30, color=BRAND_GRAY)
        ttl.next_to(axes, UP, buff=0.22)
        scene.add(ttl)

    scene.add(axes, fill_region)
    grow_t  = max(min(duration * 0.70, duration - 0.55), 0.1)
    label_t = 0.4
    hold_t  = max(duration - grow_t - label_t, 0.05)
    scene.play(Create(line), run_time=grow_t, rate_func=smooth)
    scene.play(FadeIn(end_dot), Write(end_lbl), run_time=label_t)
    scene.wait(hold_t)
    return axes, line, end_dot


def fm_animate_waterfall(scene, steps, duration=4.5):
    """Cashflow waterfall. Steps arrive sequentially via GrowFromEdge.
    steps = list of {"label": str, "value": float, "color": hex (optional)}.
    Last step is treated as the net/total bar (uses BRAND_GOLD by default).
    Handles all animation."""
    n = len(steps)
    if n < 2:
        return None, None

    bar_w   = min(1.5, 10.5 / n)
    spacing = bar_w * 1.55
    total_w = (n - 1) * spacing

    running = 0.0
    bases   = []
    for s in steps[:-1]:
        bases.append(running)
        running += s["value"]
    bases.append(0.0)

    all_tops  = [b + (v["value"] if v["value"] > 0 else 0) for b, v in zip(bases, steps)]
    all_bots  = [b + (v["value"] if v["value"] < 0 else 0) for b, v in zip(bases, steps)]
    min_base  = min(all_bots)
    max_top   = max(all_tops)
    chart_h   = 4.5
    y_scale   = chart_h / max(max_top - min_base, 1.0)
    base_y    = -chart_h / 2 - min_base * y_scale + 0.4

    baseline  = Line(
        [-total_w / 2 - 0.4, base_y - 0.45, 0],
        [ total_w / 2 + 0.4, base_y - 0.45, 0],
    ).set_stroke(color=BRAND_GRAY, opacity=0.38, width=1.5)
    scene.add(baseline)

    bars   = VGroup()
    labels = VGroup()

    for i, (step, base) in enumerate(zip(steps, bases)):
        v     = step["value"]
        x_pos = -total_w / 2 + i * spacing
        bar_h = max(abs(v) * y_scale, 0.06)

        if i == n - 1:
            c  = step.get("color", BRAND_GOLD)
            y0 = base_y - 0.45
        elif v >= 0:
            c  = step.get("color", BRAND_GREEN)
            y0 = base_y - 0.45 + base * y_scale
        else:
            c  = step.get("color", BRAND_RED)
            y0 = base_y - 0.45 + (base + v) * y_scale

        bar = Rectangle(width=bar_w, height=bar_h)
        bar.set_fill(c, opacity=0.9)
        bar.set_stroke(c, width=1.5, opacity=0.55)
        bar.move_to([x_pos, y0 + bar_h / 2, 0])
        bars.add(bar)

        prefix   = "-$" if v < 0 and i < n - 1 else "$"
        val_str  = f"{prefix}{int(abs(v)):,}" if abs(v) >= 1 else f"{prefix}{abs(v):.2f}"
        val_lbl  = Text(val_str, font_size=22, color=c, weight=BOLD)
        val_lbl.next_to(bar, UP if v >= 0 else DOWN, buff=0.08)
        cat_lbl  = Text(step.get("label", ""), font_size=18, color=BRAND_GRAY)
        cat_lbl.next_to(bar, DOWN, buff=0.5 if v >= 0 else 0.9)
        labels.add(VGroup(val_lbl, cat_lbl))

    anim_t  = max(min(duration * 0.70, duration - 0.6), 0.1)
    hold_t  = max(duration - anim_t - 0.15, 0.05)
    per_bar = anim_t / n
    for bar, lbl in zip(bars, labels):
        scene.play(GrowFromEdge(bar, DOWN), FadeIn(lbl), run_time=per_bar, rate_func=smooth)
    scene.wait(hold_t)
    return bars, labels


def fm_animate_text_reveal(scene, lines, colors=None, duration=3.0, sizes=None):
    """Sequential text fade-in for hook/chapter moments ONLY.
    lines: list of strings. colors defaults to [GOLD, WHITE, WHITE, ...].
    Handles all animation."""
    if colors is None:
        colors = [BRAND_GOLD] + [BRAND_WHITE] * (len(lines) - 1)
    if sizes is None:
        sizes  = [72] + [44] * (len(lines) - 1)

    texts = VGroup(*[
        Text(lines[i], font_size=sizes[i % len(sizes)],
              color=colors[i % len(colors)], weight=BOLD)
        for i in range(len(lines))
    ])
    texts.arrange(DOWN, buff=0.36)
    texts.move_to(ORIGIN)

    per_t  = max(min(duration / max(len(lines), 1) * 0.55, 0.85), 0.1)
    hold_t = max(duration - per_t * len(lines), 0.1)
    for t in texts:
        scene.play(FadeIn(t, shift=UP * 0.18), run_time=per_t, rate_func=smooth)
    scene.wait(hold_t)
    return texts


def fm_animate_icon_grid(scene, total, filled, label_text,
                          accent_color=BRAND_GREEN, duration=3.0,
                          cols=10, position=None, icon_radius=0.18):
    """Crowd/icon grid for population statistics.
    'filled' icons use accent_color; remainder are faint gray.
    Shows pct as hero text beside the grid. Handles all animation."""
    if position is None:
        position = ORIGIN

    filled   = max(0, min(filled, total))
    rows     = math.ceil(total / max(cols, 1))
    spacing  = icon_radius * 2.9
    grid_w   = cols * spacing
    grid_h   = rows * spacing

    icons = VGroup()
    for i in range(total):
        r_idx  = i // cols
        c_idx  = i % cols
        x      = -grid_w / 2 + c_idx * spacing + spacing / 2
        y      = grid_h / 2 - r_idx * spacing - spacing / 2
        dot    = Circle(radius=icon_radius)
        if i < filled:
            dot.set_fill(accent_color, opacity=0.92)
            dot.set_stroke(accent_color, width=1.2, opacity=0.6)
        else:
            dot.set_fill(BRAND_GRAY, opacity=0.18)
            dot.set_stroke(BRAND_GRAY, width=0.8, opacity=0.28)
        dot.move_to([x, y, 0])
        icons.add(dot)

    icons.move_to(position + LEFT * 1.6)
    pct      = (filled / total * 100) if total > 0 else 0
    pct_lbl  = Text(f"{pct:.0f}%", font_size=88, color=BRAND_WHITE, weight=BOLD)
    pct_lbl.next_to(icons, RIGHT, buff=0.65)
    cat_lbl  = Text(label_text, font_size=32, color=accent_color)
    cat_lbl.next_to(pct_lbl, DOWN, buff=0.22)

    anim_t  = max(min(duration * 0.65, duration - 0.4), 0.1)
    hold_t  = max(duration - anim_t, 0.05)
    scene.play(
        LaggedStart(*[FadeIn(ic, scale=0.5) for ic in icons], lag_ratio=0.04),
        FadeIn(pct_lbl),
        FadeIn(cat_lbl),
        run_time=anim_t,
    )
    scene.wait(hold_t)
    return icons, pct_lbl


def fm_animate_stacked_cards(scene, items, duration=4.0, spacing=0.26):
    """Bill/expense cards arrive from right one at a time.
    items = [(label, value_str, accent_color), ...]. Handles all animation."""
    cards = fm_stacked_cards(items, spacing=spacing)
    cards.move_to(ORIGIN)

    per_t  = max(min(duration / max(len(items), 1) * 0.55, 0.72), 0.1)
    hold_t = max(duration - per_t * len(items), 0.15)
    for card in cards:
        scene.play(FadeIn(card, shift=LEFT * 0.45), run_time=per_t, rate_func=smooth)
    scene.wait(hold_t)
    return cards


def fm_animate_bullet_chart(scene, actual, target, range_low, range_high,
                              label_text, accent_color=BRAND_GREEN,
                              duration=3.0, position=None, bar_length=8.0):
    """Bullet chart: gray range band + target tick + actual solid bar.
    'Are you hitting the target?' visual. Handles all animation."""
    if position is None:
        position = ORIGIN

    span   = max(range_high - range_low, 1.0)
    scale  = bar_length / span

    def _x(v):
        return -bar_length / 2 + (v - range_low) * scale

    band_w  = (range_high - range_low) * scale
    band    = Rectangle(width=band_w, height=0.55)
    band.set_fill(BRAND_GRAY, opacity=0.22)
    band.set_stroke(BRAND_GRAY, width=1.0, opacity=0.3)
    band.move_to([(_x(range_low) + _x(range_high)) / 2, 0, 0])
    band.shift(position)

    tick_x  = _x(target)
    tick    = Line([tick_x, -0.5, 0], [tick_x, 0.5, 0])
    tick.set_stroke(BRAND_WHITE, width=3.5, opacity=0.9)
    tick.shift(position)

    actual_w = max((actual - range_low) * scale, 0.05)
    tracker  = ValueTracker(0.0)

    def _bar():
        w = tracker.get_value()
        if w < 0.01:
            return VMobject()
        b = Rectangle(width=w, height=0.32)
        b.set_fill(accent_color, opacity=0.95)
        b.set_stroke(accent_color, width=1.0, opacity=0.6)
        b.move_to([_x(range_low) + w / 2, 0, 0])
        b.shift(position)
        return b

    bar = always_redraw(_bar)

    target_lbl = Text(f"Target: ${int(target):,}", font_size=26, color=BRAND_WHITE)
    target_lbl.next_to(tick, UP, buff=0.22).shift(position)
    actual_lbl = Text(f"${int(actual):,}", font_size=42, color=accent_color, weight=BOLD)
    actual_lbl.next_to(band, DOWN, buff=0.32).shift(position)
    cat_lbl    = Text(label_text, font_size=30, color=BRAND_GRAY)
    cat_lbl.next_to(actual_lbl, DOWN, buff=0.15)

    scene.add(band, tick, bar, target_lbl, actual_lbl, cat_lbl)
    anim_t = max(min(duration * 0.70, duration - 0.3), 0.1)
    hold_t = max(duration - anim_t, 0.05)
    scene.play(tracker.animate.set_value(actual_w), run_time=anim_t, rate_func=smooth)
    scene.wait(hold_t)
    return tracker, bar, actual_lbl