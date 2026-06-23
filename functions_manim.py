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
        c.scale(1 + i * 0.022)
        c.set_opacity(max(0.07 - i * 0.018, 0.01))
        try:
            c.set_stroke(color, width=1.2 * i, opacity=max(0.10 - i * 0.025, 0.02))
        except Exception:
            pass
        layers.add(c)
    return VGroup(layers, mobject)


def fm_concept_pills(labels, colors=None, panel_color=BRAND_PANEL, text_color=None,
                      font_size=44, direction=None, spacing=0.4):
    """Row or stack of short label-only pills (no values) for a set of related
    concepts shown together -- e.g. ["Savings", "Investing", "Debt", "Fun"] or
    sequential steps like ["Track", "Calculate", "Improve"]. Each label gets
    its own outlined pill auto-sized to its text, then the whole set is
    arranged with guaranteed non-overlapping spacing and auto-scaled to fit
    the frame -- never hand-position these with move_to, this is the only
    safe way to lay out multiple concept-only labels together.
    direction: RIGHT for a horizontal row (default if <=3 labels),
    DOWN for a vertical stack (default if >3 labels). Returns a VGroup."""
    if colors is None:
        colors = [BRAND_GOLD, BRAND_GREEN, BRAND_RED, BRAND_WHITE]
    if text_color is None:
        text_color = BRAND_WHITE
    if direction is None:
        direction = RIGHT if len(labels) <= 3 else DOWN

    safe_w = config.frame_width * 0.88
    safe_h = config.frame_height * 0.78

    pill_groups = []
    for i, label in enumerate(labels):
        c = colors[i % len(colors)]
        txt = Text(label, font_size=font_size, color=text_color, weight=BOLD)
        pill = SurroundingRectangle(
            txt, buff=0.32, color=c,
            fill_color=panel_color, fill_opacity=0.92,
            corner_radius=0.16,
        )
        pill_groups.append(VGroup(pill, txt))

    import numpy as _np
    if len(pill_groups) >= 4 and _np.array_equal(direction, RIGHT):
        mid = (len(pill_groups) + 1) // 2
        row1 = VGroup(*pill_groups[:mid])
        row2 = VGroup(*pill_groups[mid:])
        row1.arrange(RIGHT, buff=spacing)
        row2.arrange(RIGHT, buff=spacing)
        pills = VGroup(row1, row2).arrange(DOWN, buff=spacing * 1.1)
    else:
        pills = VGroup(*pill_groups)
        pills.arrange(direction, buff=spacing)

    if pills.width > safe_w:
        pills.scale(safe_w / pills.width)
    if pills.height > safe_h:
        pills.scale(safe_h / pills.height)

    return pills


def fm_card(label_text, value_text, accent_color=BRAND_GOLD,
             panel_color=BRAND_PANEL, text_color=BRAND_WHITE,
             label_size=32, value_size=68, buff=0.38):
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
                  label_size=30, value_size=68, spacing=0.7):
    """Two side-by-side cards with distinct accent colors, centered at ORIGIN.
    Auto-scales down if combined width would overflow the safe frame boundary.
    Returns a VGroup. FadeIn yourself."""
    left  = fm_card(left_label,  left_val,  left_color,  panel_color, text_color, label_size, value_size)
    right = fm_card(right_label, right_val, right_color, panel_color, text_color, label_size, value_size)
    group = VGroup(left, right).arrange(RIGHT, buff=spacing)
    safe_w = config.frame_width * 0.88
    if group.width > safe_w:
        group.scale(safe_w / group.width)
    return group


def fm_card_row(items, panel_color=BRAND_PANEL, text_color=BRAND_WHITE,
                 label_size=26, value_size=44, spacing=0.45):
    """Row of THREE OR MORE small label+value cards side by side -- e.g. a
    cost timeline: [("Leak","$400",BRAND_RED), ("Water Damage","$1,500",BRAND_GOLD),
    ("Mold","$4,800",BRAND_RED), ...]. This is the horizontal generalization
    of fm_two_cards -- guaranteed non-overlapping spacing via arrange(), then
    auto-scaled to fit the frame width. NEVER hand-build a row of 3+ cards
    with individual SurroundingRectangle/Text positioned via move_to or
    manual x-coordinates -- that is exactly how adjacent cards end up
    overlapping each other, the same failure class fm_concept_pills exists
    to prevent for label-only pills. This is that same guarantee, but for
    cards that pair a label WITH a value. For exactly 2 cards, prefer
    fm_two_cards instead (larger default text). For 3+ labels with NO
    values attached, use fm_concept_pills instead, not this.
    items = [(label_str, value_str, accent_color_hex), ...]. Returns a
    VGroup. FadeIn yourself, or LaggedStart per-card like fm_concept_pills."""
    cards = VGroup()
    for entry in items:
        if isinstance(entry, dict):
            label = entry.get("label", "")
            value = entry.get("value", "")
            color = entry.get("color", BRAND_GOLD)
        else:
            label, value, color = entry
        if not isinstance(value, str):
            value = f"${abs(value):,.0f}" if isinstance(value, (int, float)) else str(value)
        c = fm_card(label, value, color, panel_color, text_color, label_size, value_size, buff=0.24)
        cards.add(c)
    cards.arrange(RIGHT, buff=spacing)
    safe_w = config.frame_width * 0.92
    if cards.width > safe_w:
        cards.scale(safe_w / cards.width)
    return cards


def fm_stacked_cards(items, panel_color=BRAND_PANEL, text_color=BRAND_WHITE,
                      label_size=30, value_size=68, spacing=0.24):
    """Vertical stack of bill/expense cards.
    items = list of (label_str, value_str, accent_color_hex).
    Returns a VGroup. Animate entry yourself."""
    cards = VGroup()
    for entry in items:
        if isinstance(entry, dict):
            label = entry.get("label", "")
            value = entry.get("value", "")
            color = entry.get("color", BRAND_GOLD)
        else:
            label, value, color = entry
        if not isinstance(value, str):
            value = f"${abs(value):,.0f}" if isinstance(value, (int, float)) else str(value)
        c = fm_card(label, value, color, panel_color, text_color, label_size, value_size, buff=0.32)
        cards.add(c)
    cards.arrange(DOWN, buff=spacing)
    safe_h = config.frame_height * 0.80
    if cards.height > safe_h:
        cards.scale(safe_h / cards.height)
    return cards


def fm_clamp_to_frame(*mobjects, margin_x=0.06, margin_y=0.06):
    """Final on-screen safety net for multi-group layouts. fm_card/fm_two_cards/
    fm_card_row/fm_stacked_cards/fm_concept_pills each only guarantee THEIR OWN
    width or height fits the frame while THEY are still centered at their own
    origin -- none of them know about sibling groups, and none of them
    re-check the frame boundary once you reposition them with .next_to(),
    .to_edge(), .shift(), or .move_to(). A group that is individually safe at
    88% of frame width can still clip the camera once it's shifted toward an
    edge to sit beside or under another group (e.g. a comparison row stacked
    above a category-pill row, or two groups flanking each other left/right).
    Call this LAST, after every top-level group for the chunk has been built
    and positioned relative to each other, right before any self.play(FadeIn...).
    Pass every top-level mobject that will be on screen together; it measures
    their COMBINED bounding box against the real frame edges and, if anything
    overflows, scales the whole set down (and re-centers it if needed) by the
    minimum amount required to bring every edge back inside the safe area --
    relative spacing between the groups is preserved, nothing is reflowed.
    No-op if everything already fits. The passed-in mobjects are transformed
    in place -- keep using your original variables for FadeIn/animation.
    CRITICAL: this must be the absolute LAST thing you do to a group before
    self.play(). Calling fm_clamp_to_frame and THEN still calling .shift(),
    .move_to(), .scale(), or VGroup(...)-combining it with something else
    undoes the guarantee -- the renderer also auto-clamps everything passed
    to self.play() as a final backstop, but don't rely on that as your only
    safety net; call this explicitly whenever more than one group shares a
    chunk.
    Example: cards = fm_two_cards(...); pills = fm_concept_pills(...)
    pills.next_to(cards, DOWN, buff=0.6)
    fm_clamp_to_frame(cards, pills)
    self.play(FadeIn(cards), FadeIn(pills))"""
    combined = VGroup(*mobjects)
    safe_w = config.frame_width * (1 - 2 * margin_x)
    safe_h = config.frame_height * (1 - 2 * margin_y)
    width_scale = safe_w / combined.width if combined.width > safe_w else 1.0
    height_scale = safe_h / combined.height if combined.height > safe_h else 1.0
    scale_factor = min(width_scale, height_scale)
    if scale_factor < 1.0:
        combined.scale(scale_factor)
    max_x = config.frame_width / 2 - margin_x * config.frame_width
    max_y = config.frame_height / 2 - margin_y * config.frame_height
    shift_x = 0.0
    shift_y = 0.0
    left = combined.get_left()[0]
    right = combined.get_right()[0]
    top = combined.get_top()[1]
    bottom = combined.get_bottom()[1]
    if left < -max_x:
        shift_x = -max_x - left
    elif right > max_x:
        shift_x = max_x - right
    if bottom < -max_y:
        shift_y = -max_y - bottom
    elif top > max_y:
        shift_y = max_y - top
    if shift_x != 0.0 or shift_y != 0.0:
        combined.shift([shift_x, shift_y, 0])
    return combined


def _fm_collect_play_targets(anim, out):
    sub_animations = getattr(anim, "animations", None)
    if sub_animations:
        for sub in sub_animations:
            _fm_collect_play_targets(sub, out)
        return
    mobj = getattr(anim, "mobject", None)
    if mobj is not None:
        out.append(mobj)


def fm_animate_counter(scene, start_val, end_val, label_text,
                        accent_color=BRAND_GOLD, prefix="$", suffix="",
                        duration=3.0, position=None, value_size=130, label_size=38):
    """Counting number using ValueTracker + always_redraw (zero LaTeX).
    Handles all self.play()/self.wait(). Returns (tracker, counter_mob, label_mob)."""
    if position is None:
        position = ORIGIN
    tracker = ValueTracker(float(start_val))
    end_f   = float(end_val)
    is_whole = float(end_val) == int(float(end_val))
    use_decimal = isinstance(end_val, float) and not is_whole

    def _num():
        v = tracker.get_value()
        if use_decimal:
            s = f"{prefix}{v:,.1f}{suffix}"
        else:
            s = f"{prefix}{int(round(v)):,}{suffix}"
        return Text(s, font_size=value_size, color=BRAND_WHITE, weight=BOLD).move_to(position)

    counter = always_redraw(_num)
    lbl = Text(label_text, font_size=label_size, color=accent_color)
    lbl.next_to(position, DOWN, buff=0.85)
    scene.add(counter, lbl)

    anim_t = max(min(duration * 0.78, duration - 0.25), 0.1)
    hold_t = max(duration - anim_t, 0.05)
    scene.play(tracker.animate.set_value(end_f), run_time=anim_t, rate_func=smooth)
    scene.wait(hold_t)
    return tracker, counter, lbl


def fm_animate_bar_chart(scene, values, names, colors=None,
                          duration=3.5, title_text=""):
    """Manual bar chart: Rectangle + Text only, zero BarChart/Tex dependency.
    Real baseline + y-axis. Auto-ranges. Handles all animation."""
    if colors is None:
        colors = [BRAND_GREEN, BRAND_GOLD, BRAND_RED, BRAND_WHITE]
    bar_colors = [colors[i % len(colors)] for i in range(len(values))]

    n       = len(values)
    max_v   = max(abs(v) for v in values) if values else 1
    chart_h = 4.2
    bar_w   = min(1.6, 9.5 / max(n, 1))
    spacing = bar_w * 1.62
    total_w = (n - 1) * spacing
    y_scale = chart_h / max(max_v * 1.28, 1.0)
    base_y  = -chart_h / 2 - 0.15

    edge_margin = bar_w / 2 + 0.3
    baseline = Line([-total_w / 2 - edge_margin, base_y, 0], [total_w / 2 + edge_margin, base_y, 0])
    baseline.set_stroke(color=BRAND_GRAY, width=2.0, opacity=0.48)

    bars       = VGroup()
    val_labels = VGroup()
    cat_labels = VGroup()

    for i, (v, name, c) in enumerate(zip(values, names, bar_colors)):
        x     = -total_w / 2 + i * spacing
        bar_h = max(abs(v) * y_scale, 0.16)
        bar   = Rectangle(width=bar_w, height=bar_h)
        bar.set_fill(c, opacity=0.92)
        bar.set_stroke(c, width=1.5, opacity=0.55)
        bar.move_to([x, base_y + bar_h / 2, 0])
        bars.add(bar)

        val_str = f"{int(v):,}" if isinstance(v, int) else f"{v:.1f}"
        val_lbl = Text(val_str, font_size=26, color=c, weight=BOLD)
        val_lbl.next_to(bar, UP, buff=0.1)
        val_labels.add(val_lbl)

        cat_lbl = Text(name, font_size=20, color=BRAND_GRAY)
        cat_lbl.next_to(bar, DOWN, buff=0.15)
        cat_labels.add(cat_lbl)

    chart_group = VGroup(baseline, bars, val_labels, cat_labels)
    bars_cx = bars.get_center()[0]
    chart_group.shift(RIGHT * (-bars_cx) + UP * 0.22)

    if title_text:
        ttl = Text(title_text, font_size=30, color=BRAND_GRAY)
        ttl.next_to(chart_group, UP, buff=0.22)
        scene.add(ttl)

    scene.add(baseline, cat_labels)
    grow_t = max(min(duration * 0.70, duration - 0.38), 0.1)
    hold_t = max(duration - grow_t, 0.05)
    scene.play(
        LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.14),
        run_time=grow_t * 0.62, rate_func=smooth,
    )
    scene.play(
        LaggedStart(*[FadeIn(l) for l in val_labels], lag_ratio=0.1),
        run_time=grow_t * 0.38, rate_func=smooth,
    )
    scene.wait(hold_t)
    return bars, val_labels


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

    track = Arc(radius=radius, start_angle=start_angle, angle=sweep_total, arc_center=position)
    track.set_stroke(color=BRAND_GRAY, width=16, opacity=0.32)

    tracker = ValueTracker(0.0)

    def _arc():
        frac = tracker.get_value()
        if frac < 1e-6:
            return VMobject()
        a = Arc(radius=radius, start_angle=start_angle, angle=sweep_total * frac, arc_center=position)
        a.set_stroke(color=accent_color, width=16, opacity=1.0)
        return a

    fill_arc = always_redraw(_arc)
    val_str  = f"{int(value)}" if isinstance(value, int) or float(value) == int(value) else f"{value:.1f}"
    val_lbl  = Text(val_str, font_size=100, color=BRAND_WHITE, weight=BOLD)
    val_lbl.move_to(position + UP * 0.72)
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
                     color=BRAND_GRAY, fill_opacity=0.40, stroke_width=0)
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
            arc_center=position,
            stroke_width=int(thickness * 105),
        )
        arc.set_stroke(color=accent_color, opacity=1.0)
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
    _dot_pt = axes.c2p(n - 1, y_values[-1])
    _dot_x  = _dot_pt[0]
    _dot_y  = _dot_pt[1]
    _safe_right = config.frame_width * 0.38
    _low_thresh = -config.frame_height * 0.20
    if _dot_y < _low_thresh:
        _lbl_dir = UP
    else:
        _lbl_dir = UR if _dot_x < _safe_right else UL
    end_lbl.next_to(end_dot, _lbl_dir, buff=0.18)
    _frame_right_edge = config.frame_width / 2 - 0.25
    _frame_left_edge  = -config.frame_width / 2 + 0.25
    if end_lbl.get_right()[0] > _frame_right_edge:
        end_lbl.shift(LEFT * (end_lbl.get_right()[0] - _frame_right_edge))
    if end_lbl.get_left()[0] < _frame_left_edge:
        end_lbl.shift(RIGHT * (_frame_left_edge - end_lbl.get_left()[0]))

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


def fm_animate_line_chart_multi(scene, series, duration=4.0, title_text=""):
    """Multiple trend lines sharing ONE Axes, for direct comparison
    (e.g. rent growth vs income growth, two income paths over time).
    series: list of dicts {"y_values": [...], "label": str, "color": hex}.
    All series must have the same length (same x-spacing). This is the
    ONLY safe way to compare two or more trends on one chart -- never
    build a second raw Axes or a manual multi-line plot by hand.
    Handles all animation."""
    if not series:
        return None, None
    n = len(series[0]["y_values"])
    if n < 2:
        return None, None

    all_vals = [v for s in series for v in s["y_values"]]
    min_y   = min(all_vals)
    max_y   = max(all_vals)
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

    lines    = []
    end_dots = []
    end_lbls = []
    for s in series:
        y_values = s["y_values"]
        color    = s.get("color", BRAND_GREEN)
        pts      = [axes.c2p(i, y_values[i]) for i in range(n)]
        line     = VMobject()
        line.set_points_as_corners(pts)
        line.set_stroke(color=color, width=4.5, opacity=0.95)
        lines.append(line)

        end_dot = Dot(axes.c2p(n - 1, y_values[-1]), color=color, radius=0.11)
        end_lbl = Text(s.get("label", ""), font_size=26, color=color, weight=BOLD)
        _dot_pt2 = axes.c2p(n - 1, y_values[-1])
        _dot_x   = _dot_pt2[0]
        _dot_y2  = _dot_pt2[1]
        _safe_right = config.frame_width * 0.38
        _low_thresh2 = -config.frame_height * 0.20
        if _dot_y2 < _low_thresh2:
            _lbl_dir = UP
        else:
            _lbl_dir = UR if _dot_x < _safe_right else UL
        end_lbl.next_to(end_dot, _lbl_dir, buff=0.12)
        _frame_right_edge = config.frame_width / 2 - 0.25
        _frame_left_edge2 = -config.frame_width / 2 + 0.25
        if end_lbl.get_right()[0] > _frame_right_edge:
            end_lbl.shift(LEFT * (end_lbl.get_right()[0] - _frame_right_edge))
        if end_lbl.get_left()[0] < _frame_left_edge2:
            end_lbl.shift(RIGHT * (_frame_left_edge2 - end_lbl.get_left()[0]))
        end_dots.append(end_dot)
        end_lbls.append(end_lbl)

    order   = sorted(range(len(series)), key=lambda i: series[i]["y_values"][-1], reverse=True)
    min_gap = 0.4
    for k in range(1, len(order)):
        prev_i = order[k - 1]
        cur_i  = order[k]
        gap = end_lbls[prev_i].get_bottom()[1] - end_lbls[cur_i].get_top()[1]
        if gap < min_gap:
            end_lbls[cur_i].shift(DOWN * (min_gap - gap))

    if title_text:
        ttl = Text(title_text, font_size=30, color=BRAND_GRAY)
        ttl.next_to(axes, UP, buff=0.22)
        scene.add(ttl)

    scene.add(axes)
    grow_t  = max(min(duration * 0.65, duration - 0.6), 0.1)
    label_t = 0.45
    hold_t  = max(duration - grow_t - label_t, 0.05)
    scene.play(*[Create(l) for l in lines], run_time=grow_t, rate_func=smooth)
    scene.play(
        *[FadeIn(d) for d in end_dots],
        *[Write(l) for l in end_lbls],
        run_time=label_t,
    )
    scene.wait(hold_t)
    return axes, lines


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
    edge_margin = bar_w / 2 + 0.3

    running = 0.0
    bases   = []
    for s in steps[:-1]:
        bases.append(running)
        running += s["value"]
    bases.append(0.0)
    steps[-1]["value"] = running

    all_tops  = [b + (v["value"] if v["value"] > 0 else 0) for b, v in zip(bases, steps)]
    all_bots  = [b + (v["value"] if v["value"] < 0 else 0) for b, v in zip(bases, steps)]
    min_base  = min(all_bots)
    max_top   = max(all_tops)
    chart_h   = 4.5
    y_scale   = chart_h / max(max_top - min_base, 1.0)
    base_y    = -chart_h / 2 - min_base * y_scale + 0.4
    axis_y    = base_y - 0.45
    cat_row_y = axis_y - 0.35

    baseline  = Line(
        [-total_w / 2 - edge_margin, axis_y, 0],
        [ total_w / 2 + edge_margin, axis_y, 0],
    ).set_stroke(color=BRAND_GRAY, opacity=0.38, width=1.5)

    bars   = VGroup()
    labels = VGroup()

    for i, (step, base) in enumerate(zip(steps, bases)):
        v     = step["value"]
        x_pos = -total_w / 2 + i * spacing
        bar_h = max(abs(v) * y_scale, 0.16)

        if i == n - 1:
            c  = step.get("color", BRAND_GOLD if v >= 0 else BRAND_RED)
            y0 = axis_y if v >= 0 else axis_y - bar_h
        elif v >= 0:
            c  = step.get("color", BRAND_GREEN)
            y0 = axis_y + base * y_scale
        else:
            c  = step.get("color", BRAND_RED)
            y0 = axis_y + (base + v) * y_scale

        bar = Rectangle(width=bar_w, height=bar_h)
        bar.set_fill(c, opacity=0.9)
        bar.set_stroke(c, width=1.5, opacity=0.55)
        bar.move_to([x_pos, y0 + bar_h / 2, 0])
        bars.add(bar)

        prefix   = "-$" if v < 0 else "$"
        val_str  = f"{prefix}{int(abs(v)):,}" if abs(v) >= 1 else f"{prefix}{abs(v):.2f}"
        val_lbl  = Text(val_str, font_size=22, color=c, weight=BOLD)
        val_lbl.next_to(bar, DOWN if (v < 0) else UP, buff=0.08)
        cat_lbl  = Text(step.get("label", ""), font_size=18, color=BRAND_GRAY)
        cat_lbl.move_to([x_pos, cat_row_y, 0])
        if v < 0 and val_lbl.get_bottom()[1] < cat_lbl.get_top()[1] + 0.05:
            cat_lbl.next_to(val_lbl, DOWN, buff=0.12)
        labels.add(VGroup(val_lbl, cat_lbl))

    all_elements = VGroup(baseline, bars, labels)
    safe_bottom = -(config.frame_height * 0.44)
    actual_bottom = all_elements.get_bottom()[1]
    if actual_bottom < safe_bottom:
        shift_up = safe_bottom - actual_bottom
        all_elements.shift(UP * shift_up)

    scene.add(baseline)
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
    pct_lbl.next_to(icons, RIGHT, buff=0.5)
    cat_lbl  = Text(label_text, font_size=28, color=accent_color)
    cat_lbl.next_to(pct_lbl, DOWN, buff=0.18)
    stat_group = VGroup(pct_lbl, cat_lbl)
    if stat_group.get_left()[0] < icons.get_right()[0] + 0.3:
        stat_group.next_to(icons, RIGHT, buff=0.45)

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
    safe_h = config.frame_height * 0.82
    if cards.height > safe_h:
        cards.scale(safe_h / cards.height)

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

def fm_animate_glow_reveal(scene, text_str, accent_color=BRAND_WHITE,
                            duration=3.0, font_size=88, subtitle=None,
                            subtitle_color=None):
    """Dramatic text reveal with expanding glow rings — the 'Corporate Paycheck'
    style. Use for chapter titles, major reveals, hook moments.
    Handles all animation. Returns (text_mob, rings_group)."""
    if subtitle_color is None:
        subtitle_color = accent_color

    text = Text(text_str, font_size=font_size, color=BRAND_WHITE, weight=BOLD)
    safe_w = config.frame_width * 0.84
    if text.width > safe_w:
        text.scale(safe_w / text.width)

    sub = None
    if subtitle:
        sub = Text(subtitle, font_size=38, color=subtitle_color)
        text_group = VGroup(text, sub).arrange(DOWN, buff=0.42)
        text_group.move_to(ORIGIN)
    else:
        text.move_to(ORIGIN)

    rings = VGroup()
    for i in range(5):
        r = Circle(radius=0.5 + i * 0.55)
        r.set_stroke(accent_color, width=max(2.5 - i * 0.4, 0.4),
                     opacity=max(0.32 - i * 0.055, 0.03))
        r.move_to(text.get_center())
        rings.add(r)

    intro_t = max(min(duration * 0.38, 1.3), 0.15)
    hold_t  = max(duration - intro_t, 0.05)

    mobs = [text, *rings]
    if subtitle:
        mobs.append(sub)
        hold_t = max(hold_t - 0.28, 0.05)

    scene.play(
        FadeIn(text, scale=0.88),
        LaggedStart(*[Create(r) for r in rings], lag_ratio=0.12),
        run_time=intro_t, rate_func=smooth,
    )
    if subtitle:
        scene.play(FadeIn(sub, shift=UP * 0.12), run_time=0.28, rate_func=smooth)
    scene.wait(hold_t)
    return text, rings


def fm_animate_timeline(scene, events, accent_color=BRAND_GOLD, duration=4.0,
                         show_index=False):
    """Horizontal timeline: dots evenly spaced on a line, labels alternating
    above/below to prevent overlapping text. Dots appear with LaggedStart.
    events = list of str. Handles all animation. Returns (dots, labels)."""
    n = len(events)
    if n < 1:
        return VGroup(), VGroup()

    line_w = min(max(n * 1.75, 4.0), 11.0)
    line   = Line([-line_w / 2 - 0.1, 0, 0], [line_w / 2 + 0.1, 0, 0])
    line.set_stroke(BRAND_GRAY, width=2.0, opacity=0.45)
    scene.add(line)

    dots   = VGroup()
    labels = VGroup()

    for i, event in enumerate(events):
        x   = -line_w / 2 + i * (line_w / max(n - 1, 1)) if n > 1 else 0
        dot = Dot([x, 0, 0], radius=0.13, color=accent_color)
        dot.set_stroke(accent_color, width=1.5, opacity=0.7)
        dots.add(dot)

        prefix = f"{i + 1}. " if show_index else ""
        lbl    = Text(f"{prefix}{event}", font_size=22, color=BRAND_WHITE)
        buff   = 0.28
        if i % 2 == 0:
            lbl.next_to(dot, UP, buff=buff)
        else:
            lbl.next_to(dot, DOWN, buff=buff)
        labels.add(lbl)

    anim_t = max(min(duration * 0.72, duration - 0.4), 0.1)
    hold_t = max(duration - anim_t, 0.05)
    scene.play(
        LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.14),
        LaggedStart(
            *[FadeIn(l, shift=(UP if i % 2 == 0 else DOWN) * 0.12)
              for i, l in enumerate(labels)],
            lag_ratio=0.14,
        ),
        run_time=anim_t, rate_func=smooth,
    )
    scene.wait(hold_t)
    return dots, labels


def fm_animate_single_value(scene, value_str, label_text,
                             accent_color=BRAND_GOLD, duration=3.0,
                             value_size=140, label_size=38,
                             sublabel=None, sublabel_color=None):
    """Single hero number with a label — for beats with one number and no
    comparison. More prominent than fm_animate_counter since the value is
    already known (no counting needed). Handles all animation.
    Returns (value_mob, label_mob)."""
    if sublabel_color is None:
        sublabel_color = BRAND_GRAY

    val_mob = Text(value_str, font_size=value_size, color=BRAND_WHITE, weight=BOLD)
    lbl_mob = Text(label_text, font_size=label_size, color=accent_color)

    group = VGroup(val_mob, lbl_mob).arrange(DOWN, buff=0.55)
    if sublabel:
        sub = Text(sublabel, font_size=28, color=sublabel_color)
        group = VGroup(val_mob, lbl_mob, sub).arrange(DOWN, buff=0.48)
    group.move_to(ORIGIN)

    intro_t = max(min(duration * 0.38, 1.1), 0.1)
    hold_t  = max(duration - intro_t, 0.05)
    scene.play(
        FadeIn(val_mob, scale=0.88),
        FadeIn(lbl_mob, shift=UP * 0.15),
        run_time=intro_t, rate_func=smooth,
    )
    if sublabel:
        scene.play(FadeIn(sub, shift=UP * 0.1), run_time=0.25)
        hold_t = max(hold_t - 0.25, 0.05)
    scene.wait(hold_t)
    return val_mob, lbl_mob

def fm_formula(scene, lines, font_size=60, color=BRAND_WHITE, duration=3.0,
               position=None):
    """Plain-Text() formula display (one line or a list of lines for a
    multi-step calculation), auto-scaled to ALWAYS fit inside the frame
    no matter how long the string is -- zero LaTeX, zero overflow risk.
    Use this instead of typing a raw Text() formula yourself: a hand-sized
    font_size on a long formula string is exactly what runs off the edges
    of the 16:9 frame. lines: a single string, or a list of strings (each
    becomes its own row, e.g. the calculation on one line and the
    simplified result on the next). Handles all animation."""
    if position is None:
        position = ORIGIN
    if isinstance(lines, str):
        lines = [lines]
    safe_w = config.frame_width * 0.86
    safe_h = config.frame_height * 0.7
    text_mobs = [Text(s, font_size=font_size, color=color, weight=BOLD) for s in lines]
    group = VGroup(*text_mobs).arrange(DOWN, buff=0.28)
    if group.width > safe_w:
        group.scale_to_fit_width(safe_w)
    if group.height > safe_h:
        group.scale_to_fit_height(safe_h)
    group.move_to(position)

    intro_t = max(min(duration * 0.4, 1.2), 0.1)
    hold_t  = max(duration - intro_t, 0.05)
    scene.play(
        LaggedStart(*[FadeIn(t, scale=0.92) for t in text_mobs], lag_ratio=0.15),
        run_time=intro_t, rate_func=smooth,
    )
    scene.wait(hold_t)
    return group

def fm_animate_comparison_bars(scene, items, duration=4.0, title_text="",
                                show_net=True):
    """Clean income-vs-expense comparison bars. No axis line through bars.
    items = list of (label_str, value_float, color_hex).
    Positive values go UP (income/gain), negative go DOWN (expense/loss).
    If show_net=True, appends a computed net bar in BRAND_GOLD automatically.
    All bars sized proportionally, properly centered, labels inside-or-above.
    Handles all animation."""
    if show_net:
        net = sum(v for _, v, _ in items)
        net_color = BRAND_GREEN if net >= 0 else BRAND_RED
        items = list(items) + [("Net", net, net_color)]

    n       = len(items)
    bar_w   = min(2.2, 9.5 / max(n, 1))
    spacing = bar_w * 1.55
    total_w = (n - 1) * spacing
    edge_margin = bar_w / 2 + 0.35

    pos_vals = [v for _, v, _ in items if v > 0]
    neg_vals = [v for _, v, _ in items if v < 0]
    max_pos  = max(pos_vals) if pos_vals else 1
    max_neg  = abs(min(neg_vals)) if neg_vals else 1
    total_h  = max_pos + max_neg
    scale    = 4.2 / max(total_h, 1.0)
    zero_y   = max_neg * scale - 2.1
    cat_row_y = zero_y - 0.32

    bars       = VGroup()
    val_labels = VGroup()
    cat_labels = VGroup()

    for i, (label, value, color) in enumerate(items):
        x      = -total_w / 2 + i * spacing
        bar_h  = max(abs(value) * scale, 0.16)
        is_neg = value < 0
        y_bot  = zero_y - bar_h if is_neg else zero_y
        bar    = Rectangle(width=bar_w, height=bar_h)
        bar.set_fill(color, opacity=0.92)
        bar.set_stroke(color, width=1.5, opacity=0.55)
        bar.move_to([x, y_bot + bar_h / 2, 0])
        bars.add(bar)

        v_str   = f"${int(abs(value)):,}" if abs(value) >= 1 else f"${abs(value):.2f}"
        if is_neg:
            v_str = f"-{v_str}"
        val_lbl = Text(v_str, font_size=28, color=color, weight=BOLD)
        val_lbl.next_to(bar, UP if not is_neg else DOWN, buff=0.1)
        val_labels.add(val_lbl)

        cat_lbl = Text(label, font_size=22, color=BRAND_GRAY)
        cat_lbl.move_to([x, cat_row_y, 0])
        if is_neg and val_lbl.get_bottom()[1] < cat_lbl.get_top()[1] + 0.05:
            cat_lbl.next_to(val_lbl, DOWN, buff=0.15)
        cat_labels.add(cat_lbl)

    baseline = Line([-total_w / 2 - edge_margin, zero_y, 0], [total_w / 2 + edge_margin, zero_y, 0])
    baseline.set_stroke(BRAND_GRAY, width=2.2, opacity=0.55)
    scene.add(baseline)

    chart = VGroup(bars, val_labels, cat_labels)

    if title_text:
        ttl = Text(title_text, font_size=30, color=BRAND_GRAY)
        ttl.next_to(VGroup(bars, val_labels), UP, buff=0.28)
        scene.add(ttl)

    scene.add(cat_labels)
    grow_t = max(min(duration * 0.70, duration - 0.4), 0.1)
    hold_t = max(duration - grow_t, 0.05)
    scene.play(
        LaggedStart(
            *[GrowFromEdge(b, DOWN if v >= 0 else UP)
              for (_, v, _), b in zip(items, bars)],
            lag_ratio=0.18,
        ),
        run_time=grow_t * 0.65, rate_func=smooth,
    )
    scene.play(
        LaggedStart(*[FadeIn(l) for l in val_labels], lag_ratio=0.12),
        run_time=grow_t * 0.35, rate_func=smooth,
    )
    scene.wait(hold_t)
    return bars, val_labels

def fm_icon(name, size=1.0, color=BRAND_GOLD):
    """Pure geometry finance icons — no SVGMobject, no image loading.
    name options: 'dollar', 'coin', 'house', 'person', 'clock',
                  'arrow_up', 'arrow_down', 'warning', 'checkmark', 'fire'.
    Returns a VGroup. Position with .move_to() then self.play(FadeIn(...))."""
    g   = VGroup()
    s   = size

    if name == "dollar":
        g.add(Text("$", font_size=int(68 * s), color=color, weight=BOLD))

    elif name == "coin":
        outer = Circle(radius=0.50 * s)
        outer.set_fill(color, opacity=0.90).set_stroke(color, width=2.5)
        inner = Circle(radius=0.28 * s)
        inner.set_fill(BRAND_PANEL, opacity=0.80).set_stroke(color, width=1.0, opacity=0.45)
        sign  = Text("$", font_size=int(24 * s), color=color, weight=BOLD)
        g.add(outer, inner, sign)

    elif name == "house":
        roof_pts = [[-0.48*s, 0, 0], [0, 0.48*s, 0], [0.48*s, 0, 0]]
        roof = Polygon(*roof_pts)
        roof.set_fill(color, opacity=0.88).set_stroke(color, width=1.5, opacity=0.6)
        body = Rectangle(width=0.65*s, height=0.42*s)
        body.set_fill(color, opacity=0.65).set_stroke(color, width=1.5, opacity=0.6)
        body.next_to(roof, DOWN, buff=0)
        g.add(roof, body)

    elif name == "person":
        head = Circle(radius=0.18 * s)
        head.set_fill(color, opacity=0.88).set_stroke(color, width=1.2)
        body = Rectangle(width=0.30*s, height=0.36*s)
        body.set_fill(color, opacity=0.72).set_stroke(color, width=1.2)
        body.next_to(head, DOWN, buff=0.04 * s)
        g.add(head, body)

    elif name == "clock":
        face = Circle(radius=0.50 * s)
        face.set_fill(BRAND_PANEL, opacity=0.85).set_stroke(color, width=2.5)
        hour   = Line([0, 0, 0], [0, 0.28*s, 0]).set_stroke(color, width=3.0)
        minute = Line([0, 0, 0], [0.22*s, 0, 0]).set_stroke(color, width=2.0)
        g.add(face, hour, minute)

    elif name == "arrow_up":
        shaft = Rectangle(width=0.14*s, height=0.38*s)
        shaft.set_fill(color, opacity=0.90).shift(DOWN * 0.10 * s)
        tip = Polygon([-0.28*s, 0, 0], [0, 0.32*s, 0], [0.28*s, 0, 0])
        tip.set_fill(color, opacity=0.90).shift(UP * 0.14 * s)
        g.add(shaft, tip)

    elif name == "arrow_down":
        shaft = Rectangle(width=0.14*s, height=0.38*s)
        shaft.set_fill(color, opacity=0.90).shift(UP * 0.10 * s)
        tip = Polygon([-0.28*s, 0, 0], [0, -0.32*s, 0], [0.28*s, 0, 0])
        tip.set_fill(color, opacity=0.90).shift(DOWN * 0.14 * s)
        g.add(shaft, tip)

    elif name == "warning":
        tri = Polygon([-0.48*s, -0.38*s, 0], [0.48*s, -0.38*s, 0], [0, 0.48*s, 0])
        tri.set_fill(color, opacity=0.88).set_stroke(color, width=1.5)
        excl = Text("!", font_size=int(34 * s), color=BRAND_PANEL, weight=BOLD)
        excl.shift(DOWN * 0.04 * s)
        g.add(tri, excl)

    elif name == "checkmark":
        pts = [[-0.38*s, 0.0, 0], [-0.08*s, -0.32*s, 0], [0.48*s, 0.38*s, 0]]
        mark = VMobject()
        mark.set_points_as_corners(pts)
        mark.set_stroke(color, width=max(3.5*s, 1.5), opacity=0.95)
        g.add(mark)

    elif name == "fire":
        outer = Circle(radius=0.34 * s).stretch(0.44 / 0.68, dim=0)
        outer.set_fill(color, opacity=0.88).set_stroke(color, width=1.0)
        inner = Circle(radius=0.225 * s).stretch(0.24 / 0.45, dim=0)
        inner.set_fill(BRAND_WHITE, opacity=0.45).shift(DOWN * 0.04 * s)
        g.add(outer, inner)

    else:
        c = Circle(radius=0.38 * s)
        c.set_fill(color, opacity=0.88).set_stroke(color, width=2.0)
        g.add(c)

    return g