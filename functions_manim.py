from manim import *
import math
import numpy as _fmnp


def _fm_smooth_tangents(points):
    pts = [_fmnp.array(p, dtype=float) for p in points]
    n = len(pts)
    tangents = []
    for i in range(n):
        if i == 0:
            t = pts[1] - pts[0]
        elif i == n - 1:
            t = pts[-1] - pts[-2]
        else:
            t = (pts[i + 1] - pts[i - 1]) * 0.5
        tangents.append(t)
    for i in range(n):
        tx = tangents[i][0]
        if tx <= 1e-9:
            continue
        if i < n - 1:
            seg_dx = pts[i + 1][0] - pts[i][0]
            if seg_dx > 0 and tx / 3.0 > seg_dx:
                tangents[i] = tangents[i] * (3.0 * seg_dx / tx)
        tx = tangents[i][0]
        if tx <= 1e-9:
            continue
        if i > 0:
            seg_dx = pts[i][0] - pts[i - 1][0]
            if seg_dx > 0 and tx / 3.0 > seg_dx:
                tangents[i] = tangents[i] * (3.0 * seg_dx / tx)
    return pts, tangents


def _fm_set_line_smooth(vmobject, points):
    pts, tangents = _fm_smooth_tangents(points)
    n = len(pts)
    if n < 2:
        return vmobject
    vmobject.start_new_path(pts[0])
    for i in range(n - 1):
        p0 = pts[i]
        p1 = pts[i + 1]
        h1 = p0 + tangents[i] / 3.0
        h2 = p1 - tangents[i + 1] / 3.0
        vmobject.add_cubic_bezier_curve_to(h1, h2, p1)
    return vmobject


BRAND_WHITE = "#F5F7FA"
BRAND_GREEN = "#38D996"
BRAND_RED   = "#FF4D4D"
BRAND_GOLD  = "#FFD166"
BRAND_GRAY  = "#8A94A6"
BRAND_PANEL = "#0D1B2A"
BRAND_BG    = "#060F1A"
BRAND_NAVY  = "#0B1628"


def fm_glow_around(mobject, color=None, n_layers=3):
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
            fill_color=panel_color, fill_opacity=1.0,
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
    val = Text(value_text, font_size=value_size, color=text_color, weight=BOLD)
    lbl = Text(label_text, font_size=label_size, color=accent_color)
    content = VGroup(lbl, val).arrange(DOWN, buff=0.18)
    box = SurroundingRectangle(
        content, buff=buff,
        color=accent_color,
        fill_color=panel_color, fill_opacity=1.0,
        corner_radius=0.18,
    )
    return VGroup(box, content)


def fm_two_cards(left_label, left_val, left_color,
                  right_label, right_val, right_color,
                  panel_color=BRAND_PANEL, text_color=BRAND_WHITE,
                  label_size=30, value_size=68, spacing=0.7):
    left  = fm_card(left_label,  left_val,  left_color,  panel_color, text_color, label_size, value_size)
    right = fm_card(right_label, right_val, right_color, panel_color, text_color, label_size, value_size)
    group = VGroup(left, right).arrange(RIGHT, buff=spacing)
    safe_w = config.frame_width * 0.88
    if group.width > safe_w:
        group.scale(safe_w / group.width)
    return group


def fm_card_row(items, panel_color=BRAND_PANEL, text_color=BRAND_WHITE,
                 label_size=26, value_size=44, spacing=0.45):
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
                        accent_color=BRAND_GOLD, prefix="", suffix="",
                        duration=3.0, position=None, value_size=130, label_size=38):
    if position is None:
        position = ORIGIN
    tracker = ValueTracker(float(start_val))
    end_f   = float(end_val)
    is_whole = float(end_val) == int(float(end_val))
    use_decimal = isinstance(end_val, float) and not is_whole

    def _num():
        v = tracker.get_value()
        if use_decimal:
            s = f"{prefix}{v:,.2f}{suffix}"
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
        bar   = RoundedRectangle(width=bar_w, height=bar_h, corner_radius=0.06)
        bar.set_fill(c, opacity=0.92)
        bar.set_stroke(c, width=1.5, opacity=0.55)
        bar.move_to([x, base_y + bar_h / 2, 0])
        bars.add(bar)

        val_str = f"{int(v):,}" if isinstance(v, int) else f"{v:.2f}"
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
    grow_t = max(min(duration * 0.62, duration - 0.45), 0.1)
    hold_t = max(duration - grow_t - 0.38, 0.05)
    scene.play(
        LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.18),
        run_time=grow_t, rate_func=smooth,
    )
    scene.play(
        LaggedStart(*[FadeIn(l) for l in val_labels], lag_ratio=0.12),
        run_time=0.38, rate_func=smooth,
    )
    scene.wait(hold_t)
    return bars, val_labels


def fm_animate_gauge(scene, value, max_val, label_text,
                      accent_color=BRAND_GREEN, duration=3.0,
                      position=None, radius=2.0):
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
    _fm_set_line_smooth(line, pts)
    line.set_stroke(color=accent_color, width=4.5, opacity=0.95)

    baseline_y = y_lo
    fill_pts   = pts + [axes.c2p(n - 1, baseline_y), axes.c2p(0, baseline_y)]
    fill_region = Polygon(*fill_pts, fill_opacity=0.20, stroke_width=0)
    fill_region.set_color_by_gradient(accent_color, BRAND_BG)

    end_dot = Dot(pts[-1], color=accent_color, radius=0.13)
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
    _frame_right_edge = config.frame_width / 2 - 0.25
    _frame_left_edge  = -config.frame_width / 2 + 0.25

    if title_text:
        ttl = Text(title_text, font_size=30, color=BRAND_GRAY)
        ttl.next_to(axes, UP, buff=0.22)
        scene.add(ttl)

    scene.add(axes, fill_region)
    grow_t  = max(min(duration * 0.70, duration - 0.55), 0.1)
    label_t = 0.4
    hold_t  = max(duration - grow_t - label_t, 0.05)
    scene.play(Create(line), run_time=grow_t, rate_func=smooth)
    curve_end = pts[-1]
    end_dot.move_to(curve_end)
    end_lbl.next_to(end_dot, _lbl_dir, buff=0.18)
    if end_lbl.get_right()[0] > _frame_right_edge:
        end_lbl.shift(LEFT * (end_lbl.get_right()[0] - _frame_right_edge))
    if end_lbl.get_left()[0] < _frame_left_edge:
        end_lbl.shift(RIGHT * (_frame_left_edge - end_lbl.get_left()[0]))
    scene.play(FadeIn(end_dot), Write(end_lbl), run_time=label_t)
    scene.wait(hold_t)
    return axes, line, end_dot


def fm_animate_line_chart_multi(scene, series, duration=4.0, title_text=""):
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
    _lbl_dirs = []
    _end_pts = []
    for s in series:
        y_values = s["y_values"]
        color    = s.get("color", BRAND_GREEN)
        pts      = [axes.c2p(i, y_values[i]) for i in range(n)]
        line     = VMobject()
        _fm_set_line_smooth(line, pts)
        line.set_stroke(color=color, width=4.5, opacity=0.95)
        lines.append(line)
        _end_pts.append(pts[-1])

        end_dot = Dot(pts[-1], color=color, radius=0.11)
        end_lbl = Text(s.get("label", ""), font_size=26, color=color, weight=BOLD)
        _dot_pt2 = axes.c2p(n - 1, y_values[-1])
        _dot_x   = _dot_pt2[0]
        _dot_y2  = _dot_pt2[1]
        _safe_right = config.frame_width * 0.38
        _low_thresh2 = -config.frame_height * 0.20
        if _dot_y2 < _low_thresh2:
            _lbl_dir2 = UP
        else:
            _lbl_dir2 = UR if _dot_x < _safe_right else UL
        end_dots.append(end_dot)
        end_lbls.append(end_lbl)
        _lbl_dirs.append(_lbl_dir2)

    if title_text:
        ttl = Text(title_text, font_size=30, color=BRAND_GRAY)
        ttl.next_to(axes, UP, buff=0.22)
        scene.add(ttl)

    scene.add(axes)
    grow_t  = max(min(duration * 0.65, duration - 0.6), 0.1)
    label_t = 0.45
    hold_t  = max(duration - grow_t - label_t, 0.05)
    scene.play(*[Create(l) for l in lines], run_time=grow_t, rate_func=smooth)

    _frame_right_edge = config.frame_width / 2 - 0.25
    _frame_left_edge2 = -config.frame_width / 2 + 0.25
    for i, (line_obj, end_dot, end_lbl, ldir) in enumerate(zip(lines, end_dots, end_lbls, _lbl_dirs)):
        end_dot.move_to(_end_pts[i])
        end_lbl.next_to(end_dot, ldir, buff=0.12)
        if end_lbl.get_right()[0] > _frame_right_edge:
            end_lbl.shift(LEFT * (end_lbl.get_right()[0] - _frame_right_edge))
        if end_lbl.get_left()[0] < _frame_left_edge2:
            end_lbl.shift(RIGHT * (_frame_left_edge2 - end_lbl.get_left()[0]))

    order   = sorted(range(len(series)), key=lambda i: series[i]["y_values"][-1], reverse=True)
    min_gap = 0.4
    for k in range(1, len(order)):
        prev_i = order[k - 1]
        cur_i  = order[k]
        gap = end_lbls[prev_i].get_bottom()[1] - end_lbls[cur_i].get_top()[1]
        if gap < min_gap:
            end_lbls[cur_i].shift(DOWN * (min_gap - gap))

    scene.play(
        *[FadeIn(d) for d in end_dots],
        *[Write(l) for l in end_lbls],
        run_time=label_t,
    )
    scene.wait(hold_t)
    return axes, lines


def fm_animate_waterfall(scene, steps, duration=4.5):
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

        bar = RoundedRectangle(width=bar_w, height=bar_h, corner_radius=0.05)
        bar.set_fill(c, opacity=0.9)
        bar.set_stroke(c, width=1.5, opacity=0.55)
        bar.move_to([x_pos, y0 + bar_h / 2, 0])
        bars.add(bar)

        prefix   = "-" if v < 0 else ""
        val_str  = f"{prefix}{int(abs(v)):,}" if abs(v) >= 1 else f"{prefix}{abs(v):.2f}"
        val_lbl  = Text(val_str, font_size=22, color=c, weight=BOLD)
        val_lbl.next_to(bar, DOWN if (v < 0) else UP, buff=0.08)
        cat_lbl  = Text(step.get("label", ""), font_size=18, color=BRAND_GRAY)
        if v < 0:
            cat_lbl.next_to(val_lbl, DOWN, buff=0.10)
        else:
            cat_lbl.next_to(bar, DOWN, buff=0.08)
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
            dot.set_stroke(accent_color, width=1.2, opacity=0.7)
        else:
            dot.set_fill(BRAND_GRAY, opacity=0.12)
            dot.set_stroke(BRAND_GRAY, width=1.0, opacity=0.28)
        dot.move_to([x, y, 0])
        icons.add(dot)

    pct     = filled / max(total, 1) * 100
    pct_lbl = Text(f"{pct:.0f}%", font_size=80, color=BRAND_WHITE, weight=BOLD)
    cat_lbl = Text(label_text, font_size=32, color=accent_color)

    icon_group = VGroup(icons)
    icon_group.move_to(position + LEFT * 2.2)
    text_group = VGroup(pct_lbl, cat_lbl).arrange(DOWN, buff=0.28)
    text_group.move_to(position + RIGHT * 2.8)

    fm_clamp_to_frame(icon_group, text_group)

    anim_t = max(min(duration * 0.68, duration - 0.4), 0.1)
    hold_t = max(duration - anim_t, 0.05)
    scene.play(
        LaggedStart(*[FadeIn(ic) for ic in icons], lag_ratio=0.04),
        FadeIn(text_group),
        run_time=anim_t, rate_func=smooth,
    )
    scene.wait(hold_t)
    return icons, pct_lbl


def fm_animate_stacked_cards(scene, items, duration=4.0):
    cards = fm_stacked_cards(items)
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
    if position is None:
        position = ORIGIN

    span   = max(range_high - range_low, 1.0)
    scale  = bar_length / span

    def _x(v):
        return -bar_length / 2 + (v - range_low) * scale

    band_w  = (range_high - range_low) * scale
    band    = RoundedRectangle(width=band_w, height=0.55, corner_radius=0.08)
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
        b = RoundedRectangle(width=w, height=0.32, corner_radius=0.06)
        b.set_fill(accent_color, opacity=0.95)
        b.set_stroke(accent_color, width=1.0, opacity=0.6)
        b.move_to([_x(range_low) + w / 2, 0, 0])
        b.shift(position)
        return b

    bar = always_redraw(_bar)

    target_lbl = Text(f"Target: {int(target):,}", font_size=26, color=BRAND_WHITE)
    target_lbl.next_to(tick, UP, buff=0.22)
    actual_lbl = Text(f"{int(actual):,}", font_size=42, color=accent_color, weight=BOLD)
    actual_lbl.next_to(band, DOWN, buff=0.32)
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
        bar    = RoundedRectangle(width=bar_w, height=bar_h, corner_radius=0.06)
        bar.set_fill(color, opacity=0.92)
        bar.set_stroke(color, width=1.5, opacity=0.55)
        bar.move_to([x, y_bot + bar_h / 2, 0])
        bars.add(bar)

        v_str   = f"{int(abs(value)):,}" if abs(value) >= 1 else f"{abs(value):.2f}"
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

    min_gap = 0.08
    for k in range(1, len(cat_labels)):
        prev = cat_labels[k - 1]
        cur  = cat_labels[k]
        overlap = prev.get_right()[0] - cur.get_left()[0] + min_gap
        if overlap > 0:
            cur.shift(RIGHT * (overlap / 2))
            prev.shift(LEFT * (overlap / 2))

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
    """Pure geometry icons — no SVGMobject, no image loading.
    Math/science icons: 'sigma', 'integral', 'pi_sym', 'infinity',
                        'gradient', 'neuron', 'matrix_sym', 'derivative'.
    General icons: 'dollar', 'coin', 'house', 'person', 'clock',
                   'arrow_up', 'arrow_down', 'warning', 'checkmark', 'fire'.
    Returns a VGroup."""
    g = VGroup()
    s = size

    if name == "sigma":
        g.add(Text("Σ", font_size=int(72 * s), color=color, weight=BOLD))

    elif name == "integral":
        g.add(Text("∫", font_size=int(80 * s), color=color, weight=BOLD))

    elif name == "pi_sym":
        g.add(Text("π", font_size=int(72 * s), color=color, weight=BOLD))

    elif name == "infinity":
        g.add(Text("∞", font_size=int(72 * s), color=color, weight=BOLD))

    elif name == "gradient":
        g.add(Text("∇", font_size=int(72 * s), color=color, weight=BOLD))

    elif name == "derivative":
        g.add(Text("d/dx", font_size=int(52 * s), color=color, weight=BOLD))

    elif name == "matrix_sym":
        inner = Text("[ ]", font_size=int(72 * s), color=color, weight=BOLD)
        g.add(inner)

    elif name == "neuron":
        body = Circle(radius=0.38 * s)
        body.set_fill(color, opacity=0.88).set_stroke(color, width=2.0)
        for angle in [PI * 0.25, PI * 0.75, PI * 1.25, PI * 1.75]:
            dendrite = Line(
                [0, 0, 0],
                [0.6 * s * math.cos(angle), 0.6 * s * math.sin(angle), 0]
            )
            dendrite.set_stroke(color, width=2.0, opacity=0.7)
            g.add(dendrite)
        g.add(body)

    elif name == "dollar":
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
        body = RoundedRectangle(width=0.65*s, height=0.42*s, corner_radius=0.04)
        body.set_fill(color, opacity=0.65).set_stroke(color, width=1.5, opacity=0.6)
        body.next_to(roof, DOWN, buff=0)
        g.add(roof, body)

    elif name == "person":
        head = Circle(radius=0.18 * s)
        head.set_fill(color, opacity=0.88).set_stroke(color, width=1.2)
        body = RoundedRectangle(width=0.30*s, height=0.36*s, corner_radius=0.06)
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
        shaft = RoundedRectangle(width=0.14*s, height=0.38*s, corner_radius=0.04)
        shaft.set_fill(color, opacity=0.90).shift(DOWN * 0.10 * s)
        tip = Polygon([-0.28*s, 0, 0], [0, 0.32*s, 0], [0.28*s, 0, 0])
        tip.set_fill(color, opacity=0.90).shift(UP * 0.14 * s)
        g.add(shaft, tip)

    elif name == "arrow_down":
        shaft = RoundedRectangle(width=0.14*s, height=0.38*s, corner_radius=0.04)
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


def fm_animate_vector(scene, direction, label_text, accent_color=BRAND_GOLD,
                       duration=3.5, origin=None, scale=2.5, show_components=False):
    """Animated vector arrow drawn from origin via Create.
    direction: [dx, dy] normalized direction. scale: arrow length.
    show_components: if True, draws dashed x and y component lines.
    Returns (arrow, label_mob)."""
    if origin is None:
        origin = ORIGIN

    dx, dy = direction[0], direction[1]
    length = math.sqrt(dx**2 + dy**2)
    if length > 1e-9:
        dx, dy = dx / length * scale, dy / length * scale

    tip = [origin[0] + dx, origin[1] + dy, 0]
    arrow = Arrow(
        start=origin, end=tip,
        buff=0,
        stroke_width=5,
        max_tip_length_to_length_ratio=0.18,
        color=accent_color,
    )

    lbl = Text(label_text, font_size=36, color=accent_color, weight=BOLD)
    mid = [(origin[0] + tip[0]) / 2, (origin[1] + tip[1]) / 2, 0]
    perp_x = -dy / max(scale, 0.01) * 0.45
    perp_y =  dx / max(scale, 0.01) * 0.45
    lbl.move_to([mid[0] + perp_x, mid[1] + perp_y, 0])

    components = VGroup()
    if show_components:
        comp_x = Line(origin, [origin[0] + dx, origin[1], 0])
        comp_x.set_stroke(BRAND_GRAY, width=2.0, opacity=0.5)
        comp_y = Line([origin[0] + dx, origin[1], 0], tip)
        comp_y.set_stroke(BRAND_GRAY, width=2.0, opacity=0.5)
        comp_x_lbl = Text(f"{dx:.1f}", font_size=22, color=BRAND_GRAY)
        comp_x_lbl.next_to(comp_x, DOWN, buff=0.12)
        comp_y_lbl = Text(f"{dy:.1f}", font_size=22, color=BRAND_GRAY)
        comp_y_lbl.next_to(comp_y, RIGHT, buff=0.12)
        components.add(comp_x, comp_y, comp_x_lbl, comp_y_lbl)
        scene.add(components)

    draw_t = max(min(duration * 0.55, 1.6), 0.1)
    hold_t = max(duration - draw_t - 0.3, 0.05)
    scene.play(Create(arrow), run_time=draw_t, rate_func=smooth)
    scene.play(FadeIn(lbl, shift=UP * 0.12), run_time=0.3, rate_func=smooth)
    scene.wait(hold_t)
    return arrow, lbl


def fm_animate_matrix(scene, rows_data, label_text="", accent_color=BRAND_GOLD,
                       duration=4.0, position=None, cell_size=0.9, font_size=36):
    """Animated matrix: bracket notation with cells fading in row by row.
    rows_data: list of lists of strings or numbers e.g. [['a','b'],['c','d']].
    Returns (matrix_group, label_mob)."""
    if position is None:
        position = ORIGIN

    n_rows = len(rows_data)
    n_cols = max(len(r) for r in rows_data) if rows_data else 1

    cells = VGroup()
    for r_idx, row in enumerate(rows_data):
        for c_idx, val in enumerate(row):
            val_str = str(val)
            cell_txt = Text(val_str, font_size=font_size, color=BRAND_WHITE, weight=BOLD)
            x = (c_idx - (n_cols - 1) / 2) * cell_size
            y = ((n_rows - 1) / 2 - r_idx) * cell_size
            cell_txt.move_to([x, y, 0])
            cells.add(cell_txt)

    total_w = n_cols * cell_size
    total_h = n_rows * cell_size
    bracket_h = total_h + 0.3

    left_top    = [-total_w / 2 - 0.55, bracket_h / 2, 0]
    left_mid_t  = [-total_w / 2 - 0.35, bracket_h / 2, 0]
    left_mid_b  = [-total_w / 2 - 0.35, -bracket_h / 2, 0]
    left_bot    = [-total_w / 2 - 0.55, -bracket_h / 2, 0]
    right_top   = [total_w / 2 + 0.55, bracket_h / 2, 0]
    right_mid_t = [total_w / 2 + 0.35, bracket_h / 2, 0]
    right_mid_b = [total_w / 2 + 0.35, -bracket_h / 2, 0]
    right_bot   = [total_w / 2 + 0.55, -bracket_h / 2, 0]

    left_bracket = VMobject()
    left_bracket.set_points_as_corners([left_top, left_mid_t, left_mid_b, left_bot])
    left_bracket.set_stroke(accent_color, width=3.5, opacity=0.9)

    right_bracket = VMobject()
    right_bracket.set_points_as_corners([right_top, right_mid_t, right_mid_b, right_bot])
    right_bracket.set_stroke(accent_color, width=3.5, opacity=0.9)

    matrix_group = VGroup(left_bracket, right_bracket, cells)
    matrix_group.move_to(position)

    lbl_mob = None
    if label_text:
        lbl_mob = Text(label_text, font_size=32, color=accent_color)
        lbl_mob.next_to(matrix_group, DOWN, buff=0.4)

    safe_w = config.frame_width * 0.85
    safe_h = config.frame_height * 0.80
    combined = VGroup(matrix_group) if not lbl_mob else VGroup(matrix_group, lbl_mob)
    if combined.width > safe_w:
        combined.scale(safe_w / combined.width)
    if combined.height > safe_h:
        combined.scale(safe_h / combined.height)

    draw_t = max(min(duration * 0.28, 1.0), 0.1)
    per_row = max((duration - draw_t - 0.3) / max(n_rows, 1), 0.12)
    hold_t = max(duration - draw_t - per_row * n_rows - 0.2, 0.05)

    scene.play(
        Create(left_bracket), Create(right_bracket),
        run_time=draw_t, rate_func=smooth,
    )
    for r_idx in range(n_rows):
        row_cells = [cells[r_idx * n_cols + c] for c in range(min(n_cols, len(rows_data[r_idx])))]
        scene.play(
            LaggedStart(*[FadeIn(c, scale=0.85) for c in row_cells], lag_ratio=0.15),
            run_time=per_row, rate_func=smooth,
        )
    if lbl_mob:
        scene.play(FadeIn(lbl_mob), run_time=0.2)
    scene.wait(hold_t)
    return matrix_group, lbl_mob


def fm_animate_bell_curve(scene, label_text="", accent_color=BRAND_GOLD,
                           duration=4.0, position=None, show_std_regions=True,
                           mean_label="μ", std_label="σ"):
    """Animated normal distribution curve drawn via Create.
    show_std_regions: shades the 1σ region under the curve.
    Returns (curve, fill_region, label_mob)."""
    if position is None:
        position = ORIGIN

    n_pts = 120
    x_range = 3.8
    curve_w = 9.0
    curve_h = 3.8

    xs = [(-x_range + i * 2 * x_range / (n_pts - 1)) for i in range(n_pts)]
    ys = [math.exp(-0.5 * x * x) for x in xs]

    pts = [
        [position[0] + x / x_range * curve_w / 2,
         position[1] + y * curve_h,
         0]
        for x, y in zip(xs, ys)
    ]

    curve = VMobject()
    _fm_set_line_smooth(curve, pts)
    curve.set_stroke(accent_color, width=4.0, opacity=0.95)

    baseline_y = position[1]
    fill_pts = pts + [[pts[-1][0], baseline_y, 0], [pts[0][0], baseline_y, 0]]
    fill_region = Polygon(*fill_pts, fill_opacity=0.15, stroke_width=0)
    fill_region.set_fill(accent_color)

    std_fill = None
    if show_std_regions:
        sigma_x = curve_w / 2 / x_range
        std_xs  = [x for x in xs if abs(x) <= 1.0]
        std_ys  = [math.exp(-0.5 * x * x) for x in std_xs]
        std_pts = [
            [position[0] + x / x_range * curve_w / 2,
             position[1] + y * curve_h,
             0]
            for x, y in zip(std_xs, std_ys)
        ]
        std_fill_pts = std_pts + [
            [std_pts[-1][0], baseline_y, 0],
            [std_pts[0][0],  baseline_y, 0],
        ]
        std_fill = Polygon(*std_fill_pts, fill_opacity=0.32, stroke_width=0)
        std_fill.set_fill(accent_color)

    mean_tick = Line(
        [position[0], baseline_y - 0.1, 0],
        [position[0], baseline_y + 0.1, 0],
    )
    mean_tick.set_stroke(BRAND_WHITE, width=2.5, opacity=0.7)
    mean_lbl = Text(mean_label, font_size=28, color=BRAND_WHITE)
    mean_lbl.next_to(mean_tick, DOWN, buff=0.18)

    sigma_x_pos = position[0] + curve_w / 2 / x_range
    std_tick = Line(
        [sigma_x_pos, baseline_y - 0.1, 0],
        [sigma_x_pos, baseline_y + 0.1, 0],
    )
    std_tick.set_stroke(BRAND_GRAY, width=2.0, opacity=0.6)
    std_lbl = Text(std_label, font_size=24, color=BRAND_GRAY)
    std_lbl.next_to(std_tick, DOWN, buff=0.18)

    lbl_mob = None
    if label_text:
        lbl_mob = Text(label_text, font_size=32, color=accent_color)
        lbl_mob.next_to(
            [position[0], position[1] + curve_h + 0.2, 0],
            UP, buff=0.1
        )

    draw_t = max(min(duration * 0.50, 2.0), 0.1)
    fill_t = max(min(duration * 0.22, 0.8), 0.05)
    hold_t = max(duration - draw_t - fill_t - 0.2, 0.05)

    scene.add(fill_region)
    scene.play(Create(curve), run_time=draw_t, rate_func=smooth)
    if std_fill is not None:
        scene.play(FadeIn(std_fill), run_time=fill_t, rate_func=smooth)
    scene.play(
        FadeIn(mean_tick), FadeIn(mean_lbl),
        FadeIn(std_tick),  FadeIn(std_lbl),
        run_time=0.2,
    )
    if lbl_mob:
        scene.play(FadeIn(lbl_mob, shift=UP * 0.1), run_time=0.2)
    scene.wait(hold_t)
    return curve, fill_region, lbl_mob


def fm_animate_scatter(scene, points, label_text="", accent_color=BRAND_GOLD,
                        duration=4.0, position=None, show_regression=False,
                        x_label="x", y_label="y"):
    """Animated scatter plot. points: list of (x, y) tuples.
    show_regression: draws a best-fit line through the data.
    Returns (dots_group, regression_line)."""
    if position is None:
        position = ORIGIN
    if not points:
        return VGroup(), None

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1.0)

    plot_w = 8.0
    plot_h = 4.5
    pad    = 0.6

    def _to_screen(px, py):
        sx = position[0] - plot_w / 2 + pad + (px - x_min) / x_span * (plot_w - 2 * pad)
        sy = position[1] - plot_h / 2 + pad + (py - y_min) / y_span * (plot_h - 2 * pad)
        return [sx, sy, 0]

    x_axis = Line(
        [position[0] - plot_w / 2, position[1] - plot_h / 2, 0],
        [position[0] + plot_w / 2, position[1] - plot_h / 2, 0],
    )
    x_axis.set_stroke(BRAND_GRAY, width=2.0, opacity=0.55)
    y_axis = Line(
        [position[0] - plot_w / 2, position[1] - plot_h / 2, 0],
        [position[0] - plot_w / 2, position[1] + plot_h / 2, 0],
    )
    y_axis.set_stroke(BRAND_GRAY, width=2.0, opacity=0.55)

    x_lbl_mob = Text(x_label, font_size=24, color=BRAND_GRAY)
    x_lbl_mob.next_to(x_axis, DOWN, buff=0.2)
    y_lbl_mob = Text(y_label, font_size=24, color=BRAND_GRAY)
    y_lbl_mob.next_to(y_axis, LEFT, buff=0.2)

    dots = VGroup()
    for px, py in points:
        sp = _to_screen(px, py)
        dot = Dot(sp, radius=0.10, color=accent_color)
        dot.set_fill(accent_color, opacity=0.85)
        dots.add(dot)

    reg_line = None
    if show_regression and len(points) >= 2:
        n = len(points)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        num = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
        den = sum((xs[i] - mean_x) ** 2 for i in range(n))
        if abs(den) > 1e-9:
            slope = num / den
            intercept = mean_y - slope * mean_x
            y_at_xmin = slope * x_min + intercept
            y_at_xmax = slope * x_max + intercept
            reg_start = _to_screen(x_min, y_at_xmin)
            reg_end   = _to_screen(x_max, y_at_xmax)
            reg_line  = Line(reg_start, reg_end)
            reg_line.set_stroke(BRAND_RED, width=3.0, opacity=0.85)

    lbl_mob = None
    if label_text:
        lbl_mob = Text(label_text, font_size=30, color=accent_color)
        lbl_mob.move_to([position[0], position[1] + plot_h / 2 + 0.4, 0])

    scene.add(x_axis, y_axis, x_lbl_mob, y_lbl_mob)
    if lbl_mob:
        scene.add(lbl_mob)

    dot_t  = max(min(duration * 0.62, 2.2), 0.1)
    hold_t = max(duration - dot_t - 0.35, 0.05)

    scene.play(
        LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.06),
        run_time=dot_t, rate_func=smooth,
    )
    if reg_line is not None:
        scene.play(Create(reg_line), run_time=0.35, rate_func=smooth)
    scene.wait(hold_t)
    return dots, reg_line


def fm_animate_probability_bar(scene, outcomes, label_text="",
                                accent_color=BRAND_GOLD, duration=4.0,
                                position=None):
    """Probability distribution bar chart. Each bar height = probability (0-1).
    outcomes: list of (label_str, probability_float) tuples.
    All probabilities should sum to 1. Returns (bars, val_labels)."""
    if position is None:
        position = ORIGIN
    if not outcomes:
        return VGroup(), VGroup()

    n       = len(outcomes)
    bar_w   = min(1.4, 9.0 / max(n, 1))
    spacing = bar_w * 1.6
    total_w = (n - 1) * spacing
    chart_h = 4.0
    base_y  = position[1] - chart_h / 2

    baseline = Line(
        [position[0] - total_w / 2 - bar_w / 2 - 0.3, base_y, 0],
        [position[0] + total_w / 2 + bar_w / 2 + 0.3, base_y, 0],
    )
    baseline.set_stroke(BRAND_GRAY, width=2.0, opacity=0.5)

    bars       = VGroup()
    val_labels = VGroup()
    cat_labels = VGroup()

    for i, (name, prob) in enumerate(outcomes):
        prob = max(0.0, min(1.0, float(prob)))
        x    = position[0] - total_w / 2 + i * spacing
        bar_h = max(prob * chart_h, 0.06)
        bar = RoundedRectangle(width=bar_w, height=bar_h, corner_radius=0.06)
        bar.set_fill(accent_color, opacity=0.88)
        bar.set_stroke(accent_color, width=1.5, opacity=0.55)
        bar.move_to([x, base_y + bar_h / 2, 0])
        bars.add(bar)

        pct_str = f"{prob * 100:.1f}%"
        val_lbl = Text(pct_str, font_size=24, color=accent_color, weight=BOLD)
        val_lbl.next_to(bar, UP, buff=0.1)
        val_labels.add(val_lbl)

        cat_lbl = Text(name, font_size=20, color=BRAND_GRAY)
        cat_lbl.next_to(bar, DOWN, buff=0.15)
        cat_labels.add(cat_lbl)

    chart_group = VGroup(baseline, bars, val_labels, cat_labels)
    chart_group.move_to(position)

    lbl_mob = None
    if label_text:
        lbl_mob = Text(label_text, font_size=30, color=accent_color)
        lbl_mob.next_to(chart_group, UP, buff=0.28)
        scene.add(lbl_mob)

    scene.add(baseline, cat_labels)
    grow_t = max(min(duration * 0.65, 2.2), 0.1)
    hold_t = max(duration - grow_t - 0.35, 0.05)

    scene.play(
        LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.16),
        run_time=grow_t, rate_func=smooth,
    )
    scene.play(
        LaggedStart(*[FadeIn(l) for l in val_labels], lag_ratio=0.12),
        run_time=0.35, rate_func=smooth,
    )
    scene.wait(hold_t)
    return bars, val_labels


def fm_animate_number_line(scene, value, min_val, max_val, label_text="",
                            accent_color=BRAND_GOLD, duration=3.5,
                            position=None, line_length=9.0,
                            tick_labels=None):
    """Animated number line with a glowing dot moving to the target value.
    Returns (dot, line_mob, label_mob)."""
    if position is None:
        position = ORIGIN

    line_mob = Line(
        [position[0] - line_length / 2, position[1], 0],
        [position[0] + line_length / 2, position[1], 0],
    )
    line_mob.set_stroke(BRAND_GRAY, width=3.0, opacity=0.55)

    span = max(max_val - min_val, 1.0)

    n_ticks = 5
    ticks_group = VGroup()
    for i in range(n_ticks + 1):
        frac = i / n_ticks
        tick_val = min_val + frac * span
        tx = position[0] - line_length / 2 + frac * line_length
        tick = Line([tx, position[1] - 0.15, 0], [tx, position[1] + 0.15, 0])
        tick.set_stroke(BRAND_GRAY, width=2.0, opacity=0.4)
        ticks_group.add(tick)
        if tick_labels:
            if i < len(tick_labels):
                tl = Text(str(tick_labels[i]), font_size=20, color=BRAND_GRAY)
            else:
                tl = Text(f"{tick_val:.1f}", font_size=20, color=BRAND_GRAY)
        else:
            tl = Text(f"{tick_val:.1f}" if isinstance(tick_val, float) else str(int(tick_val)),
                      font_size=20, color=BRAND_GRAY)
        tl.next_to(tick, DOWN, buff=0.15)
        ticks_group.add(tl)

    frac_val = max(0.0, min(1.0, (value - min_val) / span))
    target_x = position[0] - line_length / 2 + frac_val * line_length

    tracker = ValueTracker(position[0] - line_length / 2)

    def _dot():
        cx = tracker.get_value()
        d  = Dot([cx, position[1], 0], radius=0.18, color=accent_color)
        d.set_fill(accent_color, opacity=1.0)
        gl = fm_glow_around(d, color=accent_color, n_layers=3)
        return gl

    dot = always_redraw(_dot)

    val_str = f"{value:.2f}" if isinstance(value, float) else str(int(value))
    val_lbl = Text(val_str, font_size=44, color=accent_color, weight=BOLD)
    val_lbl.move_to([target_x, position[1] + 0.65, 0])

    lbl_mob = None
    if label_text:
        lbl_mob = Text(label_text, font_size=30, color=BRAND_GRAY)
        lbl_mob.move_to([position[0], position[1] - 0.75, 0])

    scene.add(line_mob, ticks_group, dot)
    if lbl_mob:
        scene.add(lbl_mob)

    move_t = max(min(duration * 0.65, 2.0), 0.1)
    hold_t = max(duration - move_t - 0.3, 0.05)

    scene.play(
        tracker.animate.set_value(target_x),
        run_time=move_t, rate_func=smooth,
    )
    scene.play(FadeIn(val_lbl, shift=DOWN * 0.1), run_time=0.3)
    scene.wait(hold_t)
    return dot, line_mob, lbl_mob