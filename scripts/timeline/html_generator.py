"""HTML generation for Mirage timeline visualization.

Phase 4: Canvas-based renderer with virtual scroll.
Phase 6: CompactBarEncoder for large graphs.
"""
import json
import os
from typing import List, Optional

from timeline import _short, _parse_trace_name, STRAGGLER_RATIO
from timeline.metrics import _generate_sched_html, _generate_data_sched_html

_HERE = os.path.dirname(__file__)
_TEMPLATES = os.path.join(_HERE, 'templates')


def _tpl(name: str) -> str:
    with open(os.path.join(_TEMPLATES, name), 'r') as f:
        return f.read()


class CompactBarEncoder:
    """Encodes List[row_dict] to a compact array-of-arrays format.

    Reduces JSON payload by ~3-4x for large graphs by deduplicating
    color and name strings.

    Compact bar format: [start, end, colorIdx, nameIdx, blocks, avg_dur, execKind, traceKeyIdx, dataId]
    traceKeyIdx = -1 means trace_key == name (no separate entry needed)
    """

    def encode(self, rows: list) -> dict:
        color_idx = {}
        name_idx = {}
        colors: list = []
        names: list = []

        def ci(c):
            if c not in color_idx:
                color_idx[c] = len(colors)
                colors.append(c)
            return color_idx[c]

        def ni(n):
            if n not in name_idx:
                name_idx[n] = len(names)
                names.append(n)
            return name_idx[n]

        compact_rows = []
        for row in rows:
            cbars = []
            for b in row['bars']:
                name = b['name']
                tk = b.get('trace_key', '')
                tki = ni(tk) if (tk and tk != name) else -1
                cbars.append([
                    b['start'], b['end'],
                    ci(b.get('color', '#888')),
                    ni(name),
                    b.get('blocks', 1),
                    b.get('avg_dur', 0),
                    b.get('execution_kind', 0),
                    tki,
                    b.get('data_id', -1),
                ])
            compact_rows.append({'l': row['label'], 'b': cbars})

        return {'colors': colors, 'names': names, 'rows': compact_rows}

    @staticmethod
    def decode_js() -> str:
        """JS snippet that decodes the compact format into the standard rows format."""
        return """
function _decodeRows(d) {
  const {colors, names, rows} = d;
  return rows.map(r => ({
    label: r.l,
    bars: r.b.map(b => ({
      start: b[0], end: b[1],
      color: colors[b[2]],
      name: names[b[3]],
      blocks: b[4], avg_dur: b[5],
      execution_kind: b[6],
      trace_key: b[7] >= 0 ? names[b[7]] : names[b[3]],
      data_id: b[8],
    }))
  }));
}
"""


def _build_legend(tt_rows: list) -> list:
    seen = {}
    for row in tt_rows:
        for bar in row.get('bars', []):
            c = bar['color']
            if c not in seen:
                tts, _, _ = _parse_trace_name(bar['name'])
                seen[c] = _short(tts)
    return [{'color': c, 'name': n} for c, n in seen.items()]


def _build_stats_html(type_stats: dict, total_dur: int) -> str:
    rows_html = ''
    total_dur_sum = sum(v['total_dur'] for v in type_stats.values())
    for tts, st in sorted(type_stats.items(), key=lambda x: -x[1]['total_dur']):
        avg = st['total_dur'] / st['count'] if st['count'] else 0
        pct = st['total_dur'] / total_dur_sum * 100 if total_dur_sum else 0
        rows_html += (
            f'<tr><td>{_short(tts)}</td><td>{st["count"]}</td>'
            f'<td>{st["total_dur"]/1e6:.3f} ms</td>'
            f'<td>{avg/1e3:.1f} us</td>'
            f'<td>{pct:.1f}%</td></tr>\n'
        )
    return rows_html


def generate_html(
    view_rows: dict,          # {tab_id: [row_dicts]}
    active_adapters: list,    # applicable ViewAdapter instances (ordered)
    global_start: int,
    total_dur: int,
    type_stats: dict,
    graph_info: tuple,        # (num_events, num_tasks)
    group_deps: dict,
    sched_metrics: Optional[dict],
    data_sched_metrics: Optional[dict],
    output_path: str,
) -> None:
    encoder = CompactBarEncoder()
    num_events, num_tasks = graph_info

    # --- Tab buttons and panes ---
    tab_buttons = ''
    tab_panes = ''
    first = True
    for adapter in active_adapters:
        tid = adapter.tab_id
        active_cls = ' active' if first else ''
        tab_buttons += (
            f'<button class="tab-btn{active_cls}" data-tab="{tid}" '
            f'onclick="sw(\'{tid}\',this)">{adapter.tab_label}</button>\n'
        )
        tab_panes += (
            f'<div id="tab-{tid}" class="tab{active_cls}">'
            f'<div class="tc" id="tc-{tid}"></div></div>\n'
        )
        first = False

    # --- Per-tab JS data (compact) ---
    tab_data_js = {}
    for adapter in active_adapters:
        rows = view_rows.get(adapter.tab_id, [])
        tab_data_js[adapter.tab_id] = encoder.encode(rows)

    # --- Legend from by-task-type rows ---
    tt_rows = view_rows.get('tt', [])
    legend = _build_legend(tt_rows)

    # --- GBL object ---
    gbl = {
        'totalDur': total_dur,
        'groupDeps': group_deps or {},
        'stragglerRatio': STRAGGLER_RATIO,
        'legend': legend,
    }

    # --- Stats table ---
    stats_rows = _build_stats_html(type_stats, total_dur)

    # --- Analysis sections ---
    sched_html = _generate_sched_html(sched_metrics) if sched_metrics else ''
    data_sched_html = _generate_data_sched_html(data_sched_metrics) if data_sched_metrics else ''
    analysis_parts = []
    if sched_html:
        analysis_parts.append(
            '<div class="sa"><h3>Schedule Analysis</h3>'
            '<p style="font-size:12px;color:#888;margin:0 0 10px">Block utilization, queue wait times, '
            'straggler effects, and the critical dependency path.</p>'
            f'{sched_html}</div>'
        )
    if data_sched_html:
        analysis_parts.append(
            '<div class="sa"><h3>Data-Level Analysis</h3>'
            '<p style="font-size:12px;color:#888;margin:0 0 10px">Per-data timing and overlap metrics.</p>'
            f'{data_sched_html}</div>'
        )
    analysis_html = '\n'.join(analysis_parts)

    # --- Decode + renderer JS ---
    renderer_js = _tpl('renderer.js')
    interactions_js = (
        CompactBarEncoder.decode_js()
        + '\n// Decode compact tab data\n'
        + '\n'.join(
            f'TAB_DATA["{tid}"] = _decodeRows(TAB_DATA["{tid}"]);'
            for tid in tab_data_js
        )
        + '\n'
        + _tpl('interactions.js')
    )

    # --- Build HTML via safe string replacement (avoids CSS/JS brace conflicts) ---
    shell = _tpl('shell.html')
    html = shell
    html = html.replace('%%CSS%%', _tpl('styles.css'))
    html = html.replace('%%NUM_EVENTS%%', str(num_events))
    html = html.replace('%%NUM_TASKS%%', str(num_tasks))
    html = html.replace('%%TOTAL_DUR_MS%%', f'{total_dur / 1e6:.2f}')
    html = html.replace('%%TAB_BUTTONS%%', tab_buttons)
    html = html.replace('%%TAB_PANES%%', tab_panes)
    html = html.replace('%%STATS_ROWS%%', stats_rows)
    html = html.replace('%%ANALYSIS_HTML%%', analysis_html)
    html = html.replace('%%JS_TAB_DATA%%', json.dumps(tab_data_js, separators=(',', ':')))
    html = html.replace('%%JS_GBL%%', json.dumps(gbl, separators=(',', ':')))
    html = html.replace('%%RENDERER_JS%%', renderer_js)
    html = html.replace('%%INTERACTIONS_JS%%', interactions_js)

    with open(output_path, 'w') as f:
        f.write(html)
    print(f'  -> {output_path}')


def _build_whatif_legend(rows_per_tab: list) -> list:
    """Build legend items from whatif row lists."""
    seen = {}
    for rows in rows_per_tab:
        for row in rows:
            for bar in row.get('bars', []):
                c = bar.get('color', '#888')
                if c not in seen:
                    tts, _, _ = _parse_trace_name(bar['name'])
                    seen[c] = _short(tts)
    return [{'color': c, 'name': n} for c, n in seen.items()]


def generate_whatif_html(whatif_data: dict, output_path: str) -> None:
    """Generate what-if HTML from templates/whatif_shell.html."""
    import json as _json
    legend = _build_whatif_legend([
        whatif_data['baseline_rows'],
        whatif_data['affinity_rows'],
    ])
    template = _tpl('whatif_shell.html')
    html = template.format(
        total_dur_ms=whatif_data['baseline_wall_us'] / 1000.0,
        js_baseline=_json.dumps(whatif_data['baseline_rows']),
        js_affinity=_json.dumps(whatif_data['affinity_rows']),
        js_legend=_json.dumps(legend),
        js_baseline_dur=whatif_data['baseline_wall_ns'],
        js_affinity_dur=whatif_data['affinity_wall_ns'],
        baseline_cards=whatif_data['baseline_cards'],
        baseline_tables=whatif_data['baseline_tables'],
        affinity_cards=whatif_data['affinity_cards'],
        affinity_tables=whatif_data['affinity_tables'],
    )
    with open(output_path, 'w') as f:
        f.write(html)
    print(f'  -> {output_path}')
