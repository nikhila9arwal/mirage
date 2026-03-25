'use strict';
class TimelineRenderer {
  constructor(container, rows, totalDur, groupDeps, stragglerRatio) {
    this._container = container;
    this._rows = rows;
    this._totalDur = totalDur;
    this._groupDeps = groupDeps;
    this._stragglerRatio = stragglerRatio;

    // State
    this._zoom = 1.0;
    this._hlKey = null;
    this._predKeys = new Set();
    this._succKeys = new Set();
    this._dimSet = new Set();
    this._rafPending = false;

    // Layout constants
    this.ROW_H = 26;
    this.BAR_H = 20;
    this.BAR_TOP = 3;
    this.LABEL_W = 220;
    this.AXIS_H = 24;
    this.TICK_N = 20;

    this._build();
  }

  _build() {
    const container = this._container;

    // Logical timeline div (sets scroll extent)
    this._tl = document.createElement('div');
    this._tl.style.position = 'relative';
    this._tl.style.minWidth = '100%';
    container.appendChild(this._tl);

    // Canvas (sticky inside the tl div)
    this._canvas = document.createElement('canvas');
    this._canvas.className = 'tl-canvas';
    this._tl.appendChild(this._canvas);

    this._ctx = this._canvas.getContext('2d');

    // Initial size
    this._resize();

    // Scroll handler
    container.addEventListener('scroll', () => {
      if (!this._rafPending) {
        this._rafPending = true;
        requestAnimationFrame(() => { this._rafPending = false; this._draw(); });
      }
    });

    // Resize observer
    if (typeof ResizeObserver !== 'undefined') {
      this._ro = new ResizeObserver(() => { this._resize(); this._draw(); });
      this._ro.observe(container);
    }

    // Mouse events on canvas
    this._canvas.addEventListener('mousemove', (e) => this._onMouseMove(e));
    this._canvas.addEventListener('click', (e) => this._onClick(e));
    this._canvas.addEventListener('mouseleave', () => {
      if (window._ttHide) window._ttHide();
    });

    this._draw();
  }

  _resize() {
    const c = this._container;
    const dpr = window.devicePixelRatio || 1;
    const maxH = Math.floor(window.innerHeight * 0.82);
    const cw = c.clientWidth;
    const ch = Math.min(c.clientHeight || maxH, maxH);

    this._canvas.width = cw * dpr;
    this._canvas.height = ch * dpr;
    this._canvas.style.width = cw + 'px';
    this._canvas.style.height = ch + 'px';
    this._ctx.scale(dpr, dpr);

    this._canvasW = cw;
    this._canvasH = ch;

    // Update logical tl size
    const TW = this._timelineW();
    const totalH = this.AXIS_H + this._rows.length * this.ROW_H;
    this._tl.style.width = (this.LABEL_W + TW) + 'px';
    this._tl.style.height = totalH + 'px';
  }

  _timelineW() {
    return Math.max(this._canvasW - this.LABEL_W, 200) * this._zoom;
  }

  setZoom(z) {
    this._zoom = z;
    this._resize();
    this._draw();
  }

  _sched() {
    if (!this._rafPending) {
      this._rafPending = true;
      requestAnimationFrame(() => { this._rafPending = false; this._draw(); });
    }
  }

  // Binary search: first index where bars[i].end >= t
  _bs(bars, t) {
    let lo = 0, hi = bars.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (bars[mid].end < t) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  _draw() {
    const ctx = this._ctx;
    const rows = this._rows;
    const D = this._totalDur;
    const TW = this._timelineW();
    const canvasW = this._canvasW;
    const canvasH = this._canvasH;
    const scrollTop = this._container.scrollTop;
    const scrollLeft = this._container.scrollLeft;
    const AXIS_H = this.AXIS_H;
    const ROW_H = this.ROW_H;
    const LABEL_W = this.LABEL_W;

    ctx.clearRect(0, 0, canvasW, canvasH);

    // Background
    ctx.fillStyle = '#16213e';
    ctx.fillRect(0, 0, canvasW, canvasH);

    // Virtual scroll: visible row range
    const firstRow = Math.max(0, Math.floor((scrollTop - AXIS_H) / ROW_H));
    const lastRow = Math.min(rows.length - 1, Math.ceil((scrollTop + canvasH - AXIS_H) / ROW_H));

    // Draw rows
    for (let ri = firstRow; ri <= lastRow; ri++) {
      const ry = AXIS_H + ri * ROW_H - scrollTop;
      this._drawRow(ctx, ri, ry, scrollLeft, TW, D, canvasW);
    }

    // Row separators
    ctx.strokeStyle = '#1a1a2e';
    ctx.lineWidth = 1;
    for (let ri = firstRow; ri <= lastRow; ri++) {
      const ry = AXIS_H + ri * ROW_H - scrollTop + ROW_H;
      ctx.beginPath();
      ctx.moveTo(LABEL_W, ry);
      ctx.lineTo(canvasW, ry);
      ctx.stroke();
    }

    // Redraw label column on top (sticky left)
    ctx.fillStyle = '#16213e';
    ctx.fillRect(0, AXIS_H, LABEL_W, canvasH - AXIS_H);
    ctx.strokeStyle = '#0f3460';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(LABEL_W, AXIS_H);
    ctx.lineTo(LABEL_W, canvasH);
    ctx.stroke();

    ctx.fillStyle = '#e0e0e0';
    ctx.font = '11px monospace';
    ctx.textBaseline = 'middle';
    for (let ri = firstRow; ri <= lastRow; ri++) {
      const ry = AXIS_H + ri * ROW_H - scrollTop + ROW_H / 2;
      const label = rows[ri].label;
      ctx.save();
      ctx.beginPath();
      ctx.rect(4, ry - ROW_H / 2, LABEL_W - 8, ROW_H);
      ctx.clip();
      ctx.fillText(label, 4, ry);
      ctx.restore();
    }

    // Redraw axis on top (sticky top)
    ctx.fillStyle = '#16213e';
    ctx.fillRect(0, 0, canvasW, AXIS_H);
    ctx.strokeStyle = '#0f3460';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, AXIS_H);
    ctx.lineTo(canvasW, AXIS_H);
    ctx.stroke();

    ctx.fillStyle = '#888';
    ctx.font = '10px monospace';
    ctx.textBaseline = 'middle';
    for (let i = 0; i <= this.TICK_N; i++) {
      const xRel = i / this.TICK_N * TW;
      const x = LABEL_W + xRel - scrollLeft;
      if (x < LABEL_W - 2 || x > canvasW + 2) continue;
      const ms = (i / this.TICK_N * D) / 1e6;
      const label = ms.toFixed(2) + ' ms';
      ctx.fillStyle = '#888';
      ctx.fillText(label, x - 16, AXIS_H / 2);
      ctx.strokeStyle = '#0f3460';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, AXIS_H);
      ctx.stroke();
    }

    // Top-left corner fill
    ctx.fillStyle = '#16213e';
    ctx.fillRect(0, 0, LABEL_W, AXIS_H);
    ctx.fillStyle = '#76b7b2';
    ctx.font = 'bold 11px monospace';
    ctx.textBaseline = 'middle';
    ctx.fillText('Time \u2192', 6, AXIS_H / 2);
  }

  _drawRow(ctx, ri, ry, scrollLeft, TW, D, canvasW) {
    const row = this._rows[ri];
    const bars = row.bars;
    if (!bars || bars.length === 0) return;

    const LABEL_W = this.LABEL_W;
    const BAR_H = this.BAR_H;
    const BAR_TOP = this.BAR_TOP;
    const gd_map = this._groupDeps;
    const sr = this._stragglerRatio;

    const visStart = scrollLeft - LABEL_W;
    const visEnd = visStart + canvasW - LABEL_W;
    const startT = visStart / TW * D;
    const endT = visEnd / TW * D;

    // Binary search for first potentially visible bar
    const startIdx = Math.max(0, this._bs(bars, startT));

    for (let bi = startIdx; bi < bars.length; bi++) {
      const b = bars[bi];
      if (b.start > endT) break;

      const bx = LABEL_W + b.start / D * TW - scrollLeft;
      const bw = Math.max(1, (b.end - b.start) / D * TW);
      const by = ry + BAR_TOP;

      const groupKey = b.trace_key || b.name;
      const gd = gd_map[groupKey];

      // Determine draw state
      const isHl = this._hlKey && groupKey === this._hlKey;
      const isPred = this._predKeys.has(groupKey);
      const isSucc = this._succKeys.has(groupKey);
      const isDimmed = this._dimSet.has(groupKey);

      ctx.save();

      // Alpha
      if (isDimmed) {
        ctx.globalAlpha = 0.08;
      } else {
        ctx.globalAlpha = 0.85;
      }

      // Straggler glow
      const isStraggler = gd && gd.ad > 0 && b.avg_dur > gd.ad * sr;
      if (isStraggler && !isDimmed) {
        ctx.shadowColor = 'rgba(233,69,96,.7)';
        ctx.shadowBlur = 6;
      }

      // Bar fill
      ctx.fillStyle = b.color || '#888';
      ctx.beginPath();
      ctx.roundRect ? ctx.roundRect(bx, by, bw, BAR_H, 3) : ctx.rect(bx, by, bw, BAR_H);
      ctx.fill();

      ctx.shadowBlur = 0;

      // Streaming accent: 3px cyan left border
      if (b.execution_kind === 1) {
        ctx.fillStyle = '#00e5ff';
        ctx.globalAlpha = 1.0;
        ctx.fillRect(bx, by, 3, BAR_H);
      }

      ctx.globalAlpha = 1.0;

      // Dependency outlines
      if (isHl) {
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.setLineDash([]);
        ctx.strokeRect(bx, by, bw, BAR_H);

        // Ready-time marker
        if (gd && gd.rt >= 0) {
          const rtX = LABEL_W + gd.rt / D * TW - scrollLeft;
          if (rtX > bx - 1 && rtX < bx + bw) {
            ctx.fillStyle = '#76b7b2';
            ctx.fillRect(rtX, by, 2, BAR_H);
          }
        }
      } else if (isPred) {
        ctx.strokeStyle = '#e94560';
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 3]);
        ctx.strokeRect(bx, by, bw, BAR_H);
        ctx.setLineDash([]);
      } else if (isSucc) {
        ctx.strokeStyle = '#76b7b2';
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 3]);
        ctx.strokeRect(bx, by, bw, BAR_H);
        ctx.setLineDash([]);
      }

      ctx.restore();
    }
  }

  hitTest(canvasX, canvasY) {
    const scrollTop = this._container.scrollTop;
    const scrollLeft = this._container.scrollLeft;
    const AXIS_H = this.AXIS_H;
    const ROW_H = this.ROW_H;
    const LABEL_W = this.LABEL_W;
    const TW = this._timelineW();
    const D = this._totalDur;
    const BAR_TOP = this.BAR_TOP;
    const BAR_H = this.BAR_H;

    if (canvasX < LABEL_W) return null;
    if (canvasY < AXIS_H) return null;

    const ri = Math.floor((canvasY + scrollTop - AXIS_H) / ROW_H);
    if (ri < 0 || ri >= this._rows.length) return null;

    const tAtX = (canvasX - LABEL_W + scrollLeft) / TW * D;
    const bars = this._rows[ri].bars;
    if (!bars) return null;

    const startIdx = Math.max(0, this._bs(bars, tAtX) - 1);
    for (let bi = startIdx; bi < bars.length; bi++) {
      const b = bars[bi];
      if (b.start > tAtX) break;
      if (b.start <= tAtX && b.end >= tAtX) {
        // Also check vertical position within row
        const rowY = AXIS_H + ri * ROW_H - scrollTop;
        const barY = rowY + BAR_TOP;
        if (canvasY >= barY && canvasY <= barY + BAR_H) {
          return b;
        }
      }
    }
    return null;
  }

  showDeps(key) {
    const gd = this._groupDeps[key];
    if (!gd) return;

    const focus = new Set([key]);
    (gd.p || []).forEach(k => focus.add(k));
    (gd.s || []).forEach(k => focus.add(k));

    // Build dim set from all bar keys not in focus
    const dimSet = new Set();
    for (const row of this._rows) {
      for (const b of (row.bars || [])) {
        const k2 = b.trace_key || b.name;
        if (!focus.has(k2)) dimSet.add(k2);
      }
    }

    this._hlKey = key;
    this._predKeys = new Set(gd.p || []);
    this._succKeys = new Set(gd.s || []);
    this._dimSet = dimSet;
    this._sched();
  }

  clearDeps() {
    this._hlKey = null;
    this._predKeys = new Set();
    this._succKeys = new Set();
    this._dimSet = new Set();
    this._sched();
  }

  _onMouseMove(e) {
    const rect = this._canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const bar = this.hitTest(x, y);
    if (bar && window._ttShow) {
      window._ttShow(e, bar, this._groupDeps, this._stragglerRatio);
    } else if (window._ttHide) {
      window._ttHide();
    }
  }

  _onClick(e) {
    const rect = this._canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const bar = this.hitTest(x, y);
    if (bar) {
      const key = bar.trace_key || bar.name;
      if (this._hlKey === key) {
        this.clearDeps();
      } else {
        this.showDeps(key);
      }
    } else {
      this.clearDeps();
    }
  }
}
