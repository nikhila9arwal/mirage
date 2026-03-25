// Globals set by shell.html: TAB_DATA, GBL (totalDur, groupDeps, stragglerRatio, legend)
const _renderers = {};
let _activeRid = null;

function _initTab(id) {
  if (_renderers[id]) return _renderers[id];
  const tc = document.getElementById('tc-' + id);
  if (!tc || !TAB_DATA[id]) return null;
  const r = new TimelineRenderer(tc, TAB_DATA[id], GBL.totalDur, GBL.groupDeps, GBL.stragglerRatio);
  _renderers[id] = r;
  return r;
}

function sw(id, btn) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.ctrls .tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + id).classList.add('active');
  btn.classList.add('active');
  _activeRid = id;
  _initTab(id);
}

function zoom(dir) {
  const r = _renderers[_activeRid]; if (!r) return;
  if (dir === 0) r.setZoom(1.0);
  else if (dir > 0) r.setZoom(Math.min(20, r._zoom * 1.5));
  else r.setZoom(Math.max(0.167, r._zoom / 1.5));
  document.getElementById('zl').textContent = Math.round(r._zoom * 100) + '%';
}

// Tooltip
const _tip = document.getElementById('tt-tip');
_tip.style.cssText = 'display:none;position:fixed;background:#0f3460;color:#e0e0e0;padding:8px 12px;border-radius:6px;font-size:12px;font-family:monospace;z-index:1000;pointer-events:none;white-space:pre-line;box-shadow:0 4px 12px rgba(0,0,0,.5);max-width:400px';

window._ttShow = function(e, b, gd_map, sratio) {
  if (!b) { _tip.style.display='none'; return; }
  const gd = gd_map[b.trace_key || b.name];
  let txt = '<b>' + b.name + '</b>';
  txt += '\nWall: ' + ((b.end-b.start)/1e3).toFixed(1) + ' us  Avg/blk: ' + (b.avg_dur/1e3).toFixed(1) + ' us  Blocks: ' + b.blocks;
  if (gd) {
    const qw = (b.start - gd.rt) / 1e3;
    if (qw > 0) txt += '\nQueue wait: ' + qw.toFixed(1) + ' us';
    if (b.avg_dur > gd.ad * sratio && gd.ad > 0) txt += '\n\u26a0 STRAGGLER: ' + (b.avg_dur/gd.ad).toFixed(1) + 'x avg';
    txt += '\n\nClick to highlight deps  \u2022  Preds: ' + gd.p.length + '  Succs: ' + gd.s.length;
  }
  _tip.innerHTML = txt;
  _tip.style.display = 'block';
  _tip.style.left = (e.clientX+12)+'px';
  _tip.style.top = (e.clientY-30)+'px';
};
window._ttHide = function() { _tip.style.display = 'none'; };

// Legend
const _leg = document.getElementById('leg');
GBL.legend.forEach(i => {
  const d = document.createElement('div'); d.className='li';
  d.innerHTML = '<div class="lc" style="background:'+i.color+'"></div>'+i.name;
  _leg.appendChild(d);
});

// showGroupFromWait (called from schedule analysis section links)
function showGroupFromWait(tk) {
  if (!tk) return;
  const blkBtn = document.querySelector('.ctrls .tab-btn[data-tab="blk"]');
  if (blkBtn && _activeRid !== 'blk') blkBtn.click();
  setTimeout(() => { const r = _renderers['blk']; if (r) r.showDeps(tk); }, 80);
}

// Initial tab
const _firstBtn = document.querySelector('.ctrls .tab-btn');
if (_firstBtn) _firstBtn.click();
