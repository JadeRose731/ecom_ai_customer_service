/* acceptance.js · 验收四页公用件:api()/toast()/scoreCell()/jobButton()/jobRow()/missingBox()。
 * 重跑按钮:POST /api/jobs/{name} 发起 → 每 1.2s 轮询 → 终态停轮询并回调页面重取数;
 * heavy 作业先 confirm()。日志尾按作业名缓存在内存里——作业跑完触发页面重取数,重取数
 * 会把作业那一行连 <pre> 一起重建,重建后把缓存贴回去,日志就不会在跑完那一刻清空。 */
(function () {
  'use strict';

  const logCache = {};      // 作业名 → 最近一次看到的日志尾

  function esc(s) {
    return String(s).replace(/[&<>"]/g, (c) =>
      ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  }

  function fmtBytes(n) {
    if (n == null) return '—';
    if (n < 1024) return n + ' B';
    const units = ['KB', 'MB', 'GB'];
    let v = n;
    for (const u of units) {
      v /= 1024;
      if (v < 1024 || u === 'GB') return v.toFixed(1) + ' ' + u;
    }
  }

  async function api(path, opts) {
    const r = await fetch(path, opts);
    if (!r.ok) {
      const body = await r.json().catch(() => ({}));
      throw new Error(body.detail || (r.status + ' ' + r.statusText));
    }
    return r.json();
  }

  function toast(msg, kind) {
    let box = document.getElementById('toast-box');
    if (!box) {
      box = document.createElement('div');
      box.id = 'toast-box';
      document.body.appendChild(box);
    }
    const t = document.createElement('div');
    t.className = 'toast' + (kind ? ' ' + kind : '');
    t.textContent = msg;
    box.appendChild(t);
    setTimeout(() => t.remove(), 2600);
  }

  /* 分数配横条:scoreCell(0.87) → [0.87 ▬▬▬▬▬▬] */
  function scoreCell(v, digits) {
    const div = document.createElement('div');
    div.className = 'sc';
    const b = document.createElement('b');
    b.textContent = Number(v).toFixed(digits == null ? 3 : digits);
    const track = document.createElement('span');
    track.className = 'track';
    const bar = document.createElement('i');
    bar.style.width = Math.max(0, Math.min(1, v)) * 100 + '%';
    track.appendChild(bar);
    div.appendChild(b);
    div.appendChild(track);
    return div;
  }

  function missingBox(hint, title) {
    const div = document.createElement('div');
    div.className = 'missing';
    div.innerHTML = '<b>' + esc(title || '产物还没生成') + '</b>' + esc(hint || '');
    return div;
  }

  const PILL_TEXT = { idle: '没跑过', running: '跑着…', pass: '✅ 过', fail: '🔴 挂了', stopped: '⏹ 已停' };

  function pillEl(status) {
    const sp = document.createElement('span');
    sp.className = 'pill st-' + (status || 'idle');
    sp.textContent = PILL_TEXT[status] || status;
    return sp;
  }

  /* 轮询:1.2s 一次,running 归假即停并回调 onDone(state);网络抖动留给下一轮 */
  function poll(name, pill, pre, onDone) {
    const timer = setInterval(async () => {
      let st;
      try { st = await api('/api/jobs/' + name); } catch (e) { return; }
      const sp = pill;
      if (sp) { sp.className = 'pill st-' + st.status; sp.textContent = PILL_TEXT[st.status] || st.status; }
      logCache[name] = st.log_tail;
      if (pre && st.log_tail) { pre.textContent = st.log_tail; pre.scrollTop = pre.scrollHeight; }
      if (!st.running) {
        clearInterval(timer);
        toast((st.status === 'pass' ? '✅ ' : st.status === 'fail' ? '🔴 ' : '⏹ ') + name + ' · ' + (PILL_TEXT[st.status] || st.status),
              st.status === 'pass' ? 'ok' : st.status === 'fail' ? 'err' : '');
        if (onDone) onDone(st);
      }
    }, 1200);
  }

  /* 作业行:按钮 + 状态药丸 + (可折叠)日志窗口。meta 来自 /api/jobs 或 overview 的 jobs 字段。 */
  function jobRow(name, meta, onDone) {
    meta = meta || {};
    const row = document.createElement('div');
    row.className = 'job';
    const btn = document.createElement('button');
    btn.className = 'jb';
    btn.textContent = '▶ ' + (meta.title || name);
    if (meta.needs) btn.title = meta.needs;
    const pill = pillEl(meta.status || 'idle');
    const pre = document.createElement('pre');
    pre.className = 'jlog';
    pre.textContent = logCache[name] || '';
    if (pre.textContent) pre.classList.add('open');
    btn.addEventListener('click', () => {
      pre.classList.add('open');
      if (btn.disabled) return;
      if (meta.heavy && !confirm('「' + (meta.title || name) + '」是分钟级重活,确认现在跑?')) return;
      btn.disabled = true;
      api('/api/jobs/' + name, { method: 'POST' })
        .then(() => {
          pill.className = 'pill st-running';
          pill.textContent = PILL_TEXT.running;
          pre.textContent = '';
          poll(name, pill, pre, (st) => { btn.disabled = false; if (onDone) onDone(st); });
        })
        .catch((e) => { btn.disabled = false; toast(e.message, 'err'); });
    });
    row.appendChild(btn);
    row.appendChild(pill);
    if (meta.needs) {
      const n = document.createElement('span');
      n.className = 'needs';
      n.textContent = meta.needs;
      row.appendChild(n);
    }
    row.appendChild(pre);
    // 页面在中途刷新/重建时,作业其实还在跑:接着轮,别让药丸停在旧态
    if (meta.status === 'running') {
      btn.disabled = true;
      poll(name, pill, pre, (st) => { btn.disabled = false; if (onDone) onDone(st); });
    }
    return row;
  }

  /* 轻量按钮(卡片里一排多个作业时用,不带日志窗口) */
  function jobButton(name, meta, onDone) {
    return jobRow(name, meta, onDone).querySelector('.jb');
  }

  window.Acceptance = { api, toast, esc, fmtBytes, scoreCell, missingBox, jobRow, jobButton, pillEl };
})();
