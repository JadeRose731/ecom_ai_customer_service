/* acceptance.js · 后台共用工具:fetch 包装 / toast / 作业按钮与日志窗口
 * 作业按钮约定:data-job="作业名" data-heavy="1"(heavy 二次确认);
 * 日志窗口约定:元素带 data-log="作业名",轮询时写入,跑完后从缓存贴回,不许在跑完那一刻消失。 */
(function () {
  'use strict';

  async function api(method, url, body) {
    const opt = { method, headers: { 'Content-Type': 'application/json' } };
    if (body !== undefined) opt.body = JSON.stringify(body);
    const resp = await fetch(url, opt);
    let data = null;
    try { data = await resp.json(); } catch (e) { /* 非 JSON 也当失败处理 */ }
    if (!resp.ok) {
      const detail = data && data.detail !== undefined ? data.detail : (resp.statusText || ('HTTP ' + resp.status));
      throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
    }
    return data;
  }

  function toast(msg, isErr) {
    let box = document.getElementById('toast');
    if (!box) { box = document.createElement('div'); box.id = 'toast'; document.body.appendChild(box); }
    const t = document.createElement('div');
    t.className = 't' + (isErr ? ' err' : '');
    t.textContent = msg;
    box.appendChild(t);
    setTimeout(() => t.remove(), isErr ? 6000 : 3200);
  }

  /* 按作业名缓存日志尾:页面重建(取数回调)时贴回去 */
  const logCache = {};

  function paintLog(name, tail) {
    if (typeof tail === 'string') logCache[name] = tail;
    const win = document.querySelector('[data-log="' + name + '"]');
    if (win) win.textContent = logCache[name] || '';
  }

  function repaintCachedLogs() {
    document.querySelectorAll('[data-log]').forEach((el) => {
      const name = el.getAttribute('data-log');
      if (logCache[name]) el.textContent = logCache[name];
    });
  }

  function setJobButtons(name, running) {
    document.querySelectorAll('[data-job="' + name + '"]').forEach((btn) => {
      btn.disabled = running;
      if (running) btn.dataset.busy = '1'; else delete btn.dataset.busy;
    });
  }

  async function pollJob(name) {
    const st = await api('GET', '/api/jobs/' + name);
    paintLog(name, st.log_tail);
    if (st.running) {
      setTimeout(() => pollJob(name).catch(() => {}), 1200);
    } else {
      setJobButtons(name, false);
      document.dispatchEvent(new CustomEvent('job-done', { detail: { name } }));
    }
    return st;
  }

  function bindJob(root) {
    (root || document).querySelectorAll('button[data-job]').forEach((btn) => {
      if (btn.dataset.bound) return;
      btn.dataset.bound = '1';
      btn.addEventListener('click', async () => {
        const name = btn.getAttribute('data-job');
        const heavy = btn.getAttribute('data-heavy') === '1';
        if (heavy && !window.confirm('「' + name + '」是重活(调 LLM 或清库),确定重跑?')) return;
        try {
          await api('POST', '/api/jobs/' + name);
          setJobButtons(name, true);
          toast('作业 ' + name + ' 已启动');
          pollJob(name).catch(() => {});
        } catch (e) {
          toast('作业 ' + name + ' 启动失败:' + e.message, true);
        }
      });
    });
  }

  window.acceptance = { api, toast, bindJob, paintLog, repaintCachedLogs, logCache };
})();
