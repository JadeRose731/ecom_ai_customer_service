/* admin.js · 后台各页共用导航:mountAdminNav(active)
 * 自带样式注入(挂在样式内联的旧页上也不打架)、包在 IIFE 里(不与宿主页的 $ / el 撞名)。
 * 两层导航:模块行 + 当前模块的子页行。各模块页面保持原路径,这里只收入口。 */
(function () {
  'use strict';

  const MODULES = [
    { key: 'chat', title: '智能客服', pages: [{ href: '/', title: '聊天页' }] },
    { key: 'kb', title: '知识库', pages: [{ href: '/kb', title: '录入页' }, { href: '/admin', title: '后台首页' }] },
    { key: 'eval', title: 'RAG 评估', pages: [{ href: '/rag-eval', title: '评估报告' }] },
    { key: 'flywheel', title: '数据飞轮', pages: [{ href: '/review', title: '飞轮待审' }, { href: '/observability', title: '观测与成本' }] },
    { key: 'topics', title: '主题分布', pages: [{ href: '/topics', title: '主题分布' }, { href: '/topics/questions', title: '类目问题' }] },
  ];

  const STYLE = `
.mew-nav { background: #fff; border: 1px solid #e6e9f0; border-radius: 12px; padding: 8px 12px; display: flex; flex-direction: column; gap: 4px; }
.mew-nav .mods, .mew-nav .subs { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.mew-nav a { text-decoration: none; color: #1f2430; font-size: 13px; padding: 4px 12px; border-radius: 8px; border: 1px solid transparent; }
.mew-nav a:hover { background: #fff1ea; color: #ff7a45; }
.mew-nav a.on { background: #ff7a45; color: #fff; }
.mew-nav .subs a { font-size: 12px; color: #8a93a6; }
.mew-nav .subs a.on { background: #fff1ea; color: #ff7a45; }
.mew-nav .crumb { color: #8a93a6; font-size: 12px; margin: 0 4px; }
`;

  window.mountAdminNav = function (active) {
    if (document.querySelector('.mew-nav')) return;
    const style = document.createElement('style');
    style.textContent = STYLE;
    document.head.appendChild(style);

    const nav = document.createElement('div');
    nav.className = 'mew-nav';
    const mods = document.createElement('div');
    mods.className = 'mods';
    const subs = document.createElement('div');
    subs.className = 'subs';

    let activeMod = null;
    MODULES.forEach((m) => {
      const a = document.createElement('a');
      a.href = m.pages[0].href;
      a.textContent = m.title;
      if (m.key === active) { a.classList.add('on'); activeMod = m; }
      mods.appendChild(a);
    });
    if (activeMod) {
      const crumb = document.createElement('span');
      crumb.className = 'crumb';
      crumb.textContent = '└';
      subs.appendChild(crumb);
      activeMod.pages.forEach((p) => {
        const a = document.createElement('a');
        a.href = p.href;
        a.textContent = p.title;
        const here = location.pathname === p.href || (p.href !== '/' && location.pathname === p.href);
        if (active === 'admin' && p.href === '/admin') a.classList.add('on');
        else if (active === 'kb' && p.href === '/kb') a.classList.add('on');
        else if (here) a.classList.add('on');
        subs.appendChild(a);
      });
    }
    nav.appendChild(mods);
    nav.appendChild(subs);
    const anchor = document.querySelector('[data-admin-nav-anchor]') || document.body;
    anchor.insertBefore(nav, anchor.firstChild);
  };
})();
