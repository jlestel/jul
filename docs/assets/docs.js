// JuL docs: Mac windows, copy buttons, the table of contents that follows, and search.
(() => {
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch {} },
  };

  // ── windows: a Mac title bar (stripes, close box, centred title) on each one ──
  document.querySelectorAll(".win").forEach(win => {
    const body = document.createElement("div");
    body.className = "wbody";
    while (win.firstChild) body.appendChild(win.firstChild);
    const bar = document.createElement("div");
    bar.className = "tbar";
    bar.setAttribute("aria-hidden", "true");
    bar.innerHTML = '<span class="cbox"></span><span class="ttl"></span>';
    bar.querySelector(".ttl").textContent = win.dataset.title || "";
    win.append(bar, body);
    win.classList.add("framed");
  });

  // ── mobile: fold the contents ──
  const toggle = document.querySelector(".nav-toggle");
  toggle && toggle.addEventListener("click", () => {
    const open = toggle.closest(".side").classList.toggle("open");
    toggle.setAttribute("aria-expanded", open);
  });

  // ── copy buttons on code blocks ──
  document.querySelectorAll(".prose pre").forEach(pre => {
    const b = document.createElement("button");
    b.type = "button"; b.className = "copy"; b.textContent = "Copy";
    b.addEventListener("click", async () => {
      const text = pre.querySelector("code") ? pre.querySelector("code").innerText : pre.innerText.replace(/Copy$/, "");
      try { await navigator.clipboard.writeText(text.replace(/\n$/, "")); b.textContent = "Copied ✓"; }
      catch { b.textContent = "⌘C"; }
      setTimeout(() => { b.textContent = "Copy"; }, 1500);
    });
    pre.appendChild(b);
  });

  // ── headings: a § link to share a section ──
  document.querySelectorAll(".prose h2[id], .prose h3[id]").forEach(h => {
    const a = document.createElement("a");
    a.href = "#" + h.id; a.className = "dim"; a.textContent = " §"; a.style.textDecoration = "none";
    a.setAttribute("aria-label", "Link to this section");
    h.appendChild(a);
  });

  // ── the table of contents marks the section being read ──
  const tocLinks = [...document.querySelectorAll(".toc a")];
  if (tocLinks.length && "IntersectionObserver" in window) {
    const heads = tocLinks.map(a => document.getElementById(decodeURIComponent(a.hash.slice(1)))).filter(Boolean);
    const mark = id => tocLinks.forEach(a => a.classList.toggle("on", a.hash === "#" + id));
    const io = new IntersectionObserver(entries => {
      const top = heads.filter(h => h.getBoundingClientRect().top < innerHeight * .35).pop();
      if (top) mark(top.id);
    }, { rootMargin: "0px 0px -60% 0px", threshold: [0, 1] });
    heads.forEach(h => io.observe(h));
  }

  // ── search: a small index built with the site, no service ──
  const q = document.getElementById("q"), box = document.getElementById("results");
  let index = null, sel = -1;
  const load = () => index ? Promise.resolve(index) :
    fetch("assets/search.json").then(r => r.json()).then(d => (index = d));
  const esc = s => s.replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const score = (e, terms) => {
    let s = 0;
    const h = e.h.toLowerCase(), x = e.x.toLowerCase(), t = e.t.toLowerCase();
    for (const w of terms) {
      if (!h.includes(w) && !x.includes(w) && !t.includes(w)) return 0;
      if (h.includes(w)) s += 10;
      if (t.includes(w)) s += 4;
      s += Math.min(5, x.split(w).length - 1);
    }
    return s;
  };
  const snippet = (x, terms) => {
    const i = Math.max(0, x.toLowerCase().indexOf(terms[0]) - 50);
    let s = esc((i ? "…" : "") + x.slice(i, i + 170) + (x.length > i + 170 ? "…" : ""));
    terms.forEach(w => { s = s.replace(new RegExp("(" + w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ")", "ig"), "<mark>$1</mark>"); });
    return s;
  };
  const body = () => box.querySelector(".wbody") || box;
  const show = async () => {
    const v = q.value.trim().toLowerCase();
    if (v.length < 2) { box.hidden = true; return; }
    const terms = v.split(/\s+/).filter(Boolean);
    const hits = (await load()).map(e => [score(e, terms), e]).filter(([s]) => s).sort((a, b) => b[0] - a[0]).slice(0, 12);
    sel = -1;
    body().innerHTML = hits.length
      ? "<ol>" + hits.map(([, e]) => `<li><a href="${e.p}.html${e.a ? "#" + e.a : ""}"><span class="where">${esc(e.t)}${e.h !== e.t ? " › " + esc(e.h) : ""}</span><span class="snip">${snippet(e.x, terms)}</span></a></li>`).join("") + "</ol>"
      : `<p class="dim">No match for "${esc(v)}". Try a command (autotune), a class (Choice) or a variable (JUL_HOME).</p>`;
    box.hidden = false;
  };
  const move = d => {
    const links = [...box.querySelectorAll("a")];
    if (!links.length) return;
    sel = (sel + d + links.length) % links.length;
    links.forEach((a, i) => a.classList.toggle("sel", i === sel));
    links[sel].scrollIntoView({ block: "nearest" });
  };
  q.addEventListener("input", show);
  q.addEventListener("focus", () => { load(); if (q.value.trim().length > 1) show(); });
  q.addEventListener("keydown", e => {
    if (e.key === "ArrowDown") { e.preventDefault(); move(1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); move(-1); }
    else if (e.key === "Enter") {
      const links = box.querySelectorAll("a");
      const a = links[Math.max(0, sel)];
      if (a) location.href = a.href;
    } else if (e.key === "Escape") { box.hidden = true; q.blur(); }
  });
  document.addEventListener("click", e => { if (!e.target.closest(".results, .search")) box.hidden = true; });

  // ── keys: / searches, as everywhere ──
  addEventListener("keydown", e => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.target.matches("input, textarea, select")) return;
    const k = e.key.toLowerCase();
    const go = { d: "index.html", q: "quickstart.html", a: "python-api.html", l: "cli-reference.html" };
    if (e.key === "/") { e.preventDefault(); q.focus(); }
    else if (k === "g") location.href = "https://github.com/usejul/jul";
    else if (go[k]) location.href = go[k];
  });
})();
