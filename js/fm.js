/* FluidMind site runtime — canvas backdrops, scroll reveals, sticky nav, counters.
   Ported from the design handoff (FluidMind Home.dc.html factories, reused verbatim
   with per-page options). Honors prefers-reduced-motion: canvases render one static
   frame and reveals show immediately. */
(function () {
  'use strict';

  var REDUCED = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function onReady(fn) {
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', fn);
    else fn();
  }

  /* ---------- sticky nav ---------- */
  function initNav(opts) {
    opts = opts || {};
    var nav = document.getElementById('nav');
    if (!nav) return;
    var threshold = opts.threshold != null ? opts.threshold : 30;
    var bg = opts.bg || 'rgba(6,13,26,0.72)';
    var padTop = opts.padTop || '20px 5vw';
    var padScrolled = opts.padScrolled || '13px 5vw';
    var border = opts.border || '1px solid rgba(120,180,210,0.1)';
    var onScroll = function () {
      if (window.scrollY > threshold) {
        nav.style.background = bg;
        nav.style.backdropFilter = 'blur(18px)';
        nav.style.webkitBackdropFilter = 'blur(18px)';
        nav.style.padding = padScrolled;
        nav.style.borderBottom = border;
      } else {
        nav.style.background = 'transparent';
        nav.style.backdropFilter = 'none';
        nav.style.padding = padTop;
        nav.style.borderBottom = 'none';
      }
    };
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
  }

  /* ---------- scroll reveals (+ optional count-up stats) ---------- */
  function runCounters(scope) {
    var nums = scope.querySelectorAll('.statnum');
    Array.prototype.forEach.call(nums, function (n) {
      if (n._done) return;
      n._done = true;
      var target = parseInt(n.getAttribute('data-count'), 10) || 0;
      if (REDUCED) { n.textContent = target; return; }
      var dur = 1100;
      var t0 = performance.now();
      var tick = function (now) {
        var p = Math.min(1, (now - t0) / dur);
        var eased = 1 - Math.pow(1 - p, 3);
        n.textContent = Math.round(target * eased);
        if (p < 1) requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    });
  }

  function initReveals(opts) {
    opts = opts || {};
    var els = Array.prototype.slice.call(document.querySelectorAll('[data-reveal]'));
    if (REDUCED || !('IntersectionObserver' in window)) {
      els.forEach(function (el) {
        el.style.opacity = '1';
        el.style.transform = 'none';
        if (el.querySelector && el.querySelector('.statnum')) runCounters(el);
      });
      return;
    }
    var stagger = opts.stagger != null ? opts.stagger : 0.07;
    var ease = 'cubic-bezier(0.2,0.7,0.2,1)';
    els.forEach(function (el) {
      el.style.opacity = '0';
      el.style.transform = 'translateY(28px)';
      el.style.transition = 'opacity 0.8s ' + ease + ', transform 0.8s ' + ease;
    });
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (!e.isIntersecting) return;
        var el = e.target;
        var sibs = Array.prototype.slice.call(
          el.parentElement ? el.parentElement.querySelectorAll(':scope > [data-reveal]') : [el]
        );
        var idx = Math.max(0, sibs.indexOf(el));
        el.style.transitionDelay = (idx * stagger) + 's';
        el.style.opacity = '1';
        el.style.transform = 'none';
        if (el.querySelector && el.querySelector('.statnum')) runCounters(el);
        io.unobserve(el);
      });
    }, { threshold: opts.threshold != null ? opts.threshold : 0.14, rootMargin: opts.rootMargin || '0px 0px -6% 0px' });
    els.forEach(function (el) { io.observe(el); });
  }

  /* ---------- generic canvas mount: DPR + resize + reduced-motion ---------- */
  function startCanvas(id, factory) {
    var c = document.getElementById(id);
    if (!c) return;
    var ctx = c.getContext('2d');
    var DPR = Math.min(1.6, window.devicePixelRatio || 1);
    var state;
    var fit = function () {
      var w = Math.max(1, Math.round(c.clientWidth * DPR));
      var h = Math.max(1, Math.round(c.clientHeight * DPR));
      if (c.width !== w || c.height !== h || !state) {
        c.width = w; c.height = h;
        state = factory(ctx, w, h, DPR);
        if (REDUCED && state && state.frame) {
          for (var i = 0; i < 40; i++) state.frame(); // settle a single static frame
        }
      }
    };
    fit();
    if (!REDUCED) {
      var loop = function () {
        if (state && state.frame) state.frame();
        requestAnimationFrame(loop);
      };
      loop();
    }
    if ('ResizeObserver' in window) {
      var ro = new ResizeObserver(function () { fit(); });
      ro.observe(c);
    }
  }

  /* ---------- fluid: metaball blobs (home + products heroes) ---------- */
  function fluid(opts) {
    opts = opts || {};
    var count = opts.count || 7;
    var alpha = opts.alpha != null ? opts.alpha : 0.40;
    var cols = opts.cols || [[16,210,232],[10,150,205],[46,96,210],[0,200,232],[24,180,220]];
    return function (ctx, w, h) {
      var blobs = [];
      for (var i = 0; i < count; i++) blobs.push({
        bx: Math.random(), by: Math.random(),
        amp: 0.11 + Math.random() * 0.15,
        r: 0.30 + Math.random() * 0.22,
        ph: Math.random() * 7,
        sp: 0.35 + Math.random() * 0.65,
        col: cols[i % cols.length]
      });
      var t = 0;
      return { frame: function () {
        t += 0.006;
        ctx.globalCompositeOperation = 'source-over';
        var bg = ctx.createLinearGradient(0, 0, 0, h);
        bg.addColorStop(0, '#081226'); bg.addColorStop(1, '#04070f');
        ctx.fillStyle = bg; ctx.fillRect(0, 0, w, h);
        ctx.globalCompositeOperation = 'lighter';
        for (var k = 0; k < blobs.length; k++) {
          var b = blobs[k];
          var x = (b.bx + Math.cos(t * b.sp + b.ph) * b.amp) * w;
          var y = (b.by + Math.sin(t * b.sp * 1.25 + b.ph) * b.amp) * h;
          var r = b.r * Math.min(w, h);
          var g = ctx.createRadialGradient(x, y, 0, x, y, r);
          g.addColorStop(0, 'rgba(' + b.col[0] + ',' + b.col[1] + ',' + b.col[2] + ',' + alpha + ')');
          g.addColorStop(1, 'rgba(' + b.col[0] + ',' + b.col[1] + ',' + b.col[2] + ',0)');
          ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
        }
        ctx.globalCompositeOperation = 'source-over';
      } };
    };
  }

  /* ---------- swarm: drifting nodes + labeled agent hubs ---------- */
  function swarm(opts) {
    opts = opts || {};
    var N = opts.count || 70;
    var labels = opts.labels || ['claude', 'senna', 'noto', 'router', 'avogado'];
    var layout = opts.layout || 'scatter'; // 'scatter' (home) | 'row' (maestro)
    var ptLine = opts.ptLine || '34,211,238';       var ptLineA = opts.ptLineA != null ? opts.ptLineA : 0.3;
    var hubLine = opts.hubLine || '124,132,255';    var hubLineA = opts.hubLineA != null ? opts.hubLineA : 0.36;
    var dotColor = opts.dotColor || 'rgba(120,220,240,0.7)';
    var glow = opts.glow || '124,132,255';          var glowA = opts.glowA != null ? opts.glowA : 0.45;
    var hubDot = opts.hubDot || '#c9e9ff';
    var labelColor = opts.labelColor || 'rgba(210,228,255,0.75)';
    var fontSize = opts.fontSize || 12;
    return function (ctx, W, H, DPR) {
      var pts = [];
      for (var i = 0; i < N; i++) pts.push({
        x: Math.random() * W, y: Math.random() * H,
        vx: (Math.random() - 0.5) * 0.32 * DPR, vy: (Math.random() - 0.5) * 0.32 * DPR,
        r: (1.1 + Math.random() * 1.5) * DPR
      });
      var hubs = labels.map(function (label, i) {
        var hx, hy;
        if (layout === 'row') {
          hx = (0.12 + i * 0.19 + Math.random() * 0.06) * W;
          hy = (0.2 + Math.random() * 0.55) * H;
        } else {
          hx = (0.30 + (i % 2 ? 0.34 : 0.08) + Math.random() * 0.1) * W;
          hy = (0.14 + i * 0.17 + (Math.random() - 0.5) * 0.05) * H;
        }
        return { x: hx, y: hy, vx: (Math.random() - 0.5) * 0.12 * DPR, vy: (Math.random() - 0.5) * 0.09 * DPR, label: label, ph: Math.random() * 7 };
      });
      var D = 0.13 * Math.min(W, H) + 55 * DPR;
      var t = 0;
      var wrap = function (p) { if (p.x < 0) p.x = W; if (p.x > W) p.x = 0; if (p.y < 0) p.y = H; if (p.y > H) p.y = 0; };
      return { frame: function () {
        t += 0.016;
        ctx.clearRect(0, 0, W, H);
        pts.forEach(function (p) { p.x += p.vx; p.y += p.vy; wrap(p); });
        hubs.forEach(function (hb) { hb.x += hb.vx; hb.y += hb.vy; wrap(hb); });
        ctx.lineWidth = 1 * DPR;
        for (var i = 0; i < N; i++) for (var j = i + 1; j < N; j++) {
          var d = Math.hypot(pts[i].x - pts[j].x, pts[i].y - pts[j].y);
          if (d < D) {
            ctx.strokeStyle = 'rgba(' + ptLine + ',' + (1 - d / D) * ptLineA + ')';
            ctx.beginPath(); ctx.moveTo(pts[i].x, pts[i].y); ctx.lineTo(pts[j].x, pts[j].y); ctx.stroke();
          }
        }
        hubs.forEach(function (hb) {
          pts.forEach(function (p) {
            var d = Math.hypot(hb.x - p.x, hb.y - p.y);
            if (d < D * 1.1) {
              ctx.strokeStyle = 'rgba(' + hubLine + ',' + (1 - d / (D * 1.1)) * hubLineA + ')';
              ctx.beginPath(); ctx.moveTo(hb.x, hb.y); ctx.lineTo(p.x, p.y); ctx.stroke();
            }
          });
        });
        pts.forEach(function (p) {
          ctx.fillStyle = dotColor;
          ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2); ctx.fill();
        });
        ctx.font = (fontSize * DPR) + "px 'IBM Plex Mono', monospace";
        hubs.forEach(function (hb) {
          var pulse = 1 + Math.sin(t * 1.6 + hb.ph) * 0.25;
          var gr = ctx.createRadialGradient(hb.x, hb.y, 0, hb.x, hb.y, 24 * DPR * pulse);
          gr.addColorStop(0, 'rgba(' + glow + ',' + glowA + ')');
          gr.addColorStop(1, 'rgba(' + glow + ',0)');
          ctx.fillStyle = gr; ctx.beginPath(); ctx.arc(hb.x, hb.y, 24 * DPR * pulse, 0, Math.PI * 2); ctx.fill();
          ctx.fillStyle = hubDot; ctx.beginPath(); ctx.arc(hb.x, hb.y, 3.4 * DPR, 0, Math.PI * 2); ctx.fill();
          ctx.fillStyle = labelColor; ctx.fillText(hb.label, hb.x + 9 * DPR, hb.y - 8 * DPR);
        });
      } };
    };
  }

  /* ---------- flow-field: particles on a noise field, additive blend ---------- */
  function field(opts) {
    opts = opts || {};
    var bg = opts.bg || '#03060d';
    var fade = opts.fade || 'rgba(3,6,13,0.07)';
    var stroke = opts.stroke || 'rgba(34,211,238,0.2)';
    var density = opts.density || 6000;
    var maxP = opts.max || 560;
    var speed = opts.speed || 0.0036;
    return function (ctx, W, H, DPR) {
      ctx.fillStyle = bg; ctx.fillRect(0, 0, W, H);
      var M = Math.min(maxP, Math.floor((W * H) / density));
      var ps = [];
      for (var i = 0; i < M; i++) ps.push({ x: Math.random() * W, y: Math.random() * H });
      var t = 0;
      return { frame: function () {
        t += speed;
        ctx.fillStyle = fade; ctx.fillRect(0, 0, W, H);
        ctx.globalCompositeOperation = 'lighter';
        ctx.lineWidth = 1.1 * DPR; ctx.strokeStyle = stroke;
        for (var k = 0; k < ps.length; k++) {
          var p = ps[k];
          var ang = (Math.sin(p.x * 0.0016 + t) + Math.cos(p.y * 0.0016 - t * 1.2) + Math.sin((p.x + p.y) * 0.001 + t * 0.6)) * 1.5;
          var nx = p.x + Math.cos(ang) * 1.4 * DPR;
          var ny = p.y + Math.sin(ang) * 1.4 * DPR;
          if (nx < 0 || nx > W || ny < 0 || ny > H) { p.x = Math.random() * W; p.y = Math.random() * H; continue; }
          ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(nx, ny); ctx.stroke();
          p.x = nx; p.y = ny;
        }
        ctx.globalCompositeOperation = 'source-over';
      } };
    };
  }

  window.FM = {
    reduced: REDUCED,
    onReady: onReady,
    initNav: initNav,
    initReveals: initReveals,
    runCounters: runCounters,
    startCanvas: startCanvas,
    fluid: fluid,
    swarm: swarm,
    field: field
  };
})();
