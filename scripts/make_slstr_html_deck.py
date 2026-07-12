"""Build a self-contained HTML slide deck for the SLSTR x EarthCARE validation
progress meeting. Figures are downsampled and base64-embedded so the single .html
file is fully self-contained (no external hosts) — publishable as a claude.ai
Artifact and presentable in any browser (arrow-key navigation, light/dark themes).

Run: python scripts/make_slstr_html_deck.py
Out: docs/presentations/slstr_earthcare_deck.html
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
F = ROOT / "figures"
OUT = ROOT / "docs" / "presentations" / "slstr_earthcare_deck.html"

FIGS = {
    "MAP": F / "slstr_collocation" / "collocation_map_polar.png",
    "CTH": F / "slstr_cth_2025-12" / "cth_scatter.png",
    "COT": F / "slstr_cot_water_2025-12" / "cot_water_saturation.png",
    "SURF": F / "slstr_surface_2025-12" / "surface_type_bias.png",
    "CWP": F / "slstr_cwp_2025-12" / "cwp_validation.png",
    "PHASE": F / "slstr_phase_2025-12" / "phase_contingency.png",
}


def datauri(path: Path, max_w: int = 1600) -> str:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.width > max_w:
            h = round(im.height * max_w / im.width)
            im = im.resize((max_w, h), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


CSS = """
<style>
:root{
  --bg:#eaf1f7; --surface:#ffffff; --surface2:#f4f8fc; --plate:#ffffff;
  --ink:#0e1c2b; --muted:#54687a; --line:#d3e0ea;
  --accent:#1567ab; --accent-soft:rgba(21,103,171,.10);
  --good:#1b7837; --warn:#b7791f; --crit:#c0392b;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,"Liberation Mono",monospace;
  --maxw:1080px;
}
@media (prefers-color-scheme:dark){
  :root{
    --bg:#07131f; --surface:#0f2231; --surface2:#0b1a27; --plate:#f6f8fa;
    --ink:#e6eff6; --muted:#8ba0b2; --line:#213b4f;
    --accent:#4aa6e8; --accent-soft:rgba(74,166,232,.14);
    --good:#4bb87a; --warn:#e0b34d; --crit:#e0785f;
  }
}
:root[data-theme="light"]{
  --bg:#eaf1f7; --surface:#ffffff; --surface2:#f4f8fc; --plate:#ffffff;
  --ink:#0e1c2b; --muted:#54687a; --line:#d3e0ea;
  --accent:#1567ab; --accent-soft:rgba(21,103,171,.10);
  --good:#1b7837; --warn:#b7791f; --crit:#c0392b;
}
:root[data-theme="dark"]{
  --bg:#07131f; --surface:#0f2231; --surface2:#0b1a27; --plate:#f6f8fa;
  --ink:#e6eff6; --muted:#8ba0b2; --line:#213b4f;
  --accent:#4aa6e8; --accent-soft:rgba(74,166,232,.14);
  --good:#4bb87a; --warn:#e0b34d; --crit:#e0785f;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth; scroll-snap-type:y proximity}
body{margin:0; background:var(--bg); color:var(--ink); font-family:var(--sans);
  line-height:1.6; -webkit-font-smoothing:antialiased;}
.progress{position:fixed; top:0; left:0; height:3px; width:0%;
  background:var(--accent); z-index:50; transition:width .12s linear}
.counter{position:fixed; right:18px; bottom:16px; z-index:50;
  font-family:var(--mono); font-size:.74rem; letter-spacing:.05em;
  color:var(--muted); background:var(--surface); border:1px solid var(--line);
  padding:5px 10px; border-radius:999px}
.hint{position:fixed; left:18px; bottom:16px; z-index:50; font-family:var(--mono);
  font-size:.68rem; letter-spacing:.06em; color:var(--muted); opacity:.8}
.slide{min-height:100vh; scroll-snap-align:start; display:flex; flex-direction:column;
  justify-content:center; padding:64px 32px; position:relative}
.wrap{width:100%; max-width:var(--maxw); margin:0 auto}
.eyebrow{font-family:var(--mono); font-size:.76rem; letter-spacing:.2em;
  text-transform:uppercase; color:var(--accent); margin:0 0 14px; font-weight:600}
h1{font-size:clamp(2.1rem,5vw,3.5rem); line-height:1.05; letter-spacing:-.02em;
  margin:0 0 18px; text-wrap:balance; font-weight:700}
h2{font-size:clamp(1.5rem,3.4vw,2.3rem); line-height:1.12; letter-spacing:-.01em;
  margin:0 0 20px; text-wrap:balance; font-weight:650}
p.lead{font-size:1.08rem; color:var(--muted); max-width:64ch; margin:0 0 10px}
.plate{background:var(--plate); border:1px solid var(--line); border-radius:12px;
  padding:14px; box-shadow:0 8px 30px rgba(6,20,32,.10)}
.plate img{display:block; width:100%; height:auto; max-height:52vh;
  object-fit:contain; margin:0 auto}
.figrow{display:grid; grid-template-columns:1fr; gap:22px; align-items:center}
.points{list-style:none; margin:16px 0 0; padding:0; display:flex;
  flex-direction:column; gap:11px}
.points li{position:relative; padding-left:22px; font-size:1.02rem; max-width:70ch}
.points li::before{content:"▸"; position:absolute; left:0; top:.05em;
  color:var(--accent); font-family:var(--mono)}
.points li.key{font-weight:650}
.points li.key::before{content:"◆"}
.cap{font-family:var(--mono); font-size:.72rem; color:var(--muted);
  letter-spacing:.03em; margin-top:9px}
.cover{background:
  radial-gradient(120% 90% at 50% -20%, var(--accent-soft), transparent 62%);}
.cover .kicker{font-family:var(--mono); font-size:.8rem; letter-spacing:.22em;
  text-transform:uppercase; color:var(--accent); margin:0 0 20px; font-weight:600}
.rule{height:2px; width:64px; background:var(--accent); margin:26px 0; border:0}
.meta{font-family:var(--mono); font-size:.82rem; color:var(--muted);
  letter-spacing:.02em; line-height:2}
.thesis{display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:26px}
@media (max-width:720px){.thesis{grid-template-columns:1fr}}
.card{background:var(--surface); border:1px solid var(--line); border-radius:10px;
  padding:16px 18px}
.card .t{font-family:var(--mono); font-size:.72rem; letter-spacing:.12em;
  text-transform:uppercase; color:var(--accent); margin-bottom:6px}
.card .d{font-size:.98rem; color:var(--ink)}
table.glance{width:100%; border-collapse:collapse; font-size:.95rem;
  font-variant-numeric:tabular-nums}
table.glance th,table.glance td{text-align:left; padding:11px 12px;
  border-bottom:1px solid var(--line); vertical-align:top}
table.glance thead th{font-family:var(--mono); font-size:.72rem; letter-spacing:.08em;
  text-transform:uppercase; color:#fff; background:var(--accent); border-bottom:0}
table.glance tbody tr:nth-child(even){background:var(--surface2)}
table.glance td.v{font-weight:650; color:var(--ink)}
.badge{display:inline-block; font-family:var(--mono); font-size:.7rem;
  padding:3px 9px; border-radius:999px; letter-spacing:.04em; white-space:nowrap}
.b-good{color:var(--good); background:color-mix(in srgb,var(--good) 14%,transparent)}
.b-warn{color:var(--warn); background:color-mix(in srgb,var(--warn) 16%,transparent)}
.b-crit{color:var(--crit); background:color-mix(in srgb,var(--crit) 16%,transparent)}
.cols{display:grid; grid-template-columns:1fr 1fr; gap:26px}
@media (max-width:760px){.cols{grid-template-columns:1fr}}
.block h3{font-family:var(--mono); font-size:.76rem; letter-spacing:.12em;
  text-transform:uppercase; color:var(--accent); margin:0 0 10px}
.tag{font-family:var(--mono); font-size:.72rem; color:var(--muted)}
.overflow{overflow-x:auto}
html.js .reveal{opacity:0; transform:translateY(16px);
  transition:opacity .6s ease, transform .6s ease}
html.js .slide.in .reveal{opacity:1; transform:none}
html.js .slide.in .reveal:nth-child(2){transition-delay:.08s}
html.js .slide.in .reveal:nth-child(3){transition-delay:.16s}
html.js .slide.in .reveal:nth-child(4){transition-delay:.24s}
:focus-visible{outline:2px solid var(--accent); outline-offset:3px}
@media (prefers-reduced-motion:reduce){
  html{scroll-behavior:auto}
  html.js .reveal{opacity:1; transform:none; transition:none}
}
</style>
"""

BODY = """
<title>SLSTR × EarthCARE validation — Dec 2025</title>
<div class="progress" id="progressbar"></div>
<div class="counter" id="counter">1 / 9</div>
<div class="hint">← → to navigate</div>

<section class="slide cover">
  <div class="wrap">
    <p class="kicker reveal">Progress meeting · December 2025</p>
    <h1 class="reveal">Validating ORAC SLSTR cloud retrievals against EarthCARE</h1>
    <p class="lead reveal">Sentinel-3A × EarthCARE — a polar simultaneous-nadir-overpass
      validation of cloud-top height, optical depth, effective radius, water path,
      cloud mask and phase.</p>
    <hr class="rule reveal"/>
    <div class="thesis reveal">
      <div class="card"><div class="t">The one story</div>
        <div class="d">Two independent limitations of the polar-daytime solar retrieval:
        a bright-surface optical-depth <strong>saturation</strong> and an intrinsic
        <strong>liquid phase bias</strong>. Thermal CTH is strong; CER is robust.</div></div>
      <div class="card"><div class="t">Sample</div>
        <div class="d">Antarctic-summer daytime (solar variables); bi-hemispheric for
        thermal CTH. 0.6 M matched pixels, December 2025.</div></div>
    </div>
    <p class="meta reveal" style="margin-top:24px">
      References&nbsp;·&nbsp;A-CTH&nbsp;/&nbsp;ACM-CAP&nbsp;/&nbsp;A-EBD&nbsp;/&nbsp;A-TC</p>
  </div>
</section>

<section class="slide">
  <div class="wrap">
    <p class="eyebrow reveal">01 / Method</p>
    <h2 class="reveal">A polar comparison — by orbital mechanics, not choice</h2>
    <div class="figrow">
      <div class="plate reveal"><img src="%%MAP%%" alt="Collocation density map, polar stereographic"/>
        <div class="cap">Collocation density — SLSTR × EarthCARE crossings, Dec 2025</div></div>
      <ul class="points reveal">
        <li class="key">Two sun-synchronous orbiters coincide only near the poles (~70–83°) — every match is polar, irreducibly.</li>
        <li>Match rule: temporal gate ±60 min + nearest pixel &lt; 3 km (SLSTR ≈ EarthCARE ≈ 1 km footprint).</li>
        <li>Solar retrievals need daylight → December = Antarctic polar day → the COT/CER/CWP/phase sample is 100% Southern Hemisphere; thermal CTH also samples the Arctic (night).</li>
      </ul>
    </div>
  </div>
</section>

<section class="slide">
  <div class="wrap">
    <p class="eyebrow reveal">02 / Cloud-top height</p>
    <h2 class="reveal">The strong result</h2>
    <div class="figrow">
      <div class="plate reveal"><img src="%%CTH%%" alt="CTH scatter, ORAC vs A-CTH"/>
        <div class="cap">ORAC CTH vs EarthCARE A-CTH</div></div>
      <ul class="points reveal">
        <li class="key">Median CTH bias −0.25 km (Antarctic, polar day) — the thermal cloud-top retrieval survives the polar regime.</li>
        <li>Arctic (polar night) −0.95 km: ~4× larger — a night-vs-day + winter-sea-ice regime contrast, not a pole effect.</li>
        <li>The only variable that samples both hemispheres; the headline −0.57 km blends the two regimes.</li>
      </ul>
    </div>
  </div>
</section>

<section class="slide">
  <div class="wrap">
    <p class="eyebrow reveal">03 / Optical depth</p>
    <h2 class="reveal">The passive retrieval saturates</h2>
    <div class="figrow">
      <div class="plate reveal"><img src="%%COT%%" alt="Water-COT saturation binned by ACM-CAP tau"/>
        <div class="cap">ORAC liquid τ vs ACM-CAP τ (radar-aided synergy)</div></div>
      <ul class="points reveal">
        <li class="key">Report the median: −4.8. The quoted "+3 mean" is a skew artefact of a high-τ tail — it even flips sign.</li>
        <li>ORAC's passive liquid τ is pinned ~5–8 across the entire ACM-CAP range (0.6→34): over-reads thin cloud, saturates on thick, cannot correlate.</li>
        <li>It is ORAC saturating, not the radar reference being high (CPR shifts the bias by ~1 τ).</li>
      </ul>
    </div>
  </div>
</section>

<section class="slide">
  <div class="wrap">
    <p class="eyebrow reveal">04 / Surface control</p>
    <h2 class="reveal">The deficit is cryospheric — the surface, not the phase</h2>
    <div class="figrow">
      <div class="plate reveal"><img src="%%SURF%%" alt="Solar retrieval skill by surface type"/>
        <div class="cap">Median bias &amp; correlation by surface type</div></div>
      <ul class="points reveal">
        <li class="key">Water-COT correlates (r = +0.28) and over-reads only over open water; over sea-ice &amp; ice-sheet (95% of the scene) r → 0 and bias is −4 to −5.</li>
        <li>Same clouds, only the background changed → the τ saturation is the bright-surface radiative regime, not an algorithm bug.</li>
        <li>CER is surface-robust (median +0.2 / +0.3 µm): trustworthy droplet size even where τ is unconstrained.</li>
      </ul>
    </div>
  </div>
</section>

<section class="slide">
  <div class="wrap">
    <p class="eyebrow reveal">05 / Water path</p>
    <h2 class="reveal">The saturation reaches the water budget</h2>
    <div class="figrow">
      <div class="plate reveal"><img src="%%CWP%%" alt="Liquid water path validation"/>
        <div class="cap">ORAC cwp vs ACM-CAP integrated liquid-water-content (LWP)</div></div>
      <ul class="points reveal">
        <li class="key">ORAC liquid water path runs 34% low on the median (30 vs 47 g m⁻²), decorrelated, worse over ice sheet (−18) than ocean (−13).</li>
        <li>Validated against a radar+lidar water-content reference that never sees τ — confirming the saturation propagates into the water budget.</li>
        <li>Fixed by the same surface-albedo advance, not a CWP-specific change.</li>
      </ul>
    </div>
  </div>
</section>

<section class="slide">
  <div class="wrap">
    <p class="eyebrow reveal">06 / Phase &amp; cloud mask</p>
    <h2 class="reveal">Two-way validation against EarthCARE A-TC</h2>
    <div class="figrow">
      <div class="plate reveal"><img src="%%PHASE%%" alt="Cloud mask and phase contingency vs A-TC"/>
        <div class="cap">Confusion matrix · cloud-mask contingency · phase skill by surface</div></div>
      <ul class="points reveal">
        <li class="key">Phase: POD_liquid 89.5%, POD_ice 62.4% → ORAC has a liquid bias, calling 38% of ice cloud tops "liquid".</li>
        <li>Cloud mask: POD 0.69, FAR 0.11 — conservative; the missed 31% is 88% thin cirrus the lidar sees and a passive imager cannot.</li>
        <li>The phase bias is surface-independent (62–64% everywhere) — intrinsic, distinct from the surface-driven τ saturation.</li>
      </ul>
    </div>
  </div>
</section>

<section class="slide">
  <div class="wrap">
    <p class="eyebrow reveal">07 / At a glance</p>
    <h2 class="reveal">All variables — December 2025</h2>
    <div class="overflow reveal">
    <table class="glance">
      <thead><tr><th>Variable</th><th>EarthCARE ref.</th><th>Headline metric (median-primary)</th><th>Verdict</th></tr></thead>
      <tbody>
        <tr><td class="v">Cloud-top height</td><td class="tag">A-CTH</td><td>−0.25 km (S, day) · −0.95 km (N, night)</td><td><span class="badge b-good">Strong</span></td></tr>
        <tr><td class="v">Water-cloud COT</td><td class="tag">ACM-CAP</td><td>median −4.8 (passive τ saturates ~5–8)</td><td><span class="badge b-warn">Regime-limited</span></td></tr>
        <tr><td class="v">Ice-cloud COT</td><td class="tag">A-EBD</td><td>median +2.0 (mean +7, skewed)</td><td><span class="badge b-warn">Weak corr.</span></td></tr>
        <tr><td class="v">Effective radius</td><td class="tag">ACM-CAP</td><td>median +1 µm (robust bias, no skill)</td><td><span class="badge b-good">Robust bias</span></td></tr>
        <tr><td class="v">Liquid water path</td><td class="tag">ACM-CAP LWP</td><td>median −16 g m⁻² (−34%)</td><td><span class="badge b-warn">Inherits τ</span></td></tr>
        <tr><td class="v">Cloud mask</td><td class="tag">A-TC</td><td>POD 0.69 · FAR 0.11 (misses = thin cirrus)</td><td><span class="badge b-good">Conservative</span></td></tr>
        <tr><td class="v">Cloud phase</td><td class="tag">A-TC</td><td>POD_liq 90% · POD_ice 62% (liquid bias)</td><td><span class="badge b-crit">Liquid-biased</span></td></tr>
      </tbody>
    </table>
    </div>
  </div>
</section>

<section class="slide">
  <div class="wrap">
    <p class="eyebrow reveal">08 / Summary</p>
    <h2 class="reveal">What's validated, and what's next</h2>
    <div class="cols reveal">
      <div class="block">
        <h3>Validated — median-primary, with CIs</h3>
        <ul class="points">
          <li>CTH · water-COT · ice-COT · CER · CWP · cloud mask · phase</li>
          <li>Plus collocation methodology, surface-type &amp; phase stratification, and sensitivity to Δt / distance.</li>
        </ul>
        <h3 style="margin-top:22px">The unifying result</h3>
        <ul class="points">
          <li class="key">Bright-surface τ <strong>saturation</strong> — surface-driven, propagates into CWP, spares open water.</li>
          <li class="key">Intrinsic liquid <strong>phase bias</strong> — POD_ice 62%, surface-independent.</li>
          <li>Thermal CTH strong; CER robust — the trustworthy products over the cryosphere.</li>
        </ul>
      </div>
      <div class="block">
        <h3>Next steps — need new processing</h3>
        <ul class="points">
          <li>Arctic / boreal-summer month — run COT/CER/CWP/phase over Arctic sea-ice &amp; Greenland.</li>
          <li>Sentinel-3B (SLSTR-B) — second platform, cross-check.</li>
          <li>Other seasons — the seasonal cycle.</li>
        </ul>
        <h3 style="margin-top:22px">Data ceilings</h3>
        <ul class="points">
          <li class="tag" style="padding-left:22px">Low/mid latitudes are impossible with this orbital pairing — polar by construction.</li>
        </ul>
      </div>
    </div>
  </div>
</section>

<script>
(function(){
  var d=document.documentElement; d.classList.add('js');
  var slides=[].slice.call(document.querySelectorAll('.slide'));
  var counter=document.getElementById('counter'), bar=document.getElementById('progressbar');
  var reduce=window.matchMedia('(prefers-reduced-motion:reduce)').matches;
  function cur(){var y=window.scrollY+window.innerHeight*0.4,i=0;
    for(var k=0;k<slides.length;k++){if(slides[k].offsetTop<=y)i=k;}return i;}
  function upd(){var i=cur();if(counter)counter.textContent=(i+1)+' / '+slides.length;
    var m=d.scrollHeight-window.innerHeight;bar.style.width=(m>0?window.scrollY/m*100:0).toFixed(1)+'%';}
  function go(n){var i=Math.max(0,Math.min(slides.length-1,cur()+n));
    slides[i].scrollIntoView({behavior:reduce?'auto':'smooth',block:'start'});}
  window.addEventListener('scroll',upd,{passive:true});
  window.addEventListener('resize',upd);
  document.addEventListener('keydown',function(e){
    if(['ArrowDown','ArrowRight','PageDown',' '].indexOf(e.key)>=0){e.preventDefault();go(1);}
    else if(['ArrowUp','ArrowLeft','PageUp'].indexOf(e.key)>=0){e.preventDefault();go(-1);}
    else if(e.key==='Home'){e.preventDefault();slides[0].scrollIntoView({behavior:reduce?'auto':'smooth'});}
    else if(e.key==='End'){e.preventDefault();slides[slides.length-1].scrollIntoView({behavior:reduce?'auto':'smooth'});}
  });
  if(!reduce&&'IntersectionObserver'in window){
    var io=new IntersectionObserver(function(es){es.forEach(function(en){
      if(en.isIntersecting)en.target.classList.add('in');});},{threshold:0.15});
    slides.forEach(function(s){io.observe(s);});
  } else {slides.forEach(function(s){s.classList.add('in');});}
  upd();
})();
</script>
"""


def main() -> int:
    html = CSS + BODY
    for key, path in FIGS.items():
        html = html.replace(f"%%{key}%%", datauri(path))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT} ({kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
