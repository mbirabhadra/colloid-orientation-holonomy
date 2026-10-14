"""
Reproduces every number and figure in Paper A.
Run: python run_paperA.py   (about 10-20 min; needs numpy, matplotlib)
Outputs: ../figs/*.pdf and results_paperA.txt
"""
import numpy as np, json, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import holo

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, '..', 'figs'); os.makedirs(FIG, exist_ok=True)
OUT = open(os.path.join(HERE, 'results_paperA.txt'), 'w')
def log(*a):
    s = ' '.join(str(x) for x in a); print(s); OUT.write(s + '\n'); OUT.flush()

C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'
plt.rcParams.update({'font.size': 9, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
                     'ytick.major.width': 0.6, 'lines.linewidth': 1.5, 'font.family': 'serif',
                     'mathtext.fontset': 'cm', 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
rng = np.random.default_rng(20261011)
D05, Dr05 = holo.bulk_SE(0.5)
log(f"Bulk Stokes-Einstein, a = 0.5 um, water 25C: D = {D05:.4f} um^2/s, D_r = {Dr05:.4f} rad^2/s")

# ---------------------------------------------------------------- 1. exactness & bias vs K D T
log("\n[1] Locked orientation (alpha=1), no rotational noise: Gauss-Bonnet vs Levy-area estimator")
log("   K D T    kappa_GB (theta vs Omega)    K_Levy/K")
kdts, kGB, kLV, eGB, eLV = [], [], [], [], []
for kdt in [0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]:
    R, D = 1.0, 0.5; T = kdt/(D/R**2)
    s = holo.simulate(R, D, T, fps=200/T, M=3000, D_rn=0.0, sub=5, rng=rng)
    th = holo.chord_angles(s['P'], s['Mk']); Om = holo.spherical_area(s['P']); A = holo.levy_area(s['P'], R)
    b, e = holo.fit(Om, th); bA, eA = holo.fit(A, th)
    kdts.append(kdt); kGB.append(b); eGB.append(e); kLV.append(bA*R**2); eLV.append(eA*R**2)
    log(f"   {kdt:5.2f}    {b:.4f} +- {e:.4f}            {bA*R**2:.4f} +- {eA*R**2:.4f}")
fig, ax = plt.subplots(figsize=(3.4, 2.4))
ax.axhline(1, color='0.6', lw=0.6)
ax.errorbar(kdts, kGB, yerr=eGB, color=C1, marker='o', ms=4, label=r'Gauss–Bonnet: $\theta$ vs $\Omega$')
ax.errorbar(kdts, kLV, yerr=eLV, color=C2, marker='s', ms=4, ls='--', label=r'Lévy area: $\theta$ vs $KA$')
ax.set_xscale('log'); ax.set_xlabel(r'$K D T$'); ax.set_ylabel(r'fitted $\kappa$')
ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_bias.pdf')); plt.close(fig)

# ---------------------------------------------------------------- 2. scatter illustration
log("\n[2] Scatter illustration, R = 3 um, a = 0.5 um bulk values, T = 2 s")
s = holo.simulate(3.0, D05, 2.0, 25, 3000, D_rn=0.0, rng=rng)
th0 = holo.chord_angles(s['P'], s['Mk']); Om0 = holo.spherical_area(s['P'])
s = holo.simulate(3.0, D05, 2.0, 25, 3000, D_rn=Dr05, rng=rng)
th1 = holo.chord_angles(s['P'], s['Mk']); Om1 = holo.spherical_area(s['P'])
log(f"   no spin noise: kappa = {holo.fit(Om0, th0)[0]:.4f}, corr = {np.corrcoef(Om0, th0)[0,1]:.4f}")
log(f"   with D_r     : kappa = {holo.fit(Om1, th1)[0]:.3f} +- {holo.fit(Om1, th1)[1]:.3f}, corr = {np.corrcoef(Om1, th1)[0,1]:.4f}")
fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.5), sharex=True)
for a_, th, ttl in [(axs[0], th0, r'(a) $D_r=0$'), (axs[1], th1, r'(b) $D_r=1.47\,$rad$^2$/s')]:
    Om = Om0 if a_ is axs[0] else Om1
    a_.scatter(Om, th, s=2, color=C1, alpha=0.35, linewidths=0)
    xx = np.linspace(Om.min(), Om.max(), 2); a_.plot(xx, xx, color=C2, lw=1.2)
    a_.set_xlabel(r'enclosed solid angle $\Omega$ (sr)'); a_.set_title(ttl, fontsize=9, loc='left')
axs[0].set_ylabel(r'chord-closed angle $\theta$ (rad)')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_scatter.pdf')); plt.close(fig)

# ---------------------------------------------------------------- 3. precision vs segment length at fixed total time
log("\n[3] Precision at fixed total tracking time: delta kappa vs segment length T (R = 3 um, bulk a = 0.5 um)")
R = 3.0; K = 1/R**2; tau_tot = 20000.0       # seconds of total tracking per point
pred = np.sqrt(2*Dr05/(tau_tot))/(K*D05)
log(f"   tau_total = {tau_tot:.0f} s; prediction (sigma_theta=0): delta kappa = sqrt(2 D_r/tau)/(K D) = {pred:.4f}")
Ts, dks, kks = [0.4, 1, 2, 4, 8], [], []
for T in Ts:
    N = int(tau_tot/T)
    s = holo.simulate(R, D05, T, 25, N, D_rn=Dr05, sig_x=0.02, sig_th=0.02, rng=rng)
    th = holo.chord_angles(s['P'], s['Mk']); Om = holo.spherical_area(s['P'])
    k, dk = holo.fit(Om, th); dks.append(dk); kks.append(k)
    log(f"   T = {T:4.1f} s, N = {N:6d}: kappa = {k:.3f} +- {dk:.3f}")
fig, ax = plt.subplots(figsize=(3.4, 2.4))
ax.plot(Ts, dks, color=C1, marker='o', ms=4, label='simulation')
ax.axhline(pred, color=C2, ls='--', lw=1.2, label=r'$\sqrt{2D_r/\tau}/(KD)$')
ax.set_xscale('log'); ax.set_ylim(0, 2*pred); ax.set_xlabel(r'segment length $T$ (s)')
ax.set_xticks(Ts); ax.set_xticklabels([str(t) for t in Ts]); ax.minorticks_off()
ax.set_ylabel(r'$\delta\kappa$ at $\tau=2\times10^4$ s'); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_precision.pdf')); plt.close(fig)

# ---------------------------------------------------------------- 4. partial co-rotation diagnostic
log("\n[4] Partial co-rotation alpha (no spin noise), R = 3 um, T = 1 s (K D T = 0.055): kappa vs alpha(2-alpha)")
als, kal, eal = [0.0, 0.25, 0.5, 0.75, 1.0], [], []
for al in als:
    s = holo.simulate(3.0, D05, 1.0, 25, 6000, D_rn=0.0, alpha=al, rng=rng)
    th = holo.chord_angles(s['P'], s['Mk']); Om = holo.spherical_area(s['P'])
    k, dk = holo.fit(Om, th); kal.append(k); eal.append(dk)
    log(f"   alpha = {al:.2f}: kappa = {k:.4f} +- {dk:.4f}   alpha(2-alpha) = {al*(2-al):.4f}")
log("   larger K D T (T = 8 s, K D T = 0.44): higher-order terms")
for al in [0.0, 0.5]:
    s = holo.simulate(3.0, D05, 8.0, 25, 4000, D_rn=0.0, alpha=al, rng=rng)
    th = holo.chord_angles(s['P'], s['Mk']); Om = holo.spherical_area(s['P'])
    k, dk = holo.fit(Om, th)
    log(f"   alpha = {al:.2f}: kappa = {k:.4f} +- {dk:.4f}   alpha(2-alpha) = {al*(2-al):.4f}")
fig, ax = plt.subplots(figsize=(3.4, 2.4))
aa = np.linspace(0, 1, 100); ax.plot(aa, aa*(2-aa), color=C2, ls='--', lw=1.2, label=r'leading order $\alpha(2-\alpha)$')
ax.errorbar(als, kal, yerr=eal, color=C1, marker='o', ms=4, ls='none', label='simulation')
ax.set_xlabel(r'co-rotation fraction $\alpha$'); ax.set_ylabel(r'fitted $\kappa$'); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_alpha.pdf')); plt.close(fig)

# ---------------------------------------------------------------- 5. realistic dry runs and controls
log("\n[5] Realistic dry runs (bulk SE, 25 fps, sigma_x = 20 nm, sigma_theta = 0.02 rad, T = 2 s, N = 20000)")
log("   case                          kappa_hat +- se       truth")
rows = []
def dry(label, R, N=20000, c=0.0, shuffle=False, truth=None):
    s = holo.simulate(R, D05, 2.0, 25, N, D_rn=Dr05, c_spin=c, sig_x=0.02, sig_th=0.02, rng=rng)
    th = holo.chord_angles(s['P'], s['Mk'])
    if R > 1e3:   # flat control: regress on K_ref * Area with K_ref of the 3 um droplet, i.e. report K_hat/K_ref
        A = holo.levy_area(s['P'], R); x = A/9.0
    else:
        x = holo.spherical_area(s['P'])
    if shuffle: th = rng.permutation(th)
    k, dk = holo.fit(x, th)
    log(f"   {label:30s}{k:7.3f} +- {dk:.3f}      {truth}")
    rows.append((label, k, dk, truth))
dry('R = 3 um', 3.0, truth=1)
dry('R = 3 um, N = 5000', 3.0, N=5000, truth=1)
dry('R = 5 um', 5.0, truth=1)
dry('R = 3 um, coupling c = +0.3 K', 3.0, c=0.3/9, truth=1.3)
dry('flat (R=1e4), scaled to K=1/9', 1e4, truth=0)
dry('R = 3 um, shuffled pairing', 3.0, shuffle=True, truth=0)

# ---------------------------------------------------------------- 6. required tracking time table
log("\n[6] Total tracking time for delta kappa = 0.2 (5-sigma detection) and 0.05, sigma_theta -> 0")
log("   R(um)  a(um)   D(um2/s)  D_r(rad2/s)   tau_0.2 (h)   tau_0.05 (h)")
for R_, a_ in [(1.5, 0.25), (2.0, 0.25), (3.0, 0.5), (3.0, 0.25), (5.0, 0.5), (5.0, 1.0)]:
    D_, Dr_ = holo.bulk_SE(a_); K_ = 1/R_**2
    t1 = 2*Dr_/(K_*D_*0.2)**2/3600; t2 = 2*Dr_/(K_*D_*0.05)**2/3600
    log(f"   {R_:4.1f}   {a_:4.2f}   {D_:7.3f}    {Dr_:8.2f}       {t1:8.2f}      {t2:9.1f}")
log("\nClosed form (Stokes-Einstein): tau = 2 D_r/(K D dk)^2 = 9 pi eta R^4/(a k_B T dk^2)")
OUT.close()
