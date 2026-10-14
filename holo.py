"""
Simulation library for Paper A: orientation holonomy of Brownian colloids on spherical droplets.

Units: micrometres, seconds, radians.
State per particle: position p (unit vector on sphere of radius R) and a body frame B (3x3, columns
are body axes in the lab).  A translational geodesic step of angle phi in tangent direction u moves p
along the great circle; the body is rotated about the axis n x u by alpha*phi.
  alpha = 1 : body rolls with the interface normal ("slaved"); tangent body vectors are then
              Levi-Civita parallel transported exactly (this rotation IS parallel transport
              along the great-circle step).
  alpha = 0 : body does not rotate with the motion (free, unconstrained orientation).
Rotational noise: about the local normal with diffusion D_rn, and (optionally) about the two
tangential axes with D_rt.  An optional deterministic spin-area coupling c adds a rotation about
the normal equal to c * (signed spherical area swept in the step, as seen from the segment start),
used to show the estimator recovers kappa = 1 + c/K.
"""
import numpy as np


def rot(axis, ang):
    """Rodrigues rotation matrices, axis (M,3) unit, ang (M,) -> (M,3,3)."""
    M = axis.shape[0]
    K = np.zeros((M, 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -axis[:, 2], axis[:, 1]
    K[:, 1, 0], K[:, 1, 2] = axis[:, 2], -axis[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -axis[:, 1], axis[:, 0]
    s, c = np.sin(ang)[:, None, None], np.cos(ang)[:, None, None]
    I = np.eye(3)[None]
    return I + s*K + (1 - c)*(K @ K)


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return v/n


def tri_solid_angle(a, b, c):
    """Signed solid angle of spherical triangle (a,b,c), Van Oosterom-Strackee."""
    num = np.einsum('ij,ij->i', a, np.cross(b, c))
    den = 1 + np.einsum('ij,ij->i', a, b) + np.einsum('ij,ij->i', b, c) + np.einsum('ij,ij->i', c, a)
    return 2*np.arctan2(num, den)


def tangent_basis(p):
    """Smooth-enough tangent basis near the north pole start; used only to draw isotropic steps."""
    ref = np.where(np.abs(p[:, 2:3]) < 0.9, np.array([[0, 0, 1.0]]), np.array([[1.0, 0, 0]]))
    t1 = unit(np.cross(ref, p))
    t2 = np.cross(p, t1)
    return t1, t2


def simulate(R, D, T, fps, M, D_rn, D_rt=0.0, alpha=1.0, c_spin=0.0, sub=10,
             sig_x=0.0, sig_th=0.0, rng=None, start_random=True):
    """Return dict with measured (noisy) positions P[f] (M,3 unit) and markers Mk[f] (M,3)
    for f = 0..nfr, plus the true trajectories."""
    rng = np.random.default_rng() if rng is None else rng
    nfr = int(round(T*fps)); dt = 1.0/(fps*sub)
    if start_random:
        p = unit(rng.standard_normal((M, 3)))
    else:
        p = np.tile([0., 0., 1.], (M, 1))
    t1, t2 = tangent_basis(p)
    # body frame: column 0 = marker (tangent at start), column 2 = normal at start
    B = np.stack([t1, t2, p], axis=2)
    p_start = p.copy()
    P_true, M_true = [p.copy()], [B[:, :, 0].copy()]
    for f in range(nfr):
        for _ in range(sub):
            a1, a2 = tangent_basis(p)
            xi = rng.standard_normal((M, 2))*np.sqrt(2*D*dt)/R      # angles on unit sphere
            v = xi[:, :1]*a1 + xi[:, 1:]*a2
            phi = np.linalg.norm(v, axis=1); u = unit(v)
            p_new = np.cos(phi)[:, None]*p + np.sin(phi)[:, None]*u
            ax = unit(np.cross(p, u))
            Rm = rot(ax, alpha*phi)
            B = Rm @ B
            # spin about the new normal: noise + optional spin-area coupling
            spin = rng.standard_normal(M)*np.sqrt(2*D_rn*dt)
            if c_spin != 0.0:
                dOmega = tri_solid_angle(p_start, p, p_new)         # solid angle swept (unit sphere)
                spin = spin + (c_spin*R**2)*dOmega                  # c has units of curvature
            B = rot(p_new, spin) @ B
            if D_rt > 0:
                b1, b2 = tangent_basis(p_new)
                B = rot(b1, rng.standard_normal(M)*np.sqrt(2*D_rt*dt)) @ B
                B = rot(b2, rng.standard_normal(M)*np.sqrt(2*D_rt*dt)) @ B
            p = p_new
        P_true.append(p.copy()); M_true.append(B[:, :, 0].copy())
    # measurement noise
    P_meas, M_meas = [], []
    for pp, mm in zip(P_true, M_true):
        q = unit(pp + rng.standard_normal(pp.shape)*sig_x/R)
        mt = unit(mm - np.einsum('ij,ij->i', mm, q)[:, None]*q)
        mt = (rot(q, rng.standard_normal(M)*sig_th) @ mt[:, :, None])[:, :, 0]
        P_meas.append(q); M_meas.append(mt)
    return dict(P=np.array(P_meas), Mk=np.array(M_meas), Pt=np.array(P_true), Mt=np.array(M_true), R=R)


def chord_angles(P, Mk):
    """Chord-closed in-plane orientation angle at every frame, unwrapped; returns theta(T)-theta(0)."""
    p0, m0 = P[0], Mk[0]
    m0 = unit(m0 - np.einsum('ij,ij->i', m0, p0)[:, None]*p0)
    g0 = np.cross(p0, m0)
    angs = []
    for q, m in zip(P, Mk):
        mt = unit(m - np.einsum('ij,ij->i', m, q)[:, None]*q)
        c = np.clip(np.einsum('ij,ij->i', q, p0), -1, 1)
        ax = np.cross(q, p0); s = np.linalg.norm(ax, axis=1)
        ax = unit(ax); ang = np.arctan2(s, c)
        mb = (rot(ax, ang) @ mt[:, :, None])[:, :, 0]        # transport back along the chord
        angs.append(np.arctan2(np.einsum('ij,ij->i', mb, g0), np.einsum('ij,ij->i', mb, m0)))
    angs = np.unwrap(np.array(angs), axis=0)
    return angs[-1] - angs[0]


def spherical_area(P):
    """Signed solid angle (unit sphere) enclosed by the sampled path closed with its chord."""
    p0 = P[0]; Om = np.zeros(P.shape[1])
    for i in range(1, P.shape[0] - 1):
        Om += tri_solid_angle(p0, P[i], P[i + 1])
    return Om


def levy_area(P, R):
    """Levy area in azimuthal-equidistant (normal) coordinates at the start point, in um^2."""
    p0 = P[0]
    t1, t2 = tangent_basis(p0)
    def nc(q):
        c = np.clip(np.einsum('ij,ij->i', q, p0), -1, 1); ang = np.arccos(c)
        d = unit(q - c[:, None]*p0)
        return R*ang*np.einsum('ij,ij->i', d, t1), R*ang*np.einsum('ij,ij->i', d, t2)
    x, y = nc(P[0]); A = np.zeros(P.shape[1])
    for q in P[1:]:
        xn, yn = nc(q); A += 0.5*(x*yn - xn*y); x, y = xn, yn
    return A


def fit(x, y):
    """Least squares through the origin, y = b x. Returns b and its heteroscedasticity-robust
    (sandwich, HC0) standard error sqrt(sum x^2 r^2)/sum x^2."""
    s = np.sum(x*x); b = np.sum(x*y)/s
    r = y - b*x
    return b, np.sqrt(np.sum(x*x*r*r))/s


def bulk_SE(a_um, eta=0.890e-3, kT=4.116e-21):   # water at 25 C
    a = a_um*1e-6
    return kT/(6*np.pi*eta*a)*1e12, kT/(8*np.pi*eta*a**3)
