"""GPU local-inertial shallow-water solver (Bates et al. 2010; de Almeida et al. 2012).

A heightfield 2D solver of the same class as Fluid Flux: water depth at cell centres,
unit discharge on cell faces, explicit time stepping, buildings represented as raised
blocks in the heightfield. Used here only to measure how results change with cell size
when every other input is held fixed.

References
- Bates, P. D., Horritt, M. S., & Fewtrell, T. J. (2010). A simple inertial formulation of the
  shallow water equations for efficient two-dimensional flood inundation modelling.
  Journal of Hydrology, 387(1-2), 33-45.
- de Almeida, G. A. M., Bates, P., Freer, J. E., & Souvignet, M. (2012). Improving the stability
  of a simple formulation of the shallow water equations for 2-D flood modeling.
  Water Resources Research, 48(5).
"""
import time
from dataclasses import dataclass, field

import cupy as cp
import numpy as np

G = 9.81
H_MIN = 1e-3  # depth below which a face carries no flow (m)
FR_MAX = 1.0  # Froude cap on face discharge

_flux_x = cp.RawKernel(r"""
extern "C" __global__
void flux_x(const float* z, const float* h, const float* n, float* qx, float* qx_new,
            int ny, int nx, float dt, float dx, float theta, float s_out, int closed_w, int closed_e) {
    // qx has shape (ny, nx+1); face i sits between cell i-1 and cell i
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > nx || j >= ny) return;
    int f = j * (nx + 1) + i;
    if (i == 0 || i == nx) {
        // free outflow at normal depth with prescribed energy slope
        int c = j * nx + (i == 0 ? 0 : nx - 1);
        float hc = h[c];
        float q = 0.f;
        if (hc > H_MIN_ && !((i == 0 && closed_w) || (i == nx && closed_e)))
            q = powf(hc, 5.f / 3.f) * sqrtf(s_out) / n[c];
        qx_new[f] = (i == 0) ? -q : q;
        return;
    }
    int l = j * nx + i - 1, r = l + 1;
    float etl = z[l] + h[l], etr = z[r] + h[r];
    float hf = fmaxf(etl, etr) - fmaxf(z[l], z[r]);
    if (hf <= H_MIN_) { qx_new[f] = 0.f; return; }
    float q0 = qx[f];
    float qm = qx[f - 1], qp = qx[f + 1];
    float qa = theta * q0 + 0.5f * (1.f - theta) * (qm + qp);
    float nf = 0.5f * (n[l] + n[r]);
    float num = qa - G_ * hf * dt * (etr - etl) / dx;
    float den = 1.f + G_ * dt * nf * nf * fabsf(q0) / powf(hf, 7.f / 3.f);
    float qn = num / den;
    float qcap = FR_MAX_ * hf * sqrtf(G_ * hf);  // local-inertial form is only valid for subcritical flow
    qx_new[f] = fmaxf(-qcap, fminf(qcap, qn));
}
""".replace("H_MIN_", f"{H_MIN}f").replace("FR_MAX_", f"{FR_MAX}f").replace("G_", f"{G}f"), "flux_x")

_flux_y = cp.RawKernel(r"""
extern "C" __global__
void flux_y(const float* z, const float* h, const float* n, float* qy, float* qy_new,
            const unsigned char* inflow, int ny, int nx, float dt, float dx, float theta, float s_out) {
    // qy has shape (ny+1, nx); face j sits between cell row j-1 (north) and row j (south)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j > ny) return;
    int f = j * nx + i;
    if (j == 0 || j == ny) {
        int c = (j == 0 ? 0 : ny - 1) * nx + i;
        float hc = h[c];
        float q = 0.f;
        if (hc > H_MIN_ && !(j == 0 && inflow[c]))
            q = powf(hc, 5.f / 3.f) * sqrtf(s_out) / n[c];
        qy_new[f] = (j == 0) ? -q : q;
        return;
    }
    int u = (j - 1) * nx + i, d = j * nx + i;
    float etu = z[u] + h[u], etd = z[d] + h[d];
    float hf = fmaxf(etu, etd) - fmaxf(z[u], z[d]);
    if (hf <= H_MIN_) { qy_new[f] = 0.f; return; }
    float q0 = qy[f];
    float qm = qy[f - nx], qp = qy[f + nx];
    float qa = theta * q0 + 0.5f * (1.f - theta) * (qm + qp);
    float nf = 0.5f * (n[u] + n[d]);
    float num = qa - G_ * hf * dt * (etd - etu) / dx;
    float den = 1.f + G_ * dt * nf * nf * fabsf(q0) / powf(hf, 7.f / 3.f);
    float qn = num / den;
    float qcap = FR_MAX_ * hf * sqrtf(G_ * hf);  // local-inertial form is only valid for subcritical flow
    qy_new[f] = fmaxf(-qcap, fminf(qcap, qn));
}
""".replace("H_MIN_", f"{H_MIN}f").replace("FR_MAX_", f"{FR_MAX}f").replace("G_", f"{G}f"), "flux_y")

_limit_factor = cp.RawKernel(r"""
extern "C" __global__
void limit_factor(const float* h, const float* qx, const float* qy, const float* src, float* fac,
                  int ny, int nx, float dt, float dx) {
    // Fraction of each cell's outgoing flux that its current volume can supply this step
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int c = j * nx + i;
    float qw = qx[j * (nx + 1) + i], qe = qx[j * (nx + 1) + i + 1];
    float qn = qy[j * nx + i], qs = qy[(j + 1) * nx + i];
    float out = fmaxf(-qw, 0.f) + fmaxf(qe, 0.f) + fmaxf(-qn, 0.f) + fmaxf(qs, 0.f);
    float need = dt * out / dx;
    float have = h[c] + dt * src[c];
    fac[c] = (need > have && need > 0.f) ? have / need : 1.f;
}
""", "limit_factor")

_apply_limit = cp.RawKernel(r"""
extern "C" __global__
void apply_limit(float* qx, float* qy, const float* fac, int ny, int nx) {
    // Scale each face flux by the limiting factor of its donor cell
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j < ny && i <= nx) {
        int f = j * (nx + 1) + i;
        float q = qx[f];
        int donor = (q > 0.f) ? i - 1 : i;
        if (donor >= 0 && donor < nx) qx[f] = q * fac[j * nx + donor];
    }
    if (j <= ny && i < nx) {
        int f = j * nx + i;
        float q = qy[f];
        int donor = (q > 0.f) ? j - 1 : j;
        if (donor >= 0 && donor < ny) qy[f] = q * fac[donor * nx + i];
    }
}
""", "apply_limit")

_update_h = cp.RawKernel(r"""
extern "C" __global__
void update_h(float* h, const float* qx, const float* qy, const float* src,
              int ny, int nx, float dt, float dx, float* clipped) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int c = j * nx + i;
    float div = (qx[j * (nx + 1) + i] - qx[j * (nx + 1) + i + 1]
               + qy[j * nx + i] - qy[(j + 1) * nx + i]) / dx;
    float hn = h[c] + dt * (div + src[c]);
    if (hn < 0.f) { atomicAdd(clipped, -hn * dx * dx); hn = 0.f; }
    h[c] = hn;
}
""", "update_h")


@dataclass
class Result:
    dx: float
    h: np.ndarray
    u: np.ndarray
    v: np.ndarray
    sim_time_s: float
    wall_time_s: float
    steps: int
    outflow_m3s: float
    inflow_m3s: float
    mass_error_rel: float
    history: list = field(default_factory=list)


def run(z, manning, inflow_mask, Q, dx, h0=None, theta=0.7, alpha=0.5, s_out=3e-4,
        t_min=3600.0, t_max=4 * 3600.0, check_every=300.0, tol_q=0.005, tol_v=0.001,
        closed_w=True, closed_e=False, log=print):
    """Run to steady state. z, manning, inflow_mask are (ny, nx) arrays in metres, row 0 = north.

    Inflow Q (m3/s) enters spread evenly over the inflow cells in the north row. All other
    boundary faces are free outflow at normal depth with energy slope s_out, except the west
    edge (closed_w), which is high ground in this domain.
    """
    ny, nx = z.shape
    zd = cp.asarray(z, dtype=cp.float32)
    nd = cp.asarray(manning, dtype=cp.float32)
    infl = cp.asarray(inflow_mask.astype(np.uint8))
    n_in = int(inflow_mask.sum())
    src = cp.zeros((ny, nx), cp.float32)
    src[infl.astype(bool)] = Q / (n_in * dx * dx)
    h = cp.asarray(h0 if h0 is not None else np.zeros_like(z), dtype=cp.float32)
    qx = cp.zeros((ny, nx + 1), cp.float32)
    qy = cp.zeros((ny + 1, nx), cp.float32)
    qx2, qy2 = cp.empty_like(qx), cp.empty_like(qy)
    clipped = cp.zeros(1, cp.float32)
    fac = cp.ones((ny, nx), cp.float32)

    bs = (32, 8)
    gx = lambda w, hgt: ((w + bs[0] - 1) // bs[0], (hgt + bs[1] - 1) // bs[1])

    t, steps, dt = 0.0, 0, 0.0
    vol0 = float(h.sum()) * dx * dx
    in_vol = out_vol = 0.0
    last_check_t, last_vol = 0.0, vol0
    history = []
    t0 = time.perf_counter()
    while t < t_max:
        if steps % 20 == 0:
            hmax = max(float(h.max()), 0.01)
            dt = alpha * dx / np.sqrt(G * hmax)
        f32 = np.float32
        _flux_x(gx(nx + 1, ny), bs, (zd, h, nd, qx, qx2, np.int32(ny), np.int32(nx), f32(dt), f32(dx),
                                     f32(theta), f32(s_out), np.int32(closed_w), np.int32(closed_e)))
        _flux_y(gx(nx, ny + 1), bs, (zd, h, nd, qy, qy2, infl, np.int32(ny), np.int32(nx), f32(dt), f32(dx),
                                     f32(theta), f32(s_out)))
        qx, qx2 = qx2, qx
        qy, qy2 = qy2, qy
        _limit_factor(gx(nx, ny), bs, (h, qx, qy, src, fac, np.int32(ny), np.int32(nx), f32(dt), f32(dx)))
        _apply_limit(gx(nx + 1, ny + 1), bs, (qx, qy, fac, np.int32(ny), np.int32(nx)))
        _update_h(gx(nx, ny), bs, (h, qx, qy, src, np.int32(ny), np.int32(nx), f32(dt), f32(dx), clipped))
        t += dt
        steps += 1
        in_vol += Q * dt
        if t - last_check_t >= check_every or t >= t_max:
            out_q = float((-qx[:, 0].sum() + qx[:, nx].sum() - qy[0, :].sum() + qy[ny, :].sum()) * dx)
            vol = float(h.sum()) * dx * dx
            dvol_rel = abs(vol - last_vol) / max(vol, 1.0)
            imbalance = abs(Q - out_q) / Q
            history.append(dict(t=t, out_q=out_q, vol=vol, dvol_rel=dvol_rel, imbalance=imbalance))
            log(f"  dx={dx} Q={Q} t={t/60:6.1f} min  out={out_q:7.1f} m3/s  imbalance={imbalance:.4f}  dV/V={dvol_rel:.5f}")
            last_check_t, last_vol = t, vol
            if t >= t_min and imbalance < tol_q and dvol_rel < tol_v:
                break
    cp.cuda.Stream.null.synchronize()
    wall = time.perf_counter() - t0

    # Cell-centred velocities from face fluxes
    hc = h.get().astype(np.float64)
    qxc = 0.5 * (qx[:, :-1] + qx[:, 1:]).get()
    qyc = 0.5 * (qy[:-1, :] + qy[1:, :]).get()
    wet = hc > 0.01
    u = np.where(wet, qxc / np.maximum(hc, 0.01), 0.0)
    v = np.where(wet, -qyc / np.maximum(hc, 0.01), 0.0)  # positive v = northward
    vol = hc.sum() * dx * dx
    # Mass balance: initial + inflow - outflow(integrated approx) - final; report clipping share
    mass_error_rel = float(clipped.get()[0]) / max(in_vol, 1.0)
    return Result(dx=dx, h=hc, u=u, v=v, sim_time_s=t, wall_time_s=wall, steps=steps,
                  outflow_m3s=history[-1]["out_q"], inflow_m3s=Q, mass_error_rel=mass_error_rel,
                  history=history)
