"""
Galaxy merger initial conditions.
Units: 1 kpc, 1e10 Msun, 1 km/s.  G = 43007.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import struct, argparse, time as wt

G = 43007.0

PTYPE_DM = 0; PTYPE_DISK = 1; PTYPE_BULGE = 2; PTYPE_GAS = 3; PTYPE_BH = 5

# -- profiles --

def hernquist_density(r, M, a):
    return M * a / (2 * np.pi * r * (r + a)**3)

def hernquist_mass(r, M, a):
    return M * r**2 / (r + a)**2

def hernquist_potential(r, M, a):
    return -G * M / (r + a)

def exp_disk_sigma(R, Md, Rd):
    return Md / (2 * np.pi * Rd**2) * np.exp(-R / Rd)


class GalaxyModel:
    def __init__(self, p):
        self.p = p
        self.R_grid = np.logspace(-3, np.log10(p['R_max']), 500)
        Rd = p['Rd_star']; Md = p['M_disk']
        self.disk_mass = Md * (1 - (1 + self.R_grid/Rd) * np.exp(-self.R_grid/Rd))
        if p['M_gas'] > 0:
            Rg = p['Rd_gas']; Mg = p['M_gas']
            self.gas_mass = Mg * (1 - (1 + self.R_grid/Rg) * np.exp(-self.R_grid/Rg))
        else:
            self.gas_mass = np.zeros_like(self.R_grid)

    def potential(self, r):
        p = self.p
        phi = hernquist_potential(r, p['M_halo'], p['a_halo'])
        phi += hernquist_potential(r, p['M_bulge'], p['a_bulge'])
        phi += hernquist_potential(r, p['M_bh'], 0.01)
        if r > 0.001:
            Md_enc = p['M_disk'] * (1 - (1+r/p['Rd_star'])*np.exp(-r/p['Rd_star']))
            phi -= G * Md_enc / r
            if p['M_gas'] > 0:
                Mg_enc = p['M_gas'] * (1 - (1+r/p['Rd_gas'])*np.exp(-r/p['Rd_gas']))
                phi -= G * Mg_enc / r
        return phi

    def vc2(self, R):
        p = self.p
        v2 = G * hernquist_mass(R, p['M_halo'], p['a_halo']) / R
        v2 += G * hernquist_mass(R, p['M_bulge'], p['a_bulge']) / R
        v2 += G * np.interp(R, self.R_grid, self.disk_mass) / R
        v2 += G * np.interp(R, self.R_grid, self.gas_mass) / R
        v2 += G * p['M_bh'] / R
        return np.maximum(v2, 0.0)

    def epicyclic(self, R, dR=0.01):
        Rp, Rm = R*(1+dR), R*(1-dR)
        Op2 = self.vc2(Rp) / Rp**2
        Om2 = self.vc2(Rm) / Rm**2
        O2 = self.vc2(R) / R**2
        return np.sqrt(np.maximum(R*(Op2-Om2)/(Rp-Rm) + 4*O2, 0.0))


# -- sampling --

def sample_hernquist(N, M, a, r_max):
    mfrac = hernquist_mass(r_max, 1.0, a)
    u = np.random.uniform(0, 1, N)
    s = np.sqrt(u * mfrac)
    s = np.minimum(s, 0.9999)
    r = a * s / (1 - s)
    r = np.minimum(r, r_max)
    ct = np.random.uniform(-1, 1, N)
    st = np.sqrt(1 - ct**2)
    phi = np.random.uniform(0, 2*np.pi, N)
    return r*st*np.cos(phi), r*st*np.sin(phi), r*ct, r, mfrac

def hernquist_velocities(r, M, a, model):
    rmin = max(np.min(r)*0.1, 0.001)
    rmax = max(np.max(r)*5, 500.0)
    rt = np.logspace(np.log10(rmin), np.log10(rmax), 500)
    rho_t = hernquist_density(rt, M, a)
    vc2_t = np.array([model.vc2(ri) for ri in rt])
    g_t = vc2_t / rt
    integrand = rho_t * g_t

    sig2 = np.zeros_like(rt)
    for i in range(len(rt)-2, -1, -1):
        sig2[i] = sig2[i+1] + 0.5*(integrand[i]+integrand[i+1])*(rt[i+1]-rt[i])
    sig2 /= np.maximum(rho_t, 1e-30)

    sig_f = interp1d(np.log10(rt), np.sqrt(np.maximum(sig2, 0)),
                      fill_value='extrapolate', bounds_error=False)
    sig = np.maximum(sig_f(np.log10(np.maximum(r, rmin))), 1.0)

    vesc = np.sqrt(np.maximum(-2*np.array([model.potential(ri) for ri in r]), 0))
    vx = np.random.normal(0, sig, len(r))
    vy = np.random.normal(0, sig, len(r))
    vz = np.random.normal(0, sig, len(r))
    vmag = np.sqrt(vx**2+vy**2+vz**2)
    esc = vmag > 0.9*vesc
    if esc.any():
        s = 0.85*vesc[esc]/vmag[esc]
        vx[esc]*=s; vy[esc]*=s; vz[esc]*=s
    return vx, vy, vz

def sample_disk(N, Rd, z0, R_max):
    cdf_max = 1 - (1+R_max/Rd)*np.exp(-R_max/Rd)
    u = np.random.uniform(0,1,N) * cdf_max
    R = np.array([brentq(lambda r: (1-(1+r/Rd)*np.exp(-r/Rd))-ui, 0.001, R_max) for ui in u])
    z = 2*z0*np.arctanh(np.random.uniform(-1,1,N)*0.999)
    phi = np.random.uniform(0, 2*np.pi, N)
    return R*np.cos(phi), R*np.sin(phi), z, R, phi

def disk_velocities(R, model, Md, Rd, z0, Q=1.3):
    vc2 = model.vc2(R); vc = np.sqrt(vc2)
    Sigma = exp_disk_sigma(R, Md, Rd)
    kappa = np.array([model.epicyclic(ri) for ri in R])
    Omega = vc / np.maximum(R, 0.01)

    sig_R = np.clip(Q * 3.36 * G * Sigma / np.maximum(kappa, 1e-10), 3, 150)
    sig_phi = np.maximum(sig_R * kappa / (2*np.maximum(Omega, 1e-10)), 2)
    sig_z = np.maximum(np.sqrt(np.pi * G * Sigma * z0), 2)

    va2 = np.minimum(sig_R**2 * (0.5*R/Rd), 0.9*vc2)
    vphi = np.sqrt(np.maximum(vc2 - va2, 0.1))
    return vphi, sig_R, sig_phi, sig_z


# -- galaxy builder --

def build_galaxy(p, verbose=True):
    model = GalaxyModel(p)
    if verbose:
        print(f"  rotation curve: ", end="")
        for R in [5,10,20]:
            print(f"{R} kpc -> {np.sqrt(model.vc2(float(R))):.0f}", end="  ")
        print("km/s")

    parts = {'x':[], 'y':[], 'z':[], 'vx':[], 'vy':[], 'vz':[],
             'mass':[], 'ptype':[], 'N_gas': 0}

    def add(x,y,z,vx,vy,vz,m,pt):
        parts['x'].append(x); parts['y'].append(y); parts['z'].append(z)
        parts['vx'].append(vx); parts['vy'].append(vy); parts['vz'].append(vz)
        parts['mass'].append(np.full(len(x),m)); parts['ptype'].append(np.full(len(x),pt,dtype=np.int32))

    # gas disk (stored first)
    if p['N_gas'] > 0 and p['M_gas'] > 0:
        Rmax = 6*p['Rd_gas']
        mfrac = 1-(1+Rmax/p['Rd_gas'])*np.exp(-Rmax/p['Rd_gas'])
        mp = p['M_gas']*mfrac/p['N_gas']
        x,y,z,R,phi = sample_disk(p['N_gas'], p['Rd_gas'], p['z0_gas'], Rmax)
        vc = np.sqrt(model.vc2(R))
        vx = -vc*np.sin(phi); vy = vc*np.cos(phi); vz = np.zeros(p['N_gas'])
        add(x,y,z,vx,vy,vz,mp,PTYPE_GAS)
        parts['N_gas'] = p['N_gas']
        if verbose: print(f"  gas: {p['N_gas']} particles, mfrac={mfrac:.3f}")

    # DM halo
    if p['N_dm'] > 0:
        x,y,z,r,mf = sample_hernquist(p['N_dm'], p['M_halo'], p['a_halo'], p['r_max_halo'])
        mp = p['M_halo']*mf/p['N_dm']
        vx,vy,vz = hernquist_velocities(r, p['M_halo'], p['a_halo'], model)
        add(x,y,z,vx,vy,vz,mp,PTYPE_DM)
        if verbose: print(f"  DM: {p['N_dm']} particles, mfrac={mf:.3f}")

    # stellar disk
    if p['N_disk'] > 0:
        Rmax = 6*p['Rd_star']
        mfrac = 1-(1+Rmax/p['Rd_star'])*np.exp(-Rmax/p['Rd_star'])
        mp = p['M_disk']*mfrac/p['N_disk']
        x,y,z,R,phi = sample_disk(p['N_disk'], p['Rd_star'], p['z0_star'], Rmax)
        vphi, sR, sP, sZ = disk_velocities(R, model, p['M_disk'], p['Rd_star'], p['z0_star'])
        vR = np.random.normal(0, sR, p['N_disk'])
        vp = vphi + np.random.normal(0, sP, p['N_disk'])
        vz = np.random.normal(0, sZ, p['N_disk'])
        vx = vR*np.cos(phi) - vp*np.sin(phi)
        vy = vR*np.sin(phi) + vp*np.cos(phi)
        add(x,y,z,vx,vy,vz,mp,PTYPE_DISK)
        if verbose: print(f"  disk: {p['N_disk']} particles, mfrac={mfrac:.3f}")

    # bulge
    if p['N_bulge'] > 0:
        x,y,z,r,mf = sample_hernquist(p['N_bulge'], p['M_bulge'], p['a_bulge'], 20*p['a_bulge'])
        mp = p['M_bulge']*mf/p['N_bulge']
        vx,vy,vz = hernquist_velocities(r, p['M_bulge'], p['a_bulge'], model)
        add(x,y,z,vx,vy,vz,mp,PTYPE_BULGE)
        if verbose: print(f"  bulge: {p['N_bulge']} particles, mfrac={mf:.3f}")

    # black hole
    add([0],[0],[0],[0],[0],[0],p['M_bh'],PTYPE_BH)

    for k in ['x','y','z','vx','vy','vz','mass']:
        parts[k] = np.concatenate(parts[k])
    parts['ptype'] = np.concatenate(parts['ptype'])
    return parts


# -- M51 parameters --

def m51a(N):
    N = int(N)
    return dict(name='M51a', M_halo=100.0, M_disk=5.0, M_bulge=1.0, M_gas=1.0, M_bh=1e-3,
                a_halo=40.0, Rd_star=3.5, z0_star=0.35, Rd_gas=5.0, z0_gas=0.15, a_bulge=0.7,
                N_dm=max(int(0.40*N),100), N_disk=max(int(0.20*N),50),
                N_bulge=max(int(0.05*N),20), N_gas=max(int(0.20*N),50),
                R_max=2000.0, r_max_halo=2000.0, T_gas=1e4)

def ngc5195(N):
    N = int(N)
    return dict(name='NGC5195', M_halo=40.0, M_disk=2.0, M_bulge=1.5, M_gas=0.3, M_bh=5e-4,
                a_halo=30.0, Rd_star=2.5, z0_star=0.4, Rd_gas=3.5, z0_gas=0.15, a_bulge=0.8,
                N_dm=max(int(0.10*N),50), N_disk=max(int(0.05*N),30),
                N_bulge=max(int(0.02*N),15), N_gas=max(int(0.03*N),20),
                R_max=1500.0, r_max_halo=1500.0, T_gas=1e4)


# -- orbit setup --

def setup_orbit(g1, g2, sep=100.0, rperi=20.0, inc_deg=20.0):
    Mtot = g1['mass'].sum() + g2['mass'].sum()
    L = np.sqrt(2*G*Mtot*rperi)
    vt = L/sep
    vr = np.sqrt(max(2*G*Mtot/sep - vt**2, 0))
    inc = np.radians(inc_deg)

    g2['x'] += sep
    g2['vx'] -= vr
    g2['vy'] += vt * np.cos(inc)
    g2['vz'] += vt * np.sin(inc)

    print(f"  orbit: sep={sep:.0f} rperi={rperi:.0f} vr={vr:.0f} vt={vt:.0f} km/s")
    return g2


# -- merge and write --

def merge(g1, g2):
    gid1 = np.zeros(len(g1['x']), dtype=np.int32)
    gid2 = np.ones(len(g2['x']), dtype=np.int32)

    # gas first, then everything else
    gas1 = g1['ptype'] == PTYPE_GAS; gas2 = g2['ptype'] == PTYPE_GAS
    out = {}
    for k in ['x','y','z','vx','vy','vz','mass','ptype']:
        out[k] = np.concatenate([g1[k][gas1], g2[k][gas2], g1[k][~gas1], g2[k][~gas2]])
    out['galaxy_id'] = np.concatenate([gid1[gas1], gid2[gas2], gid1[~gas1], gid2[~gas2]])
    out['N_gas'] = gas1.sum() + gas2.sum()
    return out

def write_ic(fn, d):
    N = len(d['x']); Ng = d['N_gas']
    pt = d['ptype']
    with open(fn, 'wb') as f:
        for v in [N, Ng, (pt==0).sum(), (pt==1).sum(), (pt==2).sum(), 0, (pt==5).sum()]:
            f.write(struct.pack('i', v))
        f.write(struct.pack('f', 0.0)); f.write(struct.pack('f', 0.0))
        f.write(struct.pack('i', 0))
        for k in ['x','y','z','vx','vy','vz','mass']:
            f.write(d[k].astype(np.float32).tobytes())
        f.write(np.zeros(N, dtype=np.float32).tobytes())  # u
        f.write(np.zeros(N, dtype=np.float32).tobytes())  # rho
        f.write(d['ptype'].astype(np.int32).tobytes())
        f.write(d['galaxy_id'].astype(np.int32).tobytes())
    print(f"  wrote {fn}: N={N} ({N*11*4/1e6:.1f} MB)")


# -- main --

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, default=10000)
    ap.add_argument('--output', default='ic.bin')
    ap.add_argument('--isolation', action='store_true')
    ap.add_argument('--separation', type=float, default=100.0)
    ap.add_argument('--pericenter', type=float, default=20.0)
    ap.add_argument('--inclination', type=float, default=20.0)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    np.random.seed(args.seed)

    t0 = wt.time()
    print(f"N={args.N}")

    print("M51a:")
    g1 = build_galaxy(m51a(args.N))

    if args.isolation:
        g1['galaxy_id'] = np.zeros(len(g1['x']), dtype=np.int32)
        write_ic(args.output, g1)
    else:
        print("NGC5195:")
        g2 = build_galaxy(ngc5195(args.N))
        g2 = setup_orbit(g1, g2, args.separation, args.pericenter, args.inclination)
        out = merge(g1, g2)
        write_ic(args.output, out)

    print(f"  {wt.time()-t0:.1f}s")

if __name__ == '__main__':
    main()
