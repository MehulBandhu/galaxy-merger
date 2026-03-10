"""Snapshot reader and basic analysis."""

import numpy as np
import struct, os

PTYPE_DM=0; PTYPE_DISK=1; PTYPE_BULGE=2; PTYPE_GAS=3; PTYPE_STAR=4; PTYPE_BH=5
TIME_GYR = 0.978

def read_snapshot(fn):
    with open(fn, 'rb') as f:
        Nt = struct.unpack('i',f.read(4))[0]
        Ng = struct.unpack('i',f.read(4))[0]
        f.read(5*4)
        t = struct.unpack('f',f.read(4))[0]
        dt = struct.unpack('f',f.read(4))[0]
        sn = struct.unpack('i',f.read(4))[0]
        x  = np.frombuffer(f.read(Nt*4), np.float32)
        y  = np.frombuffer(f.read(Nt*4), np.float32)
        z  = np.frombuffer(f.read(Nt*4), np.float32)
        vx = np.frombuffer(f.read(Nt*4), np.float32)
        vy = np.frombuffer(f.read(Nt*4), np.float32)
        vz = np.frombuffer(f.read(Nt*4), np.float32)
        mass = np.frombuffer(f.read(Nt*4), np.float32)
        u   = np.frombuffer(f.read(Nt*4), np.float32)
        rho = np.frombuffer(f.read(Nt*4), np.float32)
        pt  = np.frombuffer(f.read(Nt*4), np.int32)
        gid = np.frombuffer(f.read(Nt*4), np.int32)
    return dict(N_total=Nt, N_gas=Ng, time=t, dt=dt, snap_num=sn,
                x=x, y=y, z=z, vx=vx, vy=vy, vz=vz,
                mass=mass, u=u, rho=rho, ptype=pt, galaxy_id=gid,
                time_myr=t*TIME_GYR*1000)
