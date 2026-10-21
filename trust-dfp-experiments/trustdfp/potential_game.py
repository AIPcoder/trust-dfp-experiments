"""Exact-potential target-assignment game for the trust-filtered DFP simulator.
Reuses dfp_v8_lil (algorithm unchanged, xi_t = 1.4 sqrt(t)); only the game is replaced:
  * no private distance signals (signal noise = 0),
  * a COMMON cost d_k per target (network mean of the true agent-target distances),
so that u_i(a) = 1[no other agent picks a_i] / d_{a_i}  is an exact potential game
with potential  Phi(a) = sum_{k in {a_1..a_N}} 1/d_k   (Monderer-Shapley 1996)."""
import numpy as np
from . import dfp_v8_lil as base
from itertools import product
base.XI_MODE='sqrt'

def build_potential_game(N=5,K=5,T=5000,seed=42,**kw):
    rng=np.random.default_rng(seed)
    ap=rng.normal(0,np.sqrt(0.1),size=(N,2))
    ang=np.linspace(0,2*np.pi,K,endpoint=False)
    tp=np.column_stack([np.cos(ang),np.sin(ang)])
    d_ik=np.array([[np.linalg.norm(ap[i]-tp[k]) for k in range(K)] for i in range(N)])
    d=d_ik.mean(axis=0)                     # common target cost
    def utility_fn(i,k,v):
        p=1.0
        for j in range(N):
            if j!=i: p*=float(1.0-v[i,j,k])
        return float(p/d[k])
    g=base.DFPGame(N=N,K=K,utility_fn=utility_fn,T=T,seed=seed,**kw)
    return g,d,d_ik

def true_u(a,d):
    a=np.asarray(a); return np.array([ (1.0/d[a[i]]) if (a==a[i]).sum()==1 else 0.0 for i in range(len(a))])
def phi(a,d): return sum(1.0/d[k] for k in set(np.asarray(a).tolist()))
def is_ne(a,d):
    a=np.asarray(a);K=len(d)
    for i in range(len(a)):
        u0=true_u(a,d)[i]
        for k in range(K):
            b=a.copy();b[i]=k
            if true_u(b,d)[i]>u0+1e-12: return False
    return True

def potential_defect(N,K,cost):
    """max |du_i - dPhi| over all profiles/unilateral deviations; cost: (K,) common or (N,K) agent-specific"""
    cost=np.atleast_2d(cost)
    def u(a,i):
        c=cost[i] if cost.shape[0]>1 else cost[0]
        return (1.0/c[a[i]]) if sum(1 for x in a if x==a[i])==1 else 0.0
    # Phi defined with the common-cost formula (mean over rows if agent-specific)
    cm=cost.mean(axis=0)
    ph=lambda a:sum(1.0/cm[k] for k in set(a))
    m=0.0
    for a in product(range(K),repeat=N):
        for i in range(N):
            for k in range(K):
                b=list(a);b[i]=k
                m=max(m,abs((u(b,i)-u(a,i))-(ph(b)-ph(a))))
    return m
