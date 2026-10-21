import numpy as np
from . import potential_game as p
def gap_series(A,d):
    """Nash gap of the empirical frequency profile: max_i [max_k U_i(k) - sum_k f_i(k) U_i(k)],
    U_i(k)=prod_{j!=i}(1-f_j(k))/d_k (expected utility against independent mixed strategies)."""
    T,N=A.shape;K=len(d)
    oh=np.zeros((T,N,K));
    for i in range(N): oh[np.arange(T),i,A[:,i]]=1
    F=np.cumsum(oh,axis=0)/np.arange(1,T+1)[:,None,None]
    out=np.zeros(T)
    one_minus=1-F  # (T,N,K)
    g=np.zeros((T,N))
    for i in range(N):
        U=np.prod(np.delete(one_minus,i,axis=1),axis=1)/d[None,:]      # (T,K)
        g[:,i]=U.max(1)-(F[:,i,:]*U).sum(1)
    return g.max(1)
