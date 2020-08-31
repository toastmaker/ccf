import numpy as np

def lcurve(time,fac=600,sub=0,tstart=0):
    #fac: grouping factor - binning = 1/fac seconds
    tlab=(time*fac).astype(int)
    pos=np.r_[0,np.where(tlab[1:]-tlab[:-1])[0]]
    cnt=pos[1:]-pos[:-1]
    if sub!=0: #baseline
        basel=np.median(cnt[sub:])
        cnt=cnt.astype(float)-basel
    tbin=np.r_[tlab[0]:tlab[0]+len(cnt)]/fac
    if tstart>0:
        tsel=tbin>tstart
        tbin=tbin[tsel]
        cnt=cnt[tsel]
    return tbin,cnt*fac

def wcent(cfunc,nbin=10,sbin=20,cent=0):
    if cent==0: cent=len(cfunc)//2
    #cent=(qmid-qsiz)*fac+3
    ic=np.arange(cent-nbin*sbin,cent+nbin*sbin)
    m1=cfunc[cent-nbin*sbin:cent+nbin*sbin].reshape(2*nbin,sbin).sum(1)
    m2=(ic*cfunc[cent-nbin*sbin:cent+nbin*sbin]).reshape(2*nbin,sbin).sum(1)
    bcent=(m2[nbin-1:None:-1]+m2[nbin:]).cumsum()
    bcent/=(m1[nbin-1:None:-1]+m1[nbin:]).cumsum()
    return bcent


def fit_corr(cfunc,csiz=200,doplot=True,inifac=1,cmax=0):
    if cmax==0: cmax=np.argmax(cfunc)
    p0=[cmax,np.log(cfunc.max()),150*inifac,200*inifac,0]
    x=np.r_[cmax-csiz:cmax+csiz]
    fun1=lambda p:np.exp(-(p[0]-x)/p[2]+p[1])*(x<p[0])+(x>=p[0])*np.exp(-(x-p[0])/p[3]+p[1])+p[4]
    #print(p0)
    from scipy import optimize as op
    bst=op.fmin(lambda p:((cfunc[cmax-csiz:cmax+csiz]-fun1(p))**2).sum(),p0,disp=0)
    #pl.xlim(cmax-500,cmax+500)
    if doplot:
        from matplotlib import pyplot as pl
        pl.plot(x,cfunc[cmax-csiz:cmax+csiz],x,fun1(p0),x,fun1(bst))
    #pl.plot(x,fun1(p0))
    chi2=((cfunc[cmax-csiz:cmax+csiz]-fun1(bst))**2).sum()
    return list(bst)+[chi2]
