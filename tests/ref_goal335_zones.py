"""Frozen copy of goal335 raw_beid.zones_kernel (original BEID state machine), kept only as a parity reference for tests. Do not edit."""
import numpy as np
from numba import njit

@njit(cache=True)
def zones_kernel(o, h, l, c, day, daily):
    n = len(o); cap = n // 2 + 16
    zc = np.empty(cap, np.int64); zs = np.empty(cap, np.int8); ze = np.empty(cap); zt = np.empty(cap); zl = np.empty(cap); ztop = np.empty(cap); zbot = np.empty(cap)
    nz = 0
    nan = np.nan
    up = True
    pHH = nan; pHH_i = -1; cHH = nan; cHH_i = -1; pHL = nan; pHL_i = -1; lastcHL = nan
    pLL = nan; pLL_i = -1; cLL = nan; cLL_i = -1; pLH = nan; pLH_i = -1; lastcLH = nan
    inpb = False; pg = nan; pr = nan
    for i in range(n):
        if daily and i > 0 and day[i] != day[i - 1]:
            up = True; pHH = nan; cHH = nan; pHL = nan; lastcHL = nan; pLL = nan; cLL = nan; pLH = nan; lastcLH = nan
            inpb = False; pg = nan; pr = nan
        oo = o[i]; hh = h[i]; ll = l[i]; cl = c[i]
        green = cl >= oo; red = cl < oo
        if up:
            if np.isnan(pHH) or hh > pHH:
                pHH = hh; pHH_i = i
            if (not np.isnan(lastcHL)) and cl < lastcHL:
                top = pHH; bot = lastcHL
                up = False; cLH = pHH; lastcLH = nan; pLL = ll; pLL_i = i; cLL = nan; inpb = False
                zc[nz] = i; zs[nz] = 3; ztop[nz] = top; zbot[nz] = bot; nz += 1
            else:
                if (not inpb) and red and (not np.isnan(pg)) and cl < pg:
                    inpb = True; cHH = pHH; cHH_i = pHH_i; pHL = ll; pHL_i = i
                elif inpb:
                    if ll < pHL:
                        pHL = ll; pHL_i = i
                    if cl > cHH:
                        inpb = False
                        lastcHL = pHL
                        top = cHH; bot = pHL
                        pHH = hh; pHH_i = i
                        zc[nz] = i; zs[nz] = 0; ztop[nz] = top; zbot[nz] = bot; nz += 1
        else:
            if np.isnan(pLL) or ll < pLL:
                pLL = ll; pLL_i = i
            if (not np.isnan(lastcLH)) and cl > lastcLH:
                top = lastcLH; bot = pLL
                up = True; cHL = pLL; lastcHL = nan; pHH = hh; pHH_i = i; cHH = nan; inpb = False
                zc[nz] = i; zs[nz] = 2; ztop[nz] = top; zbot[nz] = bot; nz += 1
            else:
                if (not inpb) and green and (not np.isnan(pr)) and cl > pr:
                    inpb = True; cLL = pLL; cLL_i = pLL_i; pLH = hh; pLH_i = i
                elif inpb:
                    if hh > pLH:
                        pLH = hh; pLH_i = i
                    if cl < cLL:
                        inpb = False
                        lastcLH = pLH
                        top = pLH; bot = cLL
                        pLL = ll; pLL_i = i
                        zc[nz] = i; zs[nz] = 1; ztop[nz] = top; zbot[nz] = bot; nz += 1
        if green: pg = oo
        elif red: pr = oo
    return zc[:nz], zs[:nz], ztop[:nz], zbot[:nz]

