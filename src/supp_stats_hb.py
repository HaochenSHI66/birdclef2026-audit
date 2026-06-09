import sys, time, os
sys.argv = ["supp_stats_hardening.py","--n_null","200","--n_boot","2000","--n_repsplits","20","--seed","0"]
import b2_oracle_headroom as b2
_c=[0]; _t0=time.time()
_orig=b2.macro_auc_on_mask
def _w(*a,**k):
    _c[0]+=1
    if _c[0] % 120 == 0:
        print("[hb %s] pid=%d auc_calls=%d elapsed=%ds" % (time.strftime("%H:%M:%S"), os.getpid(), _c[0], int(time.time()-_t0)), flush=True)
    return _orig(*a,**k)
b2.macro_auc_on_mask=_w
try:
    import b2_corrected as bc
    if getattr(bc,"macro_auc_on_mask",None) is not None: bc.macro_auc_on_mask=_w
except Exception as e:
    print("warn bc patch:", e, flush=True)
import supp_stats_hardening as s
if getattr(s,"macro_auc_on_mask",None) is not None: s.macro_auc_on_mask=_w
print("[hb %s] START deepstats wrapper n_null=200 n_boot=2000" % time.strftime("%H:%M:%S"), flush=True)
s.main()
