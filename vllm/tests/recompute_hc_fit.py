import json, numpy as np, os, torch
from safetensors import safe_open
W4A16='/mnt/data/models/VnimanieAI-Qwen3.8-Flash-Next-W4A16/model-00001.safetensors'
HC,N_EMBD,HC_DIM=4,2560,10240
EPS=1e-6
CWD='/home/dc/electric-sheep/llama/llama.cpp'
m=json.load(open('/tmp/llama_actdump_l0b/manifest.json'))
names=[x['name'] for x in m['tensors']]
def load_llama(name):
    e=next(x for x in m['tensors'] if x['name']==name)
    p=e['file']
    if not os.path.isabs(p): p=os.path.join(CWD,p)
    ne=e['ne']; d=np.fromfile(p,dtype=np.float32); return d.reshape(ne[3],ne[2],ne[1],ne[0])
def cos(a,b):
    a=a.ravel();b=b.ravel();na=np.linalg.norm(a);nb=np.linalg.norm(b)
    return float(np.dot(a,b)/(na*nb)) if na>0 and nb>0 else float('nan')
hn_name=[n for n in names if 'hc_norm' in n]
print('hc_norm candidates:', hn_name)
hn_name=hn_name[0]
with safe_open(W4A16,framework='pt') as sf:
    w=sf.get_tensor('model.language_model.layers.0.attn_hyper_connection.hc_norm.weight').float().numpy()
init=load_llama('hc_init')[0]  # [5,4,2560]
l_norm=load_llama(hn_name)[0].reshape(5,HC_DIM)  # [5,10240]
x=torch.from_numpy(init).float(); T=x.shape[0]
var=x.pow(2).mean(dim=-1,keepdim=True); xn=x*torch.rsqrt(var+EPS)
my_xn=xn.reshape(T,HC_DIM).numpy()
gamma_1w=1.0+w
num=(l_norm*my_xn).sum(axis=0); den=(my_xn*my_xn).sum(axis=0)
gamma_fit=num/den
print('=== fitted gamma (least squares, per element) ===')
print('cos(gamma_fit, 1+w) = %.6f' % cos(gamma_fit, gamma_1w))
print('cos(gamma_fit, w)   = %.6f' % cos(gamma_fit, w))
print('gamma_fit: mean=%.4f std=%.4f' % (gamma_fit.mean(), gamma_fit.std()))
print('1+w:       mean=%.4f std=%.4f' % (gamma_1w.mean(), gamma_1w.std()))
print('w:         mean=%.4f std=%.4f' % (w.mean(), w.std()))
resid=l_norm-my_xn*gamma_fit
print('residual norm ratio = %.6f' % (np.linalg.norm(resid)/np.linalg.norm(l_norm)))
print()
print('=== rms_norm reduction test ===')
print('cos(my_xn*1+w, l_norm)  [per-stream] = %.6f' % cos(my_xn*gamma_1w, l_norm))
xf=x.reshape(T,HC_DIM)
var_full=xf.pow(2).mean(dim=-1,keepdim=True); xn_full=xf*torch.rsqrt(var_full+EPS)
print('cos(xn_full*1+w, l_norm) [whole]     = %.6f' % cos(xn_full.numpy()*gamma_1w, l_norm))
print()
print('=== per-token residual (after gamma fit) ===')
for t in range(T):
    r=np.linalg.norm(l_norm[t]-my_xn[t]*gamma_fit)/np.linalg.norm(l_norm[t])
    print('token%d: resid ratio=%.6f' % (t, r))