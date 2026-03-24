"""
=============================================================================
  BRAHMI RESTORATION — CORRECT INFERENCE (mirrors training exactly)
=============================================================================
"""
 
import argparse, os, sys
from pathlib import Path
 
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
 
 
# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------
 
class _ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, normalize=True):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=not normalize)]
        if normalize: layers.append(nn.InstanceNorm2d(out_c, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)
    def forward(self, x): return self.block(x)
 
class _UpBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=False):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
                  nn.InstanceNorm2d(out_c, affine=True), nn.ReLU(inplace=True)]
        if dropout: layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)
    def forward(self, x): return self.block(x)
 
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.e1=_ConvBlock(in_channels,64,normalize=False); self.e2=_ConvBlock(64,128)
        self.e3=_ConvBlock(128,256);  self.e4=_ConvBlock(256,512)
        self.e6=_ConvBlock(512,512);  self.e5=_ConvBlock(512,512)
        self.d6=_UpBlock(512,512,dropout=True); self.d5=_UpBlock(1024,512,dropout=True)
        self.d4=_UpBlock(1024,256); self.d3=_UpBlock(512,128); self.d2=_UpBlock(256,64)
        self.final=nn.Sequential(nn.ConvTranspose2d(128,out_channels,4,2,1),nn.Tanh())
    def forward(self,x):
        e1=self.e1(x); e2=self.e2(e1); e3=self.e3(e2); e4=self.e4(e3)
        e5=self.e5(e4); e6=self.e6(e5); d6=self.d6(e6)
        d5=self.d5(torch.cat([d6,e5],1)); d4=self.d4(torch.cat([d5,e4],1))
        d3=self.d3(torch.cat([d4,e3],1)); d2=self.d2(torch.cat([d3,e2],1))
        return self.final(torch.cat([d2,e1],1))
 
def compute_sobel(img):
    kx=torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]],
                     device=img.device).view(1,1,3,3)
    ky=kx.transpose(2,3); i01=(img+1.)*.5
    gx=F.conv2d(i01,kx,padding=1); gy=F.conv2d(i01,ky,padding=1)
    return ((torch.sqrt(gx**2+gy**2+1e-8)*.25).clamp(0,1)*2.)-1.
 
def load_model(path, device):
    if not os.path.exists(path): raise FileNotFoundError(path)
    G=UNetGenerator(3,1).to(device)
    ck=torch.load(path,map_location=device,weights_only=False)
    G.load_state_dict(ck["G_state"] if "G_state" in ck else ck)
    print(f"✓ Model loaded (epoch {ck.get('epoch','?')})")
    G.eval(); return G
 
 
# ---------------------------------------------------------------------------
# EXACT TRAINING PREPROCESSING — this is what was missing
# ---------------------------------------------------------------------------
 
def prepare_model_input(img_np: np.ndarray,
                         mask_np: np.ndarray,
                         damage_type: str = "binary") -> tuple:
    """
    Reproduce EXACTLY what BrahmiDataset.__getitem__ does during training.
 
    Training code (from your trainer):
      normalize: [-1, 1]  via  transforms.Normalize([0.5], [0.5])
      
      binary damage:
        I_dmg = I_gt * (1 - mask)        # masked pixels → 0.0 in [-1,1]
        soft_mask = mask                  # mask value = 1.0
        
      overlay damage:
        alpha ~ uniform(0.4, 0.85)
        gray  ~ uniform(-0.2, 0.3)
        I_dmg = I_gt*(1-mask) + mask*(I_gt*(1-alpha) + gray*alpha)
        soft_mask = mask * alpha          # mask value = alpha, NOT 1.0
 
    For real damaged images:
      - The gray stroke is ALREADY blended in (like overlay training)
      - We detect it and pass the ORIGINAL image (not erased)
      - We pass soft_mask = mask * estimated_alpha  (matching overlay training)
 
    Returns: (img_tensor, mask_tensor) both shape (1,1,H,W) in [-1,1]
    """
    tf = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize([0.5],[0.5])])
 
    # img_np is uint8 [0,255] grayscale
    img_t  = tf(Image.fromarray(img_np)).unsqueeze(0)   # [-1,1]
    mask_f = mask_np.astype(np.float32)                  # [0,1] float
 
    if damage_type == "overlay":
        # Real overlay damage: the stroke is semi-transparent.
        # Estimate alpha from how bright the damage pixels are.
        # Bright gray (~190/255) over white background means low alpha.
        # Dark gray (~130/255) means high alpha.
        # We use a fixed estimate of 0.65 (middle of training range 0.4-0.85).
        alpha     = 0.65
        soft_mask = mask_f * alpha
    else:
        # Binary damage: masked pixels were set to 0.0 in [-1,1].
        # DO NOT erase to white (255). Erase to 128 (= 0.0 in [-1,1]).
        erased_np = img_np.copy().astype(np.float32)
        erased_np[mask_f > 0.5] = 128.0   # 128 uint8 = 0.0 in [-1,1] space
        img_t     = tf(Image.fromarray(erased_np.astype(np.uint8))).unsqueeze(0)
        soft_mask = mask_f
 
    mask_t = torch.from_numpy(soft_mask).unsqueeze(0).unsqueeze(0).float()
    return img_t, mask_t
 
 
@torch.no_grad()
def run_model(G, img_t, mask_t, device):
    img    = img_t.to(device)
    mask   = mask_t.to(device)
    edges  = compute_sobel(img)
    amp    = device.type == "cuda"
    with torch.amp.autocast("cuda", enabled=amp):
        out = G(torch.cat([img, mask, edges], 1))
    return (out[0,0].cpu().float().numpy() * .5 + .5).clip(0,1)
 
 
# ---------------------------------------------------------------------------
# SHAPE-BASED DAMAGE DETECTION (proven working: score=0.85)
# ---------------------------------------------------------------------------
 
def detect_damage(img_np, low=100, high=220, min_area=400,
                  dilation=12, debug_dir=None, stem=""):
    H, W = img_np.shape
    f    = img_np.astype(np.float32)
    raw  = ((f >= low) & (f <= high)).astype(np.uint8)
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    cln  = cv2.morphologyEx(raw, cv2.MORPH_OPEN, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cln)
 
    dmg = np.zeros((H,W), dtype=np.uint8)
    kept, rejected = [], []
 
    for lbl in range(1, n):
        area  = int(stats[lbl, cv2.CC_STAT_AREA])
        bw    = int(stats[lbl, cv2.CC_STAT_WIDTH])
        bh    = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        if area < min_area:
            rejected.append((area, 0.0, f"too_small({area})"))
            continue
        aspect    = max(bw,bh)/(min(bw,bh)+1e-5)
        img_frac  = area/(H*W)
        diag_span = np.sqrt(bw**2+bh**2)/np.sqrt(H**2+W**2)
        blob      = (labels==lbl).astype(np.uint8)
        cnts,_    = cv2.findContours(blob,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        solidity  = 0.5
        if cnts:
            cnt=max(cnts,key=cv2.contourArea)
            ha=cv2.contourArea(cv2.convexHull(cnt))
            solidity=area/(ha+1e-5)
        score=0.; notes=[]
        if img_frac>0.025: score+=0.25; notes.append(f"large({img_frac:.2f})")
        if img_frac>0.06:  score+=0.15; notes.append("very_large")
        if aspect>2.5:     score+=0.20; notes.append(f"elongated({aspect:.1f}x)")
        if aspect>5.0:     score+=0.15; notes.append("very_elongated")
        if diag_span>0.30: score+=0.20; notes.append(f"spans({diag_span:.2f})")
        if diag_span>0.55: score+=0.15; notes.append("crosses_image")
        if 0.25<solidity<0.92: score+=0.10; notes.append(f"sol({solidity:.2f})")
        reason=" + ".join(notes) if notes else "no_match"
        if score>=0.45:
            dmg[labels==lbl]=1
            kept.append((area,score,reason))
        else:
            rejected.append((area,score,reason))
 
    print(f"   Components: {n-1} | kept: {len(kept)} | rejected: {len(rejected)}")
    for a,s,r in sorted(kept,reverse=True):
        print(f"   ✓ area={a:5d} score={s:.2f}  [{r}]")
 
    if dilation>0 and dmg.any():
        kd=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(dilation*2+1,dilation*2+1))
        dmg=cv2.dilate(dmg,kd)
 
    mask_f   = dmg.astype(np.float32)
    coverage = mask_f.mean()*100
    print(f"   Mask coverage: {coverage:.1f}%")
 
    if debug_dir:
        fig,axes=plt.subplots(1,4,figsize=(16,4))
        axes[0].imshow(img_np,cmap="gray");  axes[0].set_title("Input")
        axes[1].imshow(raw,   cmap="gray");  axes[1].set_title(f"Raw [{low},{high}]")
        colour=np.zeros((H,W,3),dtype=np.uint8); np.random.seed(42)
        for lbl in range(1,n): colour[labels==lbl]=np.random.randint(80,255,3)
        axes[2].imshow(colour);              axes[2].set_title("All blobs")
        ov=np.stack([img_np/255]*3,axis=-1).copy()
        ov[dmg>0]=[1.,.0,.2]
        axes[3].imshow(ov);                  axes[3].set_title(f"Mask (red) {coverage:.1f}%")
        for ax in axes: ax.axis("off")
        plt.suptitle(f"kept={len(kept)} low={low} high={high} dil={dilation}",fontweight="bold")
        plt.tight_layout()
        p=os.path.join(debug_dir,f"{stem}_DEBUG.png")
        plt.savefig(p,dpi=100,bbox_inches="tight"); plt.close(fig)
        print(f"   Debug: {p}")
 
    return mask_f
 
 
# ---------------------------------------------------------------------------
# SAVE FIGURE
# ---------------------------------------------------------------------------
 
def save_fig(path, cols, labels, title=""):
    n=len(cols); fig,axes=plt.subplots(1,n,figsize=(4*n,5))
    if n==1: axes=[axes]
    for i,(ax,img,lbl) in enumerate(zip(axes,cols,labels)):
        if isinstance(img,np.ndarray) and img.dtype==np.uint8: img=img/255.
        ax.imshow(img,cmap="gray",vmin=0,vmax=1)
        hi=(i==n-1)
        ax.set_title(lbl,fontsize=11,color="#00cc44" if hi else "white",
                     fontweight="bold" if hi else "normal",pad=6,backgroundcolor="#111111")
        ax.axis("off")
        if hi:
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_edgecolor("#00cc44"); sp.set_linewidth(3)
    plt.suptitle(title,fontsize=13,fontweight="bold",color="white",backgroundcolor="#222222")
    fig.patch.set_facecolor("#1a1a1a"); plt.tight_layout()
    plt.savefig(path,dpi=120,bbox_inches="tight",facecolor="#1a1a1a"); plt.close(fig)
    print(f"   Saved: {path}")
 
 
# ---------------------------------------------------------------------------
# MAIN RESTORE
# ---------------------------------------------------------------------------
 
def restore(G, img_path, out_dir, device, size=256,
            low=100, high=220, min_area=400, dilation=12,
            damage_type="overlay", debug=False):
 
    print(f"\n[RESTORE] {img_path}")
    stem   = Path(img_path).stem
    img_np = np.array(Image.open(img_path).convert("L")
                           .resize((size,size),Image.LANCZOS))
 
    dbg    = out_dir if debug else None
    mask_f = detect_damage(img_np, low, high, min_area, dilation, dbg, stem)
 
    if mask_f.max()==0:
        print("   ⚠  No damage detected. Try --debug, --low 80, --min-area 200")
        return
 
    # Build model input exactly as training did
    img_t, mask_t = prepare_model_input(img_np, mask_f, damage_type=damage_type)
    restored      = run_model(G, img_t, mask_t, device)
    out_img       = (restored*255).astype(np.uint8)
 
    clean = os.path.join(out_dir, f"{stem}_RESTORED.png")
    Image.fromarray(out_img).save(clean)
    print(f"   Restored: {clean}")
 
    save_fig(os.path.join(out_dir,f"{stem}_comparison.png"),
             [img_np/255., mask_f, restored],
             ["Input (Damaged)", "Auto Mask", "RESTORED ✓"],
             title=f"{stem}  —  Brahmi Restoration")
 
 
def batch(G, folder, out_dir, device, **kw):
    paths=[]
    for ext in ["*.png","*.jpg","*.jpeg","*.PNG","*.JPG","*.JPEG"]:
        paths.extend(Path(folder).glob(ext))
    if not paths: print("No images"); return
    for p in paths: restore(G,str(p),out_dir,device,**kw)
 
 
# ---------------------------------------------------------------------------
# ARGS
# ---------------------------------------------------------------------------
 
def parse_args():
    p=argparse.ArgumentParser(
        description="Brahmi Restoration — exact training preprocessing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
USAGE:
  # Your image has overlay-style damage (gray stroke over character):
  python brahmi_final.py --input "C:/path/image.png"
 
  # If damage is binary (complete erasure, black/white only):
  python brahmi_final.py --input image.png --damage-type binary
 
  # Debug mask:
  python brahmi_final.py --input image.png --debug
 
  # Batch:
  python brahmi_final.py --input "C:/folder/"
        """
    )
    p.add_argument("--input",       required=True)
    p.add_argument("--checkpoint",  default="epoch_0250.pth")
    p.add_argument("--out-dir",     default="inference_results")
    p.add_argument("--img-size",    type=int, default=256)
    p.add_argument("--damage-type", default="overlay",
                   choices=["overlay","binary"],
                   help="overlay: gray semi-transparent stroke (your image). "
                        "binary: complete erasure (black patches, cracks).")
    p.add_argument("--low",         type=int, default=100)
    p.add_argument("--high",        type=int, default=220)
    p.add_argument("--min-area",    type=int, default=400)
    p.add_argument("--dilation",    type=int, default=12)
    p.add_argument("--debug",       action="store_true")
    p.add_argument("--no-cuda",     action="store_true")
    return p.parse_args()
 
 
def main():
    args=parse_args()
    device=torch.device("cpu") if (args.no_cuda or not torch.cuda.is_available()) \
           else torch.device("cuda")
    if device.type=="cuda":
        props=torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} | {props.total_memory/1024**3:.1f} GB")
    os.makedirs(args.out_dir,exist_ok=True)
    G=load_model(args.checkpoint,device)
    inp=Path(args.input)
    kw=dict(size=args.img_size,low=args.low,high=args.high,
            min_area=args.min_area,dilation=args.dilation,
            damage_type=args.damage_type,debug=args.debug)
    if inp.is_dir(): batch(G,str(inp),args.out_dir,device,**kw)
    elif inp.is_file(): restore(G,str(inp),args.out_dir,device,**kw)
    else: print(f"Not found: {inp}"); sys.exit(1)
    print(f"\n✓ Done! Results in:\n  {args.out_dir}")
 
if __name__=="__main__":
    main()
 