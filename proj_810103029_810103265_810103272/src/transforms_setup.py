from torchvision import transforms
from torchvision.transforms import functional as TF, InterpolationMode
from PIL import Image
try:
    import importlib
    importlib.import_module("sympy.printing")  # ensures attribute exists on the sympy package
except Exception:
    # last-resort: install a known-good version quietly, then import again
    try:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "sympy==1.12"])
        importlib.invalidate_caches()
        importlib.import_module("sympy.printing")
    except Exception as e:
        print(f"⚠️ Could not ensure sympy.printing is available: {e}")
        
class Letterbox:
    """Resize keeping aspect ratio, then pad to target size with a constant color."""
    def __init__(self, size, fill=255, interpolation=InterpolationMode.BILINEAR):
        self.size = (size, size) if isinstance(size, int) else size
        self.fill = fill
        self.interpolation = interpolation

    def __call__(self, img):
        W, H = img.size
        th, tw = self.size
        scale = min(tw / W, th / H)
        nw, nh = int(round(W * scale)), int(round(H * scale))
        img = TF.resize(img, (nh, nw), interpolation=self.interpolation)
        pad_left   = (tw - nw) // 2
        pad_right  = tw - nw - pad_left
        pad_top    = (th - nh) // 2
        pad_bottom = th - nh - pad_top
        return TF.pad(img, [pad_left, pad_top, pad_right, pad_bottom], fill=self.fill)

train_transform = transforms.Compose([
    transforms.RandomRotation(10, fill=255),
    transforms.RandomAffine(
        degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), fill=255
    ),
    Letterbox(224, fill=255),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

val_transform = transforms.Compose([
    Letterbox(224, fill=255),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
