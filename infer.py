from codec.generator import Generator
import torch
import torchaudio
torch.set_grad_enabled(False)

target_sr = 16000
token_hz = 50
model_path = "MagiCodec-50Hz-Base.ckpt"

model = Generator(
    sample_rate = target_sr,
    token_hz = token_hz,
)
state_dict = torch.load(model_path, map_location='cpu')
model.load_state_dict(state_dict, strict=False)

device = f"cuda:0"
model = model.to(device)
model.eval()

def preprocess(path):
    x, sr = torchaudio.load(path)
    x = x.to(device)
    x = x.mean(dim=0, keepdim=True)
    x = torchaudio.functional.resample(x, sr, target_sr)
    return x[None, ...]

def infer(path):
    x = preprocess(path)
    orig_length = x.shape[-1]

    recon, codes, zq = model.infer(x)
    recon = recon[..., :orig_length]
    return recon, codes

recon, codes = infer("audio/1580-141083-0000.flac")
torchaudio.save("recon.wav", recon[0].cpu(), target_sr)
torch.save(codes, "test_codes.pt")

def infer_from_code(codes, codebook):
    assert codes.ndim == 2   # (B, T)
    z_q = torch.nn.functional.embedding(codes, codebook)
    recon = model.decoder(z_q).float()
    return recon

with torch.autocast(
    device_type = "cuda",
    dtype = torch.bfloat16,
    enabled = True,
):
    codebook = model.quantizer.codebook_proj(model.quantizer.codebook.weight) 
    recon_from_code = infer_from_code(codes, codebook)

torchaudio.save("recon_from_code.wav", recon_from_code[0].cpu(), target_sr)

print((recon==recon_from_code).all().item())