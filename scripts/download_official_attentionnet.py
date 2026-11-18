#!/usr/bin/env python3
"""Download the authors' pinned original checkpoint and export the exact evaluated state.

Only necessary when using the *lite* release or when the normalized exported
checkpoint was deliberately removed. The original model is THIRD-PARTY.
The 38MB official checkpoint must be accessible via github.com.
"""
from __future__ import annotations
import hashlib,os,sys,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'benchmark/weights/AttentionNet_published.pt'
SRC=ROOT/'benchmark/authors_official_checkpoint.pth'
URL='https://raw.githubusercontent.com/marmotlab/HeteroMRTA/main/model/save/checkpoint.pth'
EXPECTED_GIT_SHA='ced8f9cab156916dcfdc43ea0c76de3554dc5859'
EXPECTED_TENSOR_SHA256='319f7624c9922345c7b9798f6e0167dd94477c29af7c170af58d7d44bc0592cd'
EXPECTED_EXPORT_SHA256='dd9821f630613432704da6ced89f3fdeae6c91ba689016cb07e7ea3c36046bab'
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def git_sha(data):return hashlib.sha1(b'blob '+str(len(data)).encode('ascii')+b'\x00'+data).hexdigest()
def fingerprint_state(state):
    h=hashlib.sha256()
    for key in sorted(state):
        tensor=state[key].detach().cpu().contiguous()
        h.update((key+'|'+str(tuple(tensor.shape))+'|'+str(tensor.dtype)+'|').encode('utf8'))
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()
def main():
    if OUT.exists() and digest(OUT)==EXPECTED_EXPORT_SHA256:
        print('Original AttentionNet frozen export verified:',OUT);return
    if not SRC.exists() or git_sha(SRC.read_bytes())!=EXPECTED_GIT_SHA:
        print('Downloading the original pinned authors checkpoint ...',flush=True)
        SRC.parent.mkdir(parents=True,exist_ok=True)
        temp=SRC.with_suffix('.download')
        urllib.request.urlretrieve(URL,temp)
        if git_sha(temp.read_bytes())!=EXPECTED_GIT_SHA:
            temp.unlink(missing_ok=True)
            raise RuntimeError('Git blob checksum mismatch; refusing untrusted checkpoint')
        os.replace(temp,SRC)
    import torch
    try:ckpt=torch.load(SRC,map_location='cpu',weights_only=True)
    except Exception:
        # Only reached after checking exact expected original Git blob SHA1.
        ckpt=torch.load(SRC,map_location='cpu',weights_only=False)
    state=ckpt['best_model']
    if fingerprint_state(state)!=EXPECTED_TENSOR_SHA256:
        raise RuntimeError('State tensor SHA256 differs from published weights used in our study')
    OUT.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(weights=state,model_name='AttentionNet_published',source_checkpoint_field='best_model',source_episode=943104),OUT)
    if digest(OUT)!=EXPECTED_EXPORT_SHA256:
        raise RuntimeError('Export SHA256 mismatch; verify PyTorch serialization versions')
    print('Original frozen AttentionNet checkpoint exported and verified:',OUT)
if __name__=='__main__':main()
