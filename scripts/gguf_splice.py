#!/usr/bin/env python3
"""Byte-level GGUF tensor splice: assemble one model from tensors that already exist in two files.

WHY a splice and not a requant (F119): llama-quantize re-encodes EVERY tensor, so "leave this one
alone" is inexpressible (F87) -- which forced kept IQ2_S tensors to Q2_K (imatrix-free) and cost
+73.13 us/tensor at the operating width, cancelling the gain (F115). A splice copies each tensor
verbatim from whichever file already holds it at the wanted type:
  * no requantisation  -> NO imatrix needed
  * kept tensors stay BIT-IDENTICAL iq2_s -> no Q2_K penalty
  * upgraded tensors come from the ORIGINAL-weight IQ4_XS quantisation, not upquantised from
    iq2_s, so they are strictly better than Ridge-splice which requantised FROM the champion.

SAFETY (every item from the splice audit, all fatal):
  1. raw TYPE-ID equality, never the display label -- repacked variants share names, not layouts
  2. shape equality per tensor
  3. offsets/alignment REBUILT from scratch; no data-section copying
  4. metadata from ONE source (the champion) -- never merged
  5. per-tensor byte-length check against the type's own block arithmetic
  6. per-tensor SHA256 self-test after write, re-read from the OUTPUT file
  7. provenance guard: the donor must share upstream weights with the base (F121)
"""
import sys, os, hashlib, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.environ.get('LLAMA_DIR', '.'), 'gguf-py'))
from gguf import GGUFReader, GGUFWriter
from gguf.constants import GGMLQuantizationType, GGML_QUANT_SIZES

def sha(b): return hashlib.sha256(b).hexdigest()

def block_len(qtype, nelem):
    """Expected byte length from the type's block size/type size -- catches a truncated donor."""
    bs, ts = GGML_QUANT_SIZES[GGMLQuantizationType(qtype)]
    if nelem % bs: return None
    return nelem // bs * ts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, help='metadata + default tensors (the champion)')
    ap.add_argument('--donor', required=True, help='source of upgraded tensors (IQ4_XS)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--upgrade-list', required=True, help='one tensor name per line')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    want = [l.strip() for l in open(a.upgrade_list) if l.strip() and not l.startswith('#')]
    wantset = set(want)
    print(f"reading base  {a.base}");  rb = GGUFReader(a.base)
    print(f"reading donor {a.donor}"); rd = GGUFReader(a.donor)
    bt = {t.name: t for t in rb.tensors}
    dt = {t.name: t for t in rd.tensors}

    # ---- provenance guard (F121): f32 tensors are stored verbatim; derived ones (ssm_a) may
    # differ by one ULP because A = -exp(A_log) is computed, so compare those by tolerance.
    ident = diff = 0
    for n, t in bt.items():
        if int(t.tensor_type) != int(GGMLQuantizationType.F32): continue
        if n not in dt: continue
        A, B = np.asarray(t.data).ravel(), np.asarray(dt[n].data).ravel()
        if A.shape != B.shape: diff += 1; continue
        if A.tobytes() == B.tobytes(): ident += 1
        else:
            m = np.max(np.abs(A - B) / np.maximum(np.abs(A), 1e-9))
            if m <= 1e-6 and 'ssm_a' in n: ident += 1
            else:
                diff += 1
                print(f"  PROVENANCE MISMATCH {n}: max rel {m:.3e}")
    print(f"provenance: {ident} f32 tensors agree, {diff} disagree")
    if diff:
        sys.exit("ABORT: base and donor do not share upstream weights -- this would be model-mixing, not a precision map")

    # ---- validate every requested upgrade BEFORE writing anything
    plan, errs = [], []
    for n in want:
        if n not in bt: errs.append(f"{n}: absent from base")
        elif n not in dt: errs.append(f"{n}: absent from donor")
        elif list(bt[n].shape) != list(dt[n].shape):
            errs.append(f"{n}: shape {list(bt[n].shape)} vs donor {list(dt[n].shape)}")
        elif int(dt[n].tensor_type) != int(GGMLQuantizationType.IQ4_XS):
            errs.append(f"{n}: donor raw type-id is {int(dt[n].tensor_type)} "
                        f"({GGMLQuantizationType(dt[n].tensor_type).name}), not IQ4_XS(23)")
        else:
            nel = 1
            for d in dt[n].shape: nel *= int(d)
            exp = block_len(int(dt[n].tensor_type), nel)
            got = int(dt[n].data.nbytes)
            if exp is not None and exp != got:
                errs.append(f"{n}: donor bytes {got} != block arithmetic {exp}")
            else:
                plan.append(n)
    if errs:
        for e in errs[:20]: print("  ERROR", e)
        sys.exit(f"ABORT: {len(errs)} tensors failed validation; nothing written")
    print(f"validated {len(plan)}/{len(want)} upgrades")

    upg = sum(int(dt[n].data.nbytes) for n in plan)
    keep = sum(int(t.data.nbytes) for t in rb.tensors if t.name not in wantset)
    total = (upg + keep) / 1048576
    print(f"projected size: {total:.1f} MiB  (upgraded {upg/1048576:.1f} + kept {keep/1048576:.1f})")
    if a.dry_run:
        print("dry run -- nothing written"); return 0

    arch = rb.get_field('general.architecture')
    w = GGUFWriter(a.out, arch.contents() if arch else 'qwen35')
    # metadata: ONE source, the base. Copy every field verbatim except the architecture marker.
    ncopy = 0
    for key, fld in rb.fields.items():
        if key in ('GGUF.version','GGUF.tensor_count','GGUF.kv_count','general.architecture'): continue
        try:
            w.add_key_value(key, fld.contents(), fld.types[0]); ncopy += 1
        except Exception:
            pass
    print(f"metadata: {ncopy} fields copied from base only")
    for t in rb.tensors:
        src = dt[t.name] if t.name in wantset else t
        qt = GGMLQuantizationType(int(src.tensor_type))
        # For quantised tensors the reader hands back raw uint8; the writer expects the BYTE
        # shape there and converts to the logical shape itself. For f32 it wants the logical
        # shape. Passing the wrong one trips quant_shape_from_byte_shape.
        shp = list(src.data.shape) if src.data.dtype == np.uint8 else [int(x) for x in reversed(src.shape)]
        w.add_tensor_info(t.name, shp, src.data.dtype, int(src.data.nbytes), qt)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    exp_sha = {}
    for t in rb.tensors:
        src = dt[t.name] if t.name in wantset else t
        arr = src.data                      # writer wants an ndarray (uses .nbytes/.tofile)
        exp_sha[t.name] = sha(arr.tobytes())
        w.write_tensor_data(arr)
    w.close()
    print(f"wrote {a.out} ({os.path.getsize(a.out)/1048576:.1f} MiB)")

    # ---- self-test: re-read the OUTPUT and hash every tensor
    print("self-test: re-reading output and hashing every tensor...")
    ro = GGUFReader(a.out); bad = 0
    for t in ro.tensors:
        if sha(t.data.tobytes()) != exp_sha.get(t.name):
            print(f"  HASH MISMATCH {t.name}"); bad += 1
    if bad: sys.exit(f"ABORT: {bad} tensors failed the hash self-test -- output is NOT trustworthy")
    from collections import Counter
    c = Counter(GGMLQuantizationType(t.tensor_type).name for t in ro.tensors)
    print(f"self-test PASSED: {len(ro.tensors)} tensors, all hashes match")
    print("composition:", dict(c))
    return 0

if __name__ == '__main__':
    sys.exit(main())
