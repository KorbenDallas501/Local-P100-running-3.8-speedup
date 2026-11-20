import os
import sys
sys.path.insert(0, os.path.join(os.environ.get('LLAMA_DIR', '.'), 'gguf-py'))
from gguf import GGUFReader
from gguf.constants import GGMLQuantizationType
r = GGUFReader(sys.argv[1])
out = []
for t in r.tensors:
    out.append(f"{t.name}: {GGMLQuantizationType(t.tensor_type).name}")
print("\n".join(out))
