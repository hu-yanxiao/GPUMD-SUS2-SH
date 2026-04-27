#!/usr/bin/env python3
"""Generate and compile model-specific SUS2 v1.1 CUDA product-graph kernels.

This is an experimental GPUMD-SUS2 tool. It does not change the mathematical
form of SUS2; it only bakes the model's topology (`alpha_index_basic`,
`alpha_index_times`, and `alpha_moment_mapping`) into CUDA source so we can test
whether a model-specific core is practical before wiring runtime dispatch into
GPUMD.
"""

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Tuple


class Sus2Topology:
    def __init__(
        self,
        model_path,
        version,
        radial_basis_type,
        scaling_map,
        species_count,
        L,
        radial_funcs_count,
        alpha_basic_count,
        alpha_times_count,
        alpha_moments_count,
        alpha_scalar_moments,
        alpha_basic,
        alpha_times,
        alpha_moment_mapping,
    ):
        self.model_path = model_path
        self.version = version
        self.radial_basis_type = radial_basis_type
        self.scaling_map = scaling_map
        self.species_count = species_count
        self.L = L
        self.radial_funcs_count = radial_funcs_count
        self.alpha_basic_count = alpha_basic_count
        self.alpha_times_count = alpha_times_count
        self.alpha_moments_count = alpha_moments_count
        self.alpha_scalar_moments = alpha_scalar_moments
        self.alpha_basic = alpha_basic
        self.alpha_times = alpha_times
        self.alpha_moment_mapping = alpha_moment_mapping


def value_after(text, name):
    match = re.search(rf"^\s*{re.escape(name)}\s*=\s*(.+)$", text, re.MULTILINE)
    if not match:
        raise ValueError(f"missing model key: {name}")
    return match.group(1).strip()


def braced_after(text, token):
    start = text.find(token)
    if start < 0:
        raise ValueError(f"missing model token: {token}")
    brace = text.find("{", start)
    if brace < 0:
        raise ValueError(f"missing '{{' after model token: {token}")
    depth = 0
    group_start = brace
    for i in range(brace, len(text)):
        if text[i] == "{":
            if depth == 0:
                group_start = i
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[group_start + 1 : i]
    raise ValueError(f"unbalanced braces after model token: {token}")


def parse_ints(text):
    return [int(x) for x in re.findall(r"[-+]?\d+", text)]


def load_topology(path):
    text = path.read_text()
    topo = Sus2Topology(
        model_path=path,
        version=value_after(text, "version"),
        radial_basis_type=value_after(text, "radial_basis_type"),
        scaling_map=value_after(text, "scaling_map"),
        species_count=int(value_after(text, "species_count")),
        L=int(value_after(text, "L")),
        radial_funcs_count=int(value_after(text, "radial_funcs_count")),
        alpha_basic_count=int(value_after(text, "alpha_index_basic_count")),
        alpha_times_count=int(value_after(text, "alpha_index_times_count")),
        alpha_moments_count=int(value_after(text, "alpha_moments_count")),
        alpha_scalar_moments=int(value_after(text, "alpha_scalar_moments")),
        alpha_basic=parse_ints(braced_after(text, "alpha_index_basic =")),
        alpha_times=parse_ints(braced_after(text, "alpha_index_times =")),
        alpha_moment_mapping=parse_ints(braced_after(text, "alpha_moment_mapping =")),
    )
    if len(topo.alpha_basic) != 4 * topo.alpha_basic_count:
        raise ValueError("alpha_index_basic_count does not match parsed rows")
    if len(topo.alpha_times) != 4 * topo.alpha_times_count:
        raise ValueError("alpha_index_times_count does not match parsed rows")
    if len(topo.alpha_moment_mapping) != topo.alpha_scalar_moments:
        raise ValueError("alpha_scalar_moments does not match parsed mapping")
    return topo


def compress_active_dag(topo):
    needed = [False] * topo.alpha_moments_count

    def require(moment_id: int, section: str) -> None:
        if moment_id < 0 or moment_id >= topo.alpha_moments_count:
            raise ValueError(f"invalid moment id {moment_id} in {section}")
        needed[moment_id] = True

    for basic in range(topo.alpha_basic_count):
        require(basic, "alpha_index_basic")
    for moment_id in topo.alpha_moment_mapping:
        require(moment_id, "alpha_moment_mapping")

    changed = True
    while changed:
        changed = False
        for t in range(topo.alpha_times_count - 1, -1, -1):
            src0, src1, _mult, dst = topo.alpha_times[4 * t : 4 * t + 4]
            require(src0, "alpha_index_times")
            require(src1, "alpha_index_times")
            require(dst, "alpha_index_times")
            if not needed[dst]:
                continue
            if not needed[src0]:
                needed[src0] = True
                changed = True
            if not needed[src1]:
                needed[src1] = True
                changed = True

    old_to_new = [-1] * topo.alpha_moments_count
    active_count = 0
    for old_id, is_needed in enumerate(needed):
        if is_needed:
            old_to_new[old_id] = active_count
            active_count += 1

    for basic in range(topo.alpha_basic_count):
        if old_to_new[basic] != basic:
            raise ValueError("compressed DAG expected contiguous basic moments")

    new_times = []
    for t in range(topo.alpha_times_count):
        src0, src1, mult, dst = topo.alpha_times[4 * t : 4 * t + 4]
        if needed[dst]:
            new_times.extend([old_to_new[src0], old_to_new[src1], mult, old_to_new[dst]])

    new_mapping = [old_to_new[m] for m in topo.alpha_moment_mapping]
    return new_times, new_mapping, active_count


def detect_lk_basic_layout(topo):
    L = topo.L
    channels = L + 1
    if topo.radial_funcs_count % channels != 0:
        return {"matches": False}
    k_count = topo.radial_funcs_count // channels
    expected = []
    for group in range(k_count):
        for rank in range(channels):
            mu = group * channels + rank
            for a in range(rank, -1, -1):
                for b in range(rank - a, -1, -1):
                    c = rank - a - b
                    expected.extend([mu, a, b, c])
    return {
        "matches": expected == topo.alpha_basic,
        "L": L,
        "k_count": k_count,
        "expected_basic_count": len(expected) // 4,
    }


def topology_hash(topo, times, mapping, active_count):
    payload = {
        "codegen_cache_schema": "sus2_v11_product_graph_v1",
        "kernel_scope": "alpha_basic_times_scalar_mapping",
        "version": topo.version,
        "scaling_map": topo.scaling_map,
        "L": topo.L,
        "radial_funcs_count": topo.radial_funcs_count,
        "alpha_basic": topo.alpha_basic,
        "alpha_times_compressed": times,
        "alpha_moment_mapping_compressed": mapping,
        "active_moments_count": active_count,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def try_link_or_copy(src, dst):
    if dst.exists():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except Exception:
        shutil.copy2(str(src), str(dst))


def cache_paths(cache_dir, digest):
    entry_dir = cache_dir / digest
    return {
        "entry_dir": entry_dir,
        "source": entry_dir / "generated.cu",
        "object": entry_dir / "generated.o",
        "metadata": entry_dir / "metadata.json",
        "build_log": entry_dir / "build.log",
    }


def format_forward_rule(src0, src1, mult, dst):
    return (
        "  moments[(size_t){dst} * N + i] += {mult}.0 * "
        "moments[(size_t){src0} * N + i] * moments[(size_t){src1} * N + i];".format(
            src0=src0, src1=src1, mult=mult, dst=dst
        )
    )


def format_backward_rule(src0, src1, mult, dst):
    return (
        "  {{ const double gdst = grads[(size_t){dst} * N + i] * {mult}.0; "
        "grads[(size_t){src1} * N + i] += gdst * moments[(size_t){src0} * N + i]; "
        "grads[(size_t){src0} * N + i] += gdst * moments[(size_t){src1} * N + i]; }}".format(
            src0=src0, src1=src1, mult=mult, dst=dst
        )
    )


def emit_cuda_source(topo, times, digest, chunk_size):
    symbol = f"sus2_v11_topo_{digest[:16]}"
    if chunk_size > 0:
        return emit_chunked_cuda_source(times, symbol, chunk_size)

    lines = [
        "// Generated by tools/sus2_v11_codegen.py; do not edit by hand.",
        "#include <stddef.h>",
        "extern \"C\" __global__",
        f"void {symbol}_forward(int N, double* __restrict__ moments) {{",
        "  const int i = blockIdx.x * blockDim.x + threadIdx.x;",
        "  if (i >= N) return;",
    ]
    for t in range(len(times) // 4):
        src0, src1, mult, dst = times[4 * t : 4 * t + 4]
        lines.append(format_forward_rule(src0, src1, mult, dst))
    lines.extend(
        [
            "}",
            "",
            "extern \"C\" __global__",
            f"void {symbol}_backward(int N, const double* __restrict__ moments, double* __restrict__ grads) {{",
            "  const int i = blockIdx.x * blockDim.x + threadIdx.x;",
            "  if (i >= N) return;",
        ]
    )
    for t in range(len(times) // 4 - 1, -1, -1):
        src0, src1, mult, dst = times[4 * t : 4 * t + 4]
        lines.append(format_backward_rule(src0, src1, mult, dst))
    lines.append("}")
    return "\n".join(lines) + "\n"


def emit_chunked_cuda_source(times, symbol, chunk_size):
    total_rules = len(times) // 4
    chunks = [(start, min(start + chunk_size, total_rules)) for start in range(0, total_rules, chunk_size)]
    lines = [
        "// Generated by tools/sus2_v11_codegen.py; do not edit by hand.",
        "#include <stddef.h>",
    ]

    for idx, (start, stop) in enumerate(chunks):
        lines.extend(
            [
                "__device__ __noinline__",
                f"void {symbol}_forward_chunk_{idx:03d}(int N, int i, double* __restrict__ moments) {{",
            ]
        )
        for t in range(start, stop):
            src0, src1, mult, dst = times[4 * t : 4 * t + 4]
            lines.append(format_forward_rule(src0, src1, mult, dst))
        lines.append("}")
        lines.append("")

    for idx, (start, stop) in enumerate(chunks):
        lines.extend(
            [
                "__device__ __noinline__",
                f"void {symbol}_backward_chunk_{idx:03d}(int N, int i, const double* __restrict__ moments, double* __restrict__ grads) {{",
            ]
        )
        for t in range(stop - 1, start - 1, -1):
            src0, src1, mult, dst = times[4 * t : 4 * t + 4]
            lines.append(format_backward_rule(src0, src1, mult, dst))
        lines.append("}")
        lines.append("")

    lines.extend(
        [
            "extern \"C\" __global__",
            f"void {symbol}_forward(int N, double* __restrict__ moments) {{",
            "  const int i = blockIdx.x * blockDim.x + threadIdx.x;",
            "  if (i >= N) return;",
        ]
    )
    for idx in range(len(chunks)):
        lines.append(f"  {symbol}_forward_chunk_{idx:03d}(N, i, moments);")
    lines.extend(
        [
            "}",
            "",
            "extern \"C\" __global__",
            f"void {symbol}_backward(int N, const double* __restrict__ moments, double* __restrict__ grads) {{",
            "  const int i = blockIdx.x * blockDim.x + threadIdx.x;",
            "  if (i >= N) return;",
        ]
    )
    for idx in range(len(chunks) - 1, -1, -1):
        lines.append(f"  {symbol}_backward_chunk_{idx:03d}(N, i, moments, grads);")
    lines.append("}")
    return "\n".join(lines) + "\n"


def compile_cuda(source_path, out_path, nvcc, arch, progress_interval):
    cmd = [nvcc, f"-arch={arch}", "-O3", "-c", str(source_path), "-o", str(out_path)]
    start = time.time()
    proc = subprocess.Popen(cmd)
    spinner = "|/-\\"
    tick = 0
    last_report = 0.0
    print("[sus2-codegen] compile command: " + " ".join(cmd), file=sys.stderr, flush=True)
    while True:
        rc = proc.poll()
        now = time.time()
        elapsed = now - start
        if rc is not None:
            break
        if progress_interval > 0 and elapsed - last_report >= progress_interval:
            phase = spinner[tick % len(spinner)]
            filled = (tick % 20) + 1
            bar = "#" * filled + "." * (20 - filled)
            print(
                "[sus2-codegen] nvcc/ptxas working {phase} [{bar}] elapsed={elapsed:.1f}s".format(
                    phase=phase, bar=bar, elapsed=elapsed
                ),
                file=sys.stderr,
                flush=True,
            )
            tick += 1
            last_report = elapsed
        time.sleep(0.5)
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)
    print(
        "[sus2-codegen] compile finished in {elapsed:.2f}s".format(elapsed=time.time() - start),
        file=sys.stderr,
        flush=True,
    )
    return time.time() - start


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--arch", default="sm_80")
    parser.add_argument(
        "--cache-dir",
        type=pathlib.Path,
        default=None,
        help="Cache compiled topology cores by hash. Default: $SUS2_CODEGEN_CACHE_DIR or codegen_cache/sus2_v11.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Ignore a matching cached object and rebuild it.",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=5.0,
        help="Print an activity progress line while nvcc/ptxas is compiling. Use 0 to disable.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="Emit noinline device chunks with this many product rules per chunk. Default 0 emits one fully unrolled kernel.",
    )
    args = parser.parse_args(argv)

    topo = load_topology(args.model)
    times, mapping, active_count = compress_active_dag(topo)
    digest = topology_hash(topo, times, mapping, active_count)
    layout = detect_lk_basic_layout(topo)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    source_path = args.out_dir / f"sus2_v11_topo_{digest[:16]}.cu"
    object_path = args.out_dir / f"sus2_v11_topo_{digest[:16]}.o"
    meta_path = args.out_dir / f"sus2_v11_topo_{digest[:16]}.json"
    cache_dir = args.cache_dir
    if cache_dir is None:
        env_cache = os.environ.get("SUS2_CODEGEN_CACHE_DIR")
        cache_dir = pathlib.Path(env_cache) if env_cache else pathlib.Path("codegen_cache/sus2_v11")
    cpaths = cache_paths(cache_dir, digest)

    metadata = {
        "model": str(topo.model_path),
        "hash": digest,
        "cache_key": digest,
        "cache_dir": str(cpaths["entry_dir"]),
        "cache_hit": False,
        "source": str(source_path),
        "object": str(object_path),
        "version": topo.version,
        "radial_basis_type": topo.radial_basis_type,
        "scaling_map": topo.scaling_map,
        "species_count": topo.species_count,
        "L": topo.L,
        "radial_funcs_count": topo.radial_funcs_count,
        "alpha_basic_count": topo.alpha_basic_count,
        "alpha_times_count_original": topo.alpha_times_count,
        "alpha_times_count_compressed": len(times) // 4,
        "alpha_moments_count_original": topo.alpha_moments_count,
        "alpha_moments_count_compressed": active_count,
        "alpha_scalar_moments": topo.alpha_scalar_moments,
        "lk_basic_layout": layout,
        "codegen_chunk_size": args.chunk_size,
        "expected_compile_seconds_when_cache_miss": 160,
    }

    source_text = None
    if args.compile:
        cache_ready = (
            cpaths["source"].exists()
            and cpaths["object"].exists()
            and cpaths["metadata"].exists()
            and not args.force_rebuild
        )
        if cache_ready:
            print(
                "[sus2-codegen] cache hit: {path}".format(path=cpaths["entry_dir"]),
                file=sys.stderr,
                flush=True,
            )
            try_link_or_copy(cpaths["source"], source_path)
            try_link_or_copy(cpaths["object"], object_path)
            cached_metadata = json.loads(cpaths["metadata"].read_text())
            metadata["cache_hit"] = True
            metadata["cached_compile_seconds"] = cached_metadata.get("compile_seconds")
            metadata["compile_seconds"] = 0.0
            metadata["cuda_source_bytes"] = source_path.stat().st_size
            metadata["object_bytes"] = object_path.stat().st_size
        else:
            print(
                "[sus2-codegen] cache miss: building topology core; expected compile time is about 160 s.",
                file=sys.stderr,
                flush=True,
            )
            cpaths["entry_dir"].mkdir(parents=True, exist_ok=True)
            source_text = emit_cuda_source(topo, times, digest, args.chunk_size)
            cpaths["source"].write_text(source_text)
            elapsed = compile_cuda(
                cpaths["source"], cpaths["object"], args.nvcc, args.arch, args.progress_interval
            )
            metadata["compile_seconds"] = elapsed
            metadata["cuda_source_bytes"] = cpaths["source"].stat().st_size
            metadata["object_bytes"] = cpaths["object"].stat().st_size
            cache_metadata = dict(metadata)
            cache_metadata["source"] = str(cpaths["source"])
            cache_metadata["object"] = str(cpaths["object"])
            cache_metadata["build_log"] = str(cpaths["build_log"])
            cpaths["metadata"].write_text(json.dumps(cache_metadata, indent=2, sort_keys=True) + "\n")
            cpaths["build_log"].write_text(
                "model={model}\nhash={hash}\narch={arch}\nchunk_size={chunk}\ncompile_seconds={seconds:.6f}\n".format(
                    model=topo.model_path,
                    hash=digest,
                    arch=args.arch,
                    chunk=args.chunk_size,
                    seconds=elapsed,
                )
            )
            try_link_or_copy(cpaths["source"], source_path)
            try_link_or_copy(cpaths["object"], object_path)
    else:
        source_text = emit_cuda_source(topo, times, digest, args.chunk_size)
        source_path.write_text(source_text)
        metadata["cuda_source_bytes"] = source_path.stat().st_size

    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
