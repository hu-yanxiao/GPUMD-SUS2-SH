#!/usr/bin/env python3
"""Validate SUS2-SH model metadata consumed by the GPUMD backend."""

import argparse
import re
from pathlib import Path


def scalar_int(text, key):
    match = re.search(r"^\s*" + re.escape(key) + r"\s*=\s*([-+]?\d+)", text, re.M)
    if not match:
        raise ValueError("missing {}".format(key))
    return int(match.group(1))


def braced_after(text, key):
    pos = text.find(key)
    if pos < 0:
        raise ValueError("missing {}".format(key))
    start = text.find("{", pos)
    if start < 0:
        raise ValueError("missing braced section for {}".format(key))
    depth = 0
    for idx in range(start, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:idx]
    raise ValueError("unterminated braced section for {}".format(key))


def parse_ints(section):
    return [int(value) for value in re.findall(r"[-+]?\d+", section)]


def product_groups(section):
    groups = []
    depth = 0
    start = None
    for idx, char in enumerate(section):
        if char == "{":
            if depth == 0:
                start = idx + 1
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                groups.append(section[start:idx])
                start = None
    if depth != 0:
        raise ValueError("unterminated sh_products section")
    return groups


def parse_products(section):
    products = []
    number_pattern = r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"
    for group in product_groups(section):
        values = re.findall(number_pattern, group)
        if len(values) != 4:
            raise ValueError("invalid sh_products entry: {}".format(group))
        products.append((
            int(float(values[0])),
            int(float(values[1])),
            int(float(values[2])),
            float(values[3]),
        ))
    return products


def validate_model(path):
    resolved = path.resolve()
    text = resolved.read_text()
    lmax = scalar_int(text, "sh_l_max")
    kmax = scalar_int(text, "sh_k_max")
    rb_size = scalar_int(text, "radial_basis_size")
    moments = scalar_int(text, "alpha_moments_count")
    basics = scalar_int(text, "alpha_index_basic_count")
    scalars = scalar_int(text, "alpha_scalar_moments")
    product_count = scalar_int(text, "sh_product_count")

    basic_raw = parse_ints(braced_after(text, "alpha_index_basic"))
    mapping = parse_ints(braced_after(text, "alpha_moment_mapping"))
    products = parse_products(braced_after(text, "sh_products"))

    expected_basics = kmax * (lmax + 1) * (lmax + 1)
    full_layout = basics == expected_basics and len(basic_raw) == 3 * basics and rb_size == 10
    index = 0
    if full_layout:
        for l in range(lmax, -1, -1):
            for k in range(kmax - 1, -1, -1):
                for m in range(-l, l + 1):
                    if basic_raw[3 * index:3 * index + 3] != [k, l, m]:
                        full_layout = False
                        break
                    index += 1
                if not full_layout:
                    break
            if not full_layout:
                break

    last_definition = [-1] * moments
    invalid_products = []
    terms_by_target = [0] * moments
    for product_index, (left, right, target, _coeff) in enumerate(products):
        if min(left, right, target) < 0 or max(left, right, target) >= moments:
            invalid_products.append((product_index, "range", left, right, target))
        if target < basics:
            invalid_products.append((product_index, "basic_write", left, right, target))
        last_definition[target] = product_index
        terms_by_target[target] += 1

    topo_bad = []
    defined = [False] * moments
    for basic in range(basics):
        defined[basic] = True
    node_layer = [0] * moments
    rows = set()
    max_layer = 0
    for product_index, (left, right, target, _coeff) in enumerate(products):
        if left >= basics and last_definition[left] >= product_index:
            topo_bad.append((product_index, "left", left, last_definition[left]))
        if right >= basics and last_definition[right] >= product_index:
            topo_bad.append((product_index, "right", right, last_definition[right]))
        layer = max(node_layer[left], node_layer[right]) + 1
        node_layer[target] = max(node_layer[target], layer)
        max_layer = max(max_layer, node_layer[target])
        defined[target] = True
        rows.add(target)

    mapping_bad = [moment for moment in mapping
                   if moment < 0 or moment >= moments or not defined[moment]]
    back_terms = sum(2 * terms for terms in terms_by_target if terms)

    return {
        "path": str(resolved),
        "lmax": lmax,
        "kmax": kmax,
        "rb_size": rb_size,
        "basics": basics,
        "expected_basics": expected_basics,
        "products": len(products),
        "expected_products": product_count,
        "moments": moments,
        "scalars": scalars,
        "mapping_count": len(mapping),
        "static_full_layout_guard": full_layout,
        "rows": len(rows),
        "max_layer": max_layer,
        "back_terms": back_terms,
        "product_index_invalid": len(invalid_products),
        "topo_bad": len(topo_bad),
        "mapping_bad": len(mapping_bad),
    }


def print_report(label, report):
    guard = "PASS" if report["static_full_layout_guard"] else "FAIL"
    print("[{}] file={}".format(label, report["path"]))
    print(
        "  lmax={lmax} kmax={kmax} rb={rb_size} basics={basics}/{expected_basics} "
        "products={expected_products}/{products} moments={moments} "
        "scalars={scalars}/{mapping_count}".format(**report)
    )
    print(
        "  static_full_layout_guard={} rows={} max_layer={} back_terms={}".format(
            guard, report["rows"], report["max_layer"], report["back_terms"])
    )
    print(
        "  product_index_invalid={} topo_bad={} mapping_bad={}".format(
            report["product_index_invalid"], report["topo_bad"], report["mapping_bad"])
    )


def main():
    parser = argparse.ArgumentParser(
        description="Validate SUS2-SH metadata used by GPUMD explicit graph loading.")
    parser.add_argument("models", nargs="+", help="SUS2-SH .mtp files to validate")
    args = parser.parse_args()

    failed = False
    for model in args.models:
        path = Path(model)
        label = path.parent.name if path.name == "p.mtp" else path.stem
        report = validate_model(path)
        print_report(label, report)
        if (report["product_index_invalid"] or report["topo_bad"] or
                report["mapping_bad"] or
                report["products"] != report["expected_products"]):
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
