from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict
import json
from pathlib import Path

@dataclass
class LinguisticProbe:
    text: str
    expected_label: str
    phenomenon: str
    phenomenon_tokens: List[str]
    description: str
    minimal_pair: Optional[str] = None
    source: str = "manual"

PHENOMENON_CATEGORIES = {
    "negation":           "Lexical negation (tidak, bukan, tak, belum, jangan)",
    "intensifier":        "Degree intensifiers (sangat, sekali, banget, amat)",
    "reduplication":      "Morphological reduplication (XXX-XXX forms)",
    "contrastive":        "Contrastive discourse markers (tapi, namun, walaupun)",
    "double_negation":    "Double negation resolving to positive",
    "scope_negation":     "Scope-sensitive negation (tidak semua vs semua tidak)",
    "hedging":            "Epistemic hedging (mungkin, sepertinya, kayaknya)",
}

PROBE_SET: List[LinguisticProbe] = [

    #  NEGATION 
    LinguisticProbe(
        text="Produk ini tidak bagus sama sekali.",
        expected_label="NEGATIVE",
        phenomenon="negation",
        phenomenon_tokens=["tidak"],
        description="'tidak bagus' = not good. Negation inverts positive adjective.",
        minimal_pair="Produk ini bagus sama sekali.",
    ),
    LinguisticProbe(
        text="Pelayanannya tidak buruk, cukup memuaskan.",
        expected_label="POSITIVE",
        phenomenon="negation",
        phenomenon_tokens=["tidak"],
        description="Double negation: 'tidak buruk' (not bad) resolves to positive.",
        minimal_pair="Pelayanannya buruk, tidak memuaskan.",
    ),
    LinguisticProbe(
        text="Saya belum puas dengan hasilnya.",
        expected_label="NEGATIVE",
        phenomenon="negation",
        phenomenon_tokens=["belum"],
        description="'belum puas' = not yet satisfied. Temporal negation.",
        minimal_pair="Saya puas dengan hasilnya.",
    ),
    LinguisticProbe(
        text="Bukan produk yang saya harapkan.",
        expected_label="NEGATIVE",
        phenomenon="negation",
        phenomenon_tokens=["bukan"],
        description="'bukan' negates noun phrase identification.",
        minimal_pair="Produk yang saya harapkan.",
    ),
    LinguisticProbe(
        text="Tak ada yang salah dengan pelayanan mereka.",
        expected_label="POSITIVE",
        phenomenon="negation",
        phenomenon_tokens=["tak"],
        description="'tak ada yang salah' = nothing wrong. Negative existential → positive.",
        minimal_pair="Ada yang salah dengan pelayanan mereka.",
    ),
    LinguisticProbe(
        text="Jangan beli produk ini, kualitasnya jelek.",
        expected_label="NEGATIVE",
        phenomenon="negation",
        phenomenon_tokens=["jangan"],
        description="Imperative negation 'jangan beli' signals strong negative recommendation.",
        minimal_pair="Beli produk ini, kualitasnya bagus.",
    ),

    #  INTENSIFIERS 
    LinguisticProbe(
        text="Produk ini sangat memuaskan.",
        expected_label="POSITIVE",
        phenomenon="intensifier",
        phenomenon_tokens=["sangat"],
        description="'sangat memuaskan' = very satisfying. Standard intensifier.",
        minimal_pair="Produk ini memuaskan.",
    ),
    LinguisticProbe(
        text="Kualitasnya buruk sekali.",
        expected_label="NEGATIVE",
        phenomenon="intensifier",
        phenomenon_tokens=["sekali"],
        description="Postfix intensifier 'sekali' amplifies 'buruk'.",
        minimal_pair="Kualitasnya buruk.",
    ),
    LinguisticProbe(
        text="Saya suka banget sama produk ini.",
        expected_label="POSITIVE",
        phenomenon="intensifier",
        phenomenon_tokens=["banget"],
        description="Colloquial intensifier 'banget' (very/really) — informal register.",
        minimal_pair="Saya suka sama produk ini.",
    ),
    LinguisticProbe(
        text="Pengirimannya lambat banget, mengecewakan.",
        expected_label="NEGATIVE",
        phenomenon="intensifier",
        phenomenon_tokens=["banget"],
        description="'lambat banget' = very slow. Intensifier on negative adjective.",
        minimal_pair="Pengirimannya lambat, mengecewakan.",
    ),
    LinguisticProbe(
        text="Amat mengecewakan, tidak sesuai deskripsi.",
        expected_label="NEGATIVE",
        phenomenon="intensifier",
        phenomenon_tokens=["amat"],
        description="'amat mengecewakan' = extremely disappointing. Formal intensifier.",
        minimal_pair="Mengecewakan, tidak sesuai deskripsi.",
    ),
    LinguisticProbe(
        text="Benar-benar produk yang luar biasa.",
        expected_label="POSITIVE",
        phenomenon="intensifier",
        phenomenon_tokens=["benar-benar"],
        description="Reduplicated intensifier 'benar-benar' = truly/really.",
        minimal_pair="Produk yang luar biasa.",
    ),

    #  REDUPLICATION 
    LinguisticProbe(
        text="Barang ini baik-baik saja kondisinya.",
        expected_label="POSITIVE",
        phenomenon="reduplication",
        phenomenon_tokens=["baik-baik"],
        description="'baik-baik saja' = just fine. Reduplication conveys general adequacy.",
        minimal_pair="Barang ini baik kondisinya.",
    ),
    LinguisticProbe(
        text="Lambat-laun saya menjadi tidak puas dengan layanan ini.",
        expected_label="NEGATIVE",
        phenomenon="reduplication",
        phenomenon_tokens=["lambat-laun"],
        description="'lambat-laun' = gradually. Aspectual reduplication marking temporal gradience.",
        minimal_pair="Saya menjadi tidak puas dengan layanan ini.",
    ),
    LinguisticProbe(
        text="Tolong-menolong antara penjual dan pembeli menciptakan pengalaman positif.",
        expected_label="POSITIVE",
        phenomenon="reduplication",
        phenomenon_tokens=["tolong-menolong"],
        description="Reciprocal reduplication 'tolong-menolong' = mutual helping.",
        minimal_pair="Bantuan antara penjual dan pembeli menciptakan pengalaman positif.",
    ),
    LinguisticProbe(
        text="Produknya biasa-biasa saja, tidak ada yang istimewa.",
        expected_label="NEUTRAL",
        phenomenon="reduplication",
        phenomenon_tokens=["biasa-biasa"],
        description="'biasa-biasa saja' = ordinary/mediocre. Core neutral expression in Indonesian.",
        minimal_pair="Produknya biasa saja, tidak ada yang istimewa.",
    ),

    #  CONTRASTIVE DISCOURSE 
    LinguisticProbe(
        text="Harga mahal tapi kualitasnya sepadan.",
        expected_label="POSITIVE",
        phenomenon="contrastive",
        phenomenon_tokens=["tapi"],
        description="'tapi' introduces contrast. Final clause (positive) should dominate attribution.",
        minimal_pair="Harga mahal dan kualitasnya sepadan.",
    ),
    LinguisticProbe(
        text="Pengiriman cepat, namun produk rusak.",
        expected_label="NEGATIVE",
        phenomenon="contrastive",
        phenomenon_tokens=["namun"],
        description="'namun' (however) shifts from positive to negative. Final clause dominates.",
        minimal_pair="Pengiriman cepat dan produk rusak.",
    ),
    LinguisticProbe(
        text="Walaupun mahal, kualitasnya sangat bagus.",
        expected_label="POSITIVE",
        phenomenon="contrastive",
        phenomenon_tokens=["walaupun"],
        description="'walaupun' (although) concedes a negative to assert a positive.",
        minimal_pair="Kualitasnya sangat bagus.",
    ),
    LinguisticProbe(
        text="Meskipun tampilannya menarik, fungsinya mengecewakan.",
        expected_label="NEGATIVE",
        phenomenon="contrastive",
        phenomenon_tokens=["meskipun"],
        description="'meskipun' (even though) concedes a positive to assert a negative.",
        minimal_pair="Tampilannya menarik tapi fungsinya mengecewakan.",
    ),

    #  DOUBLE NEGATION 
    LinguisticProbe(
        text="Tidak ada yang tidak suka dengan produk ini.",
        expected_label="POSITIVE",
        phenomenon="double_negation",
        phenomenon_tokens=["tidak", "tidak"],
        description="Double negation 'tidak ada yang tidak suka' = everyone likes it.",
        minimal_pair="Semua orang suka dengan produk ini.",
    ),
    LinguisticProbe(
        text="Bukan berarti produk ini tidak berguna.",
        expected_label="POSITIVE",
        phenomenon="double_negation",
        phenomenon_tokens=["bukan", "tidak"],
        description="'bukan berarti tidak berguna' = doesn't mean it's useless → useful.",
        minimal_pair="Produk ini berguna.",
    ),
    LinguisticProbe(
        text="Tidak bisa dibilang tidak memuaskan.",
        expected_label="POSITIVE",
        phenomenon="double_negation",
        phenomenon_tokens=["tidak", "tidak"],
        description="'tidak bisa dibilang tidak memuaskan' = can't be called unsatisfying → satisfying.",
        minimal_pair="Bisa dibilang memuaskan.",
    ),

    #  SCOPE-SENSITIVE NEGATION 
    LinguisticProbe(
        text="Tidak semua produk di sini berkualitas bagus.",
        expected_label="NEUTRAL",
        phenomenon="scope_negation",
        phenomenon_tokens=["tidak", "semua"],
        description="'tidak semua' = not all. Partial negation → neutral, some are good.",
        minimal_pair="Semua produk di sini berkualitas bagus.",
    ),
    LinguisticProbe(
        text="Semua produk di sini tidak berkualitas bagus.",
        expected_label="NEGATIVE",
        phenomenon="scope_negation",
        phenomenon_tokens=["semua", "tidak"],
        description="'semua tidak berkualitas' = none are high quality. Universal negation → negative.",
        minimal_pair="Produk di sini tidak berkualitas bagus.",
    ),

    #  EPISTEMIC HEDGING 
    LinguisticProbe(
        text="Mungkin produk ini cocok untuk sebagian orang.",
        expected_label="NEUTRAL",
        phenomenon="hedging",
        phenomenon_tokens=["mungkin"],
        description="'mungkin' (maybe) hedges the positive claim → uncertain/neutral.",
        minimal_pair="Produk ini cocok untuk sebagian orang.",
    ),
    LinguisticProbe(
        text="Sepertinya ada masalah dengan produk ini.",
        expected_label="NEGATIVE",
        phenomenon="hedging",
        phenomenon_tokens=["sepertinya"],
        description="'sepertinya' (it seems) hedges but doesn't cancel the negative.",
        minimal_pair="Ada masalah dengan produk ini.",
    ),
    LinguisticProbe(
        text="Kayaknya lumayan bagus deh.",
        expected_label="POSITIVE",
        phenomenon="hedging",
        phenomenon_tokens=["kayaknya", "lumayan"],
        description="Colloquial hedging 'kayaknya lumayan bagus' = seems pretty good.",
        minimal_pair="Lumayan bagus.",
    ),
]

def get_probes_by_phenomenon(phenomenon: str) -> List[LinguisticProbe]:
    return [p for p in PROBE_SET if p.phenomenon == phenomenon]


def get_all_probes() -> List[LinguisticProbe]:
    return list(PROBE_SET)


def get_probe_samples(include_minimal_pairs: bool = False) -> List[dict]:
    samples = []
    for probe in PROBE_SET:
        samples.append({
            "text": probe.text,
            "expected": probe.expected_label,
            "meta": {
                "phenomenon": probe.phenomenon,
                "phenomenon_tokens": probe.phenomenon_tokens,
                "description": probe.description,
                "is_minimal_pair": False,
            },
        })
        if include_minimal_pairs and probe.minimal_pair:
            samples.append({
                "text": probe.minimal_pair,
                "expected": probe.expected_label,
                "meta": {
                    "phenomenon": probe.phenomenon,
                    "phenomenon_tokens": probe.phenomenon_tokens,
                    "description": f"[MINIMAL PAIR] {probe.description}",
                    "is_minimal_pair": True,
                    "original_text": probe.text,
                },
            })
    return samples


def probe_accuracy_by_phenomenon(
    predictions: List[dict],
    probes: Optional[List[LinguisticProbe]] = None,
) -> dict:
    probes = probes or PROBE_SET
    assert len(predictions) == len(probes), (
        f"Got {len(predictions)} predictions but {len(probes)} probes"
    )
    counts: Dict[str, dict] = {phenom: {"correct": 0, "total": 0} for phenom in PHENOMENON_CATEGORIES}

    for pred, probe in zip(predictions, probes):
        phenom = probe.phenomenon
        counts[phenom]["total"] += 1
        if pred["predicted"] == probe.expected_label:
            counts[phenom]["correct"] += 1

    result = {}
    for phenom, c in counts.items():
        if c["total"] > 0:
            result[phenom] = {**c, "accuracy": c["correct"] / c["total"]}
    return result


def save_probes_as_tsv(output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("text\texpected_label\tphenomenon\tphenomenon_tokens\tdescription\tminimal_pair\n")
        for probe in PROBE_SET:
            tokens_str = "|".join(probe.phenomenon_tokens)
            minimal = probe.minimal_pair or ""
            f.write(f"{probe.text}\t{probe.expected_label}\t{probe.phenomenon}\t"
                    f"{tokens_str}\t{probe.description}\t{minimal}\n")


def save_probes_as_json(output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(p) for p in PROBE_SET]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def probe_set_summary() -> dict:
    from collections import Counter
    by_phenom = Counter(p.phenomenon for p in PROBE_SET)
    by_label  = Counter(p.expected_label for p in PROBE_SET)
    has_pair  = sum(1 for p in PROBE_SET if p.minimal_pair)
    return {
        "total": len(PROBE_SET),
        "by_phenomenon": dict(by_phenom),
        "by_label": dict(by_label),
        "with_minimal_pair": has_pair,
    }


if __name__ == "__main__":
    import pprint
    pprint.pprint(probe_set_summary())
    print(f"\nTotal probes: {len(PROBE_SET)}")
    print(f"Phenomena covered: {list(PHENOMENON_CATEGORIES.keys())}")
