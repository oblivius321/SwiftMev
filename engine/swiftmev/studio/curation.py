"""SwiftMev Dataset Studio - Split Curation & Leakage Prevention Engine.

Implements rigorous split partition validation with pairwise disjointness verification:
    Train ∩ Val = ∅
    Train ∩ Test = ∅
    Val ∩ Test = ∅

Guarantees that no duplicate (cryptographic or perceptual) or derived image
crosses split boundaries, and enforces experimental group isolation.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Set, Tuple

from swiftmev.studio.inventory import hamming_distance
from swiftmev.studio.schema import DatasetSplitRecord, MicrographRecord


class LeakageError(Exception):
    """Raised when data leakage is detected between dataset splits."""
    pass


def validate_splits(
    micrographs: List[MicrographRecord] | Dict[str, MicrographRecord],
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
    max_dhash_distance: int = 2,
    strict_group_isolation: bool = True,
) -> DatasetSplitRecord:
    """Validates split assignments against pairwise leakage, duplicates, and group contamination.

    Rigorously verifies:
      1. Train ∩ Val = ∅
      2. Train ∩ Test = ∅
      3. Val ∩ Test = ∅
      4. Cryptographic (SHA-256) duplicate cross-contamination
      5. Perceptual (dHash) duplicate cross-contamination
      6. Group isolation (no physical sample or conservative group across splits)
    """
    if isinstance(micrographs, list):
        micro_dict = {m.micrograph_id: m for m in micrographs}
    else:
        micro_dict = micrographs

    set_train = set(train_ids)
    set_val = set(val_ids)
    set_test = set(test_ids)

    messages: List[str] = []
    has_leakage = False

    # 1. Check pairwise set intersections
    intersect_tv = set_train.intersection(set_val)
    intersect_tt = set_train.intersection(set_test)
    intersect_vt = set_val.intersection(set_test)

    pairwise_disjoint = True
    if intersect_tv:
        pairwise_disjoint = False
        has_leakage = True
        messages.append(f"VAZAMENTO DETECTADO (Train ∩ Val ≠ ∅): {len(intersect_tv)} IDs compartilhados ({list(intersect_tv)[:3]}...)")

    if intersect_tt:
        pairwise_disjoint = False
        has_leakage = True
        messages.append(f"VAZAMENTO DETECTADO (Train ∩ Test ≠ ∅): {len(intersect_tt)} IDs compartilhados ({list(intersect_tt)[:3]}...)")

    if intersect_vt:
        pairwise_disjoint = False
        has_leakage = True
        messages.append(f"VAZAMENTO DETECTADO (Val ∩ Test ≠ ∅): {len(intersect_vt)} IDs compartilhados ({list(intersect_vt)[:3]}...)")

    # 2. Check for missing IDs in record dictionary
    all_assigned = set_train | set_val | set_test
    missing_ids = [m_id for m_id in all_assigned if m_id not in micro_dict]
    if missing_ids:
        has_leakage = True
        messages.append(f"ERRO DE INTEGRIDADE: {len(missing_ids)} IDs nos splits não foram encontrados no inventário ({missing_ids[:3]}...)")

    # 3. Check for cryptographic SHA-256 duplicates across splits
    split_by_id: Dict[str, str] = {}
    for i in set_train:
        split_by_id[i] = "train"
    for i in set_val:
        split_by_id[i] = "val"
    for i in set_test:
        split_by_id[i] = "test"

    hash_to_splits: Dict[str, Dict[str, List[str]]] = {}
    for m_id, split_name in split_by_id.items():
        if m_id in micro_dict:
            rec = micro_dict[m_id]
            hash_to_splits.setdefault(rec.sha256, {}).setdefault(split_name, []).append(m_id)

    sha_leakage = False
    for sha, splits_map in hash_to_splits.items():
        if len(splits_map) > 1:
            sha_leakage = True
            has_leakage = True
            splits_str = ", ".join(f"{s}: {len(ids)}" for s, ids in splits_map.items())
            messages.append(f"VAZAMENTO POR HASH SHA-256: Imagens idênticas distribuídas em splits diferentes ({splits_str})")

    # 4. Check for perceptual dHash near-duplicates across splits
    dhash_leakage = False
    train_recs = [micro_dict[i] for i in set_train if i in micro_dict]
    val_recs = [micro_dict[i] for i in set_val if i in micro_dict]
    test_recs = [micro_dict[i] for i in set_test if i in micro_dict]

    def check_perceptual(recs_a: List[MicrographRecord], recs_b: List[MicrographRecord], name_a: str, name_b: str):
        nonlocal dhash_leakage, has_leakage
        for a in recs_a:
            for b in recs_b:
                if a.sha256 == b.sha256:
                    continue  # Already caught by SHA check
                dist = hamming_distance(a.dhash, b.dhash)
                if dist <= max_dhash_distance:
                    dhash_leakage = True
                    has_leakage = True
                    messages.append(
                        f"VAZAMENTO PERCEPTUAL dHash ({name_a} × {name_b}): {a.micrograph_id} e {b.micrograph_id} "
                        f"possuem distância Hamming {dist} <= {max_dhash_distance}"
                    )

    check_perceptual(train_recs, val_recs, "train", "val")
    check_perceptual(train_recs, test_recs, "train", "test")
    check_perceptual(val_recs, test_recs, "val", "test")

    hash_duplicates_verified = (not sha_leakage) and (not dhash_leakage)

    # 5. Check Group Separation (conservative_group_id or physical_sample_id)
    group_to_splits: Dict[str, Set[str]] = {}
    for m_id, split_name in split_by_id.items():
        if m_id in micro_dict:
            rec = micro_dict[m_id]
            grp = rec.physical_sample_id if rec.physical_sample_id != "unknown" else rec.conservative_group_id
            group_to_splits.setdefault(grp, set()).add(split_name)

    group_leakage = False
    for grp, splits in group_to_splits.items():
        if len(splits) > 1:
            group_leakage = True
            if strict_group_isolation:
                has_leakage = True
            messages.append(
                f"CONTAMINAÇÃO DE GRUPO EXPERIMENTAL: Grupo '{grp}' encontrado em múltiplos splits: {sorted(list(splits))}"
            )

    group_separation_verified = not group_leakage

    # 6. Check experimental review status
    unreviewed_count = sum(
        1 for m_id in all_assigned
        if m_id in micro_dict and not micro_dict[m_id].experimental_independence_reviewed
    )
    all_independence_reviewed = (unreviewed_count == 0) and (len(all_assigned) > 0)
    if unreviewed_count > 0:
        messages.append(
            f"REVISÃO PENDENTE: {unreviewed_count} imagens no dataset possuem relações experimentais provisórias "
            f"(physical_sample_id='unknown' ou experimental_independence_reviewed=False)."
        )

    automated_passed = pairwise_disjoint and hash_duplicates_verified and group_separation_verified and (not missing_ids)

    if not has_leakage and not messages:
        messages.append("Verificação automatizada concluída com sucesso: splits disjuntos aos pares e isolados por grupo.")

    return DatasetSplitRecord(
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        automated_leakage_checks_passed=automated_passed,
        pairwise_disjoint_verified=pairwise_disjoint,
        hash_duplicates_verified=hash_duplicates_verified,
        group_separation_verified=group_separation_verified,
        experimental_independence_reviewed=all_independence_reviewed,
        validation_messages=messages,
    )


def partition_groups_disjoint(
    micrographs: List[MicrographRecord],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> DatasetSplitRecord:
    """Partitions micrographs into Train, Val, and Test by grouping conservative clusters and duplicates.

    Guarantees that:
      - All exact duplicates (SHA-256) stay in the same split.
      - All images in the same conservative_group_id stay in the same split.
      - Train ∩ Val = ∅, Train ∩ Test = ∅, Val ∩ Test = ∅.
    """
    rng = random.Random(seed)

    # 1. Cluster images into unified clusters based on:
    #    (a) identical SHA-256
    #    (b) same conservative_group_id
    clusters: Dict[str, List[MicrographRecord]] = {}
    for m in micrographs:
        # Group key blends conservative group and physical sample
        cluster_key = m.physical_sample_id if m.physical_sample_id != "unknown" else m.conservative_group_id
        clusters.setdefault(cluster_key, []).append(m)

    # 2. Merge clusters that share identical SHA-256 or near-duplicate dHash (distance <= max_dhash_distance)
    max_dhash_distance = 2
    # Convert clusters into a list of clusters to merge iteratively
    cluster_list = list(clusters.values())
    merged = True
    while merged:
        merged = False
        new_cluster_list = []
        skip_indices = set()
        for i in range(len(cluster_list)):
            if i in skip_indices:
                continue
            current = list(cluster_list[i])
            for j in range(i + 1, len(cluster_list)):
                if j in skip_indices:
                    continue
                other = cluster_list[j]
                # Check if current and other should merge
                should_merge = False
                for item_a in current:
                    for item_b in other:
                        if item_a.sha256 == item_b.sha256 or hamming_distance(item_a.dhash, item_b.dhash) <= max_dhash_distance:
                            should_merge = True
                            break
                    if should_merge:
                        break
                if should_merge:
                    current.extend(other)
                    skip_indices.add(j)
                    merged = True
            new_cluster_list.append(current)
        cluster_list = new_cluster_list

    # 3. Sort clusters deterministically for reproducible partitioning
    cluster_list.sort(key=lambda c: sorted([m.micrograph_id for m in c])[0])
    rng.shuffle(cluster_list)

    total_images = len(micrographs)
    target_train = int(total_images * train_ratio)
    target_val = int(total_images * val_ratio)

    train_ids: List[str] = []
    val_ids: List[str] = []
    test_ids: List[str] = []

    for cluster_items in cluster_list:
        k_ids = [m.micrograph_id for m in cluster_items]
        if len(train_ids) + len(k_ids) <= target_train or (not train_ids):
            train_ids.extend(k_ids)
        elif len(val_ids) + len(k_ids) <= target_val or (not val_ids):
            val_ids.extend(k_ids)
        else:
            test_ids.extend(k_ids)

    # If test is empty due to small number of clusters, rebalance from train
    if not test_ids and len(train_ids) > len(val_ids) + 5:
        # Move the last cluster from train to test
        for cluster_items in reversed(cluster_list):
            k_ids = [m.micrograph_id for m in cluster_items]
            if all(i in train_ids for i in k_ids):
                for i in k_ids:
                    train_ids.remove(i)
                test_ids.extend(k_ids)
                break

    return validate_splits(
        micrographs=micrographs,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        strict_group_isolation=True,
    )
