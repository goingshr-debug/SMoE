#!/usr/bin/env python3
"""CPU-only equivalence checks for the fused replacement-routing sort."""

import random

from utils.expertcache import cache_router, replaceset_between_tokens


class FakeCache:
    def __init__(self, resident):
        self.resident = set(resident)
        self.ready = []

    def query_expert(self, uid):
        return uid in self.resident

    def ready_compute(self, uid):
        self.ready.append(uid)


def reference_replaceset(scores, ratio, topk):
    replaceset = set()
    allset = set()
    n_tokens = len(scores)
    n_experts = len(scores[0])
    sort_index = [
        sorted(range(len(row)), key=lambda i: row[i], reverse=True)
        for row in scores
    ]
    sort_scores = [
        [scores[token][i] for i in sort_index[token]]
        for token in range(n_tokens)
    ]
    for token in range(n_tokens):
        midscore = sort_scores[token][topk]
        lscore = midscore + ratio * midscore
        for expert_i in range(n_experts):
            if sort_scores[token][expert_i] >= lscore and expert_i < topk:
                replaceset.add(sort_index[token][expert_i])
            if expert_i < topk:
                allset.add(sort_index[token][expert_i])
    return list(replaceset), list(allset)


def reference_cache_router(scores, cache, ratio, topk, replaceset, layer_id):
    n_tokens = len(scores)
    n_experts = len(scores[0])
    routed = [[None for _ in range(topk)] for _ in range(n_tokens)]
    top_uid = [[] for _ in range(n_tokens)]
    sort_index = [
        sorted(range(len(row)), key=lambda i: row[i], reverse=True)
        for row in scores
    ]
    sort_scores = [
        [scores[token][i] for i in sort_index[token]]
        for token in range(n_tokens)
    ]

    for token in range(n_tokens):
        midscore = sort_scores[token][topk]
        lscore = midscore + ratio * midscore
        rscore = midscore - ratio * midscore
        high_num = 0
        canreplaceset = set()
        for expert_i in range(n_experts):
            expertid = sort_index[token][expert_i]
            if sort_scores[token][expert_i] >= lscore and expert_i < topk:
                cache.ready_compute((layer_id, expertid))
                routed[token][expert_i] = expertid
                top_uid[token].append((layer_id, expertid))
                high_num += 1
            elif rscore < sort_scores[token][expert_i] < midscore:
                canreplaceset.add(expertid)

        for expert_i in range(high_num, topk):
            expertid = sort_index[token][expert_i]
            uid = (layer_id, expertid)
            if cache.query_expert(uid) or expertid in replaceset:
                routed[token][expert_i] = expertid
                cache.ready_compute(uid)
                continue
            replaced = False
            for replaceexpertid in canreplaceset:
                replaceuid = (layer_id, replaceexpertid)
                if ((cache.query_expert(replaceuid)
                     or replaceexpertid in replaceset)
                        and replaceexpertid not in routed[token]):
                    routed[token][expert_i] = replaceexpertid
                    canreplaceset.remove(replaceexpertid)
                    cache.ready_compute(replaceuid)
                    replaced = True
                    break
            if not replaced:
                routed[token][expert_i] = expertid
                cache.ready_compute(uid)
    return routed, top_uid


def normalized_scores(seed, tokens, experts=64):
    rng = random.Random(seed)
    rows = []
    for _ in range(tokens):
        raw = [rng.random() for _ in range(experts)]
        total = sum(raw)
        rows.append([value / total for value in raw])
    return rows


def run_case(scores, resident, ratio=0.25, topk=4, layer_id=7):
    ref_replace, ref_all = reference_replaceset(scores, ratio, topk)
    default_replace, default_all = replaceset_between_tokens(
        scores, ratio, topk)
    assert set(default_replace) == set(ref_replace)
    assert set(default_all) == set(ref_all)

    got_replace, got_all, indices, sorted_scores = replaceset_between_tokens(
        scores, ratio, topk, return_sorted=True)
    assert set(got_replace) == set(ref_replace)
    assert set(got_all) == set(ref_all)

    ref_cache = FakeCache(resident)
    expected = reference_cache_router(
        scores, ref_cache, ratio, topk, ref_replace, layer_id)

    reused_cache = FakeCache(resident)
    reused = cache_router(
        scores, reused_cache, ratio, topk, got_replace, layer_id,
        sorted_indices=indices, sorted_scores=sorted_scores)
    assert reused == expected
    assert reused_cache.ready == ref_cache.ready
    assert all(None not in row for row in reused[0])

    fallback_cache = FakeCache(resident)
    fallback = cache_router(
        scores, fallback_cache, ratio, topk, got_replace, layer_id)
    assert fallback == expected
    assert fallback_cache.ready == ref_cache.ready


def main():
    for seed in range(20):
        for tokens in (1, 2, 17):
            scores = normalized_scores(seed, tokens)
            resident = {(7, eid) for eid in range(64) if (eid + seed) % 3 == 0}
            run_case(scores, resident)

    tied = [[0.04] * 5 + [0.02] * 10 + [0.01] * 49]
    run_case(tied, {(7, 1), (7, 8), (7, 31)})
    print("PASS: fused routing sort matches the pre-change reference")


if __name__ == "__main__":
    main()
