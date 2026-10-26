#!/usr/bin/env python3
# patch-lazykv.py [ENGINE_DIR] - larry, STRATA_DENSE_LAZY_KV=1 (default 0): card 0's KV state (the dense half's
# compressed KV and indexer keys of every KV-source layer) in Level Zero virtual memory, mapped as the context grows -
# the way the TP cards' replicas already are.  At --ctx 1048576 those tensors are ~2.3 GiB more than at 262K, all of it
# allocated at load out of card 0's expert cache (1,850 -> 1,669 experts: the 1M config's ordinary-length agent replay
# ran ~10 % slower).  Now the whole context's rows are reserved and STRATA_DENSE_KV_INIT positions (262144) mapped at
# load (the 262K config's VRAM); a prompt or pass that reaches further first maps 32K positions more (card 0's expert
# cache gives the VRAM back: shrink_cache_for), the new pages zeroed.  vmm_peer_read (bench/engine1): the other
# cards' copy engines read card 0's virtual memory exactly, pages mapped later too (the replicas copy their rows from
# card 0).  The serve loop only (one-shot runs map the whole context at load).  Idempotent per file.
import os
import sys

D = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine"


def patch(path, edits, marker):
    s = open(path).read()
    if marker in s:
        print("already patched", path)
        return
    for old, new in edits:
        if s.count(old) != 1:
            sys.exit("%s: anchor x%d: %s" % (path, s.count(old), old[:100]))
        s = s.replace(old, new)
    open(path, "w").write(s)
    print("patched", path)


# ---------------------------------------------------------------- tp_pager.hpp / .cpp: the map
patch(os.path.join(D, "tp/tp_pager.hpp"), [("""    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace strata::v41
""", """    struct Impl;
    std::unique_ptr<Impl> p_;
};

/// larry (STRATA_DENSE_LAZY_KV): card 0's KV state in Level Zero virtual memory, mapped as the context grows.  alloc()
/// reserves the units (one a tensor, each page-aligned) and maps the page holding each unit's spare byte (its last
/// row) up front; grow() maps each unit's pages up to bytes[i] from its start (never past that page), zeroing them.
class Ds41KvMap {
public:
    Ds41KvMap();
    ~Ds41KvMap();
    Ds41KvMap(const Ds41KvMap &) = delete;
    Ds41KvMap & operator=(const Ds41KvMap &) = delete;

    /// queue: the dense backend's sycl::queue * (card 0)
    bool init(void * queue, std::string & err);
    void * alloc(const std::vector<size_t> & unit_bytes, const std::vector<size_t> & spare_off, std::vector<size_t> & unit_off,
                 size_t & total);
    /// the VRAM a grow to bytes[] would map
    size_t grow_bytes(const std::vector<size_t> & bytes) const;
    bool grow(const std::vector<size_t> & bytes, std::string & err);
    size_t mapped() const;         ///< bytes mapped (all units)
    size_t reserved() const;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace strata::v41
""")], "class Ds41KvMap")

patch(os.path.join(D, "tp/tp_pager.cpp"), [("""std::string Ds41Pager::report() const {""", """// ---------------------------------------------------------------- Ds41KvMap
struct Ds41KvMap::Impl {
    sycl::queue * q0 = nullptr;
    ze_context_handle_t zc = nullptr;
    ze_device_handle_t zd = nullptr;
    size_t page = 0;
    uint8_t * va = nullptr;
    size_t va_bytes = 0;
    struct Unit { size_t off = 0, bytes = 0, tail = 0, done = 0; };   // tail: the spare row's page (mapped to the end)
    std::vector<Unit> units;
    std::vector<std::pair<void *, size_t>> maps;
    std::vector<ze_physical_mem_handle_t> phys;
    bool map(size_t off, size_t bytes, std::string & err) {
        ze_physical_mem_desc_t d = { ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC, nullptr, 0, bytes };
        ze_physical_mem_handle_t h = nullptr;
        if (zePhysicalMemCreate(zc, zd, &d, &h) != ZE_RESULT_SUCCESS || !h) { err = "kv map: zePhysicalMemCreate failed"; return false; }
        if (zeVirtualMemMap(zc, va + off, bytes, h, 0, ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE) != ZE_RESULT_SUCCESS) {
            zePhysicalMemDestroy(zc, h);
            err = "kv map: zeVirtualMemMap failed";
            return false;
        }
        phys.push_back(h);
        maps.emplace_back(va + off, bytes);
        q0->memset(va + off, 0, bytes).wait();   // fresh pages: the indexer scores every row up to its cap before masking
        return true;
    }
    ~Impl() {
        for (auto & m : maps) zeVirtualMemUnmap(zc, m.first, m.second);
        for (auto h : phys) zePhysicalMemDestroy(zc, h);
        if (va) zeVirtualMemFree(zc, va, va_bytes);
    }
};

Ds41KvMap::Ds41KvMap() : p_(std::make_unique<Impl>()) {}
Ds41KvMap::~Ds41KvMap() = default;

bool Ds41KvMap::init(void * queue, std::string & err) {
    Impl & im = *p_;
    im.q0 = static_cast<sycl::queue *>(queue);
    if (!im.q0) { err = "kv map: no queue"; return false; }
    im.zc = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(im.q0->get_context());
    im.zd = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(im.q0->get_device());
    size_t pg = 0;
    if (zeVirtualMemQueryPageSize(im.zc, im.zd, (size_t) 1 << 30, &pg) != ZE_RESULT_SUCCESS || !pg) { err = "kv map: page size query failed"; return false; }
    im.page = std::max<size_t>(pg, (size_t) 2 << 20);
    return true;
}

void * Ds41KvMap::alloc(const std::vector<size_t> & unit_bytes, const std::vector<size_t> & spare_off, std::vector<size_t> & unit_off,
                        size_t & total) {
    Impl & im = *p_;
    total = 0;
    im.units.assign(unit_bytes.size(), {});
    unit_off.assign(unit_bytes.size(), 0);
    size_t off = 0;
    for (size_t i = 0; i < unit_bytes.size(); ++i) {
        Impl::Unit & U = im.units[i];
        U.off = off;
        U.bytes = up_to(unit_bytes[i], im.page);
        U.tail = std::min(U.bytes - im.page, (i < spare_off.size() ? spare_off[i] : unit_bytes[i] - 1) / im.page * im.page);
        unit_off[i] = off;
        off += U.bytes;
    }
    if (!off) return nullptr;
    void * p = nullptr;
    if (zeVirtualMemReserve(im.zc, nullptr, off, &p) != ZE_RESULT_SUCCESS || !p) return nullptr;
    im.va = (uint8_t *) p;
    im.va_bytes = off;
    std::string err;
    for (auto & U : im.units)
        if (!im.map(U.off + U.tail, U.bytes - U.tail, err)) { std::fprintf(stderr, "%s\\n", err.c_str()); return nullptr; }
    total = off;
    return im.va;
}

size_t Ds41KvMap::grow_bytes(const std::vector<size_t> & bytes) const {
    const Impl & im = *p_;
    size_t b = 0;
    for (size_t i = 0; i < im.units.size() && i < bytes.size(); ++i) {
        const size_t w = std::min(im.units[i].tail, up_to(bytes[i], im.page));
        if (w > im.units[i].done) b += w - im.units[i].done;
    }
    return b;
}

bool Ds41KvMap::grow(const std::vector<size_t> & bytes, std::string & err) {
    Impl & im = *p_;
    for (size_t i = 0; i < im.units.size() && i < bytes.size(); ++i) {
        Impl::Unit & U = im.units[i];
        const size_t w = std::min(U.tail, up_to(bytes[i], im.page));
        if (w <= U.done) continue;
        if (!im.map(U.off + U.done, w - U.done, err)) return false;
        U.done = w;
    }
    return true;
}

size_t Ds41KvMap::mapped() const {
    size_t b = 0;
    for (const auto & U : p_->units) b += U.done + (U.bytes - U.tail);
    return b;
}

size_t Ds41KvMap::reserved() const { return p_->va_bytes; }

std::string Ds41Pager::report() const {""")], "struct Ds41KvMap::Impl")

# ---------------------------------------------------------------- ds41_dense.hpp / .cpp
patch(os.path.join(D, "ds41_dense.hpp"), [("""    std::function<ggml_backend_buffer_t(const std::vector<size_t> & unit_bytes, std::vector<size_t> & unit_off)> paged_alloc;
};
""", """    std::function<ggml_backend_buffer_t(const std::vector<size_t> & unit_bytes, std::vector<size_t> & unit_off)> paged_alloc;
    /// larry (STRATA_DENSE_LAZY_KV): the KV-source layers' compressed KV and indexer keys in memory the caller provides
    /// and maps as the context grows (kv_bytes_for).  Called once at init with each tensor's bytes and its spare row's
    /// offset; returns a backend buffer over the units (unit_off filled in) or null (the state buffer holds them).
    std::function<ggml_backend_buffer_t(const std::vector<size_t> & unit_bytes, const std::vector<size_t> & spare_off,
                                        std::vector<size_t> & unit_off)> kv_alloc;
};
"""), ("""    bool park(int64_t positions);
""", """    bool park(int64_t positions);
    /// larry (cfg.kv_alloc): per KV unit (kv_alloc's order) the bytes from its start that `positions` positions use
    void kv_bytes_for(int64_t positions, std::vector<size_t> & bytes) const;
    bool kv_lazy() const;
""")], "kv_bytes_for")

patch(os.path.join(D, "ds41_dense.cpp"), [("""    ggml_backend_buffer_t sbuf = nullptr;
""", """    ggml_backend_buffer_t sbuf = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;   ///< larry (cfg.kv_alloc): over the caller's KV units
    std::vector<std::pair<ggml_tensor *, int>> kv_units;   ///< larry: (comp / idxk tensor, its ratio) in kv_alloc's order
"""), ("""        if (sbuf) ggml_backend_buffer_free(sbuf);
""", """        if (sbuf) ggml_backend_buffer_free(sbuf);
        if (kv_buf) ggml_backend_buffer_free(kv_buf);
"""), ("""    im.sbuf = ggml_backend_alloc_ctx_tensors(im.sctx, im.backend);
""", """    // larry (cfg.kv_alloc, STRATA_DENSE_LAZY_KV): the compressed KV and indexer keys in the caller's virtual memory,
    // mapped as the context grows - a 1M context's rows stay out of card 0's VRAM until a conversation reaches them
    if (im.sycl && im.cfg.kv_alloc) {
        const ggml_backend_buffer_type_t bt = ggml_backend_get_default_buffer_type(im.backend);
        std::vector<size_t> ub, spare, uo;
        for (auto & L : im.ly)
            for (ggml_tensor * t : { L.comp, L.idxk })
                if (t) {
                    im.kv_units.push_back({ t, (int) L.ratio });
                    ub.push_back(ggml_backend_buft_get_alloc_size(bt, t));
                    spare.push_back((size_t) (t->ne[1] - 1) * t->nb[1]);   // the spare row (open compressor groups)
                }
        if (!im.kv_units.empty())
            if (ggml_backend_buffer_t kb = im.cfg.kv_alloc(ub, spare, uo)) {
                uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(kb);
                for (size_t i = 0; i < im.kv_units.size(); ++i)
                    if (ggml_backend_tensor_alloc(kb, im.kv_units[i].first, base + uo[i]) != GGML_STATUS_SUCCESS) {
                        err = "cannot bind the KV state in the lazy map";
                        ggml_backend_buffer_free(kb);
                        return false;
                    }
                im.kv_buf = kb;
            }
        if (!im.kv_buf) im.kv_units.clear();
    }
    im.sbuf = ggml_backend_alloc_ctx_tensors(im.sctx, im.backend);
"""), ("""bool Ds41Dense::park(int64_t positions) {""", """void Ds41Dense::kv_bytes_for(int64_t positions, std::vector<size_t> & bytes) const {
    const Impl & im = *p_;
    bytes.clear();
    for (const auto & u : im.kv_units) {   // row r of a ratio-k source holds a position >= r * k: rows [0, p / k] + a margin
        const int64_t k = u.second == 2 ? 2 : 1;
        const int64_t rows = std::min<int64_t>(u.first->ne[1] - 1, positions / k + 2);
        bytes.push_back((size_t) rows * u.first->nb[1]);
    }
}

bool Ds41Dense::kv_lazy() const { return p_->kv_buf != nullptr; }

bool Ds41Dense::park(int64_t positions) {""")], "kv_buf")

# ---------------------------------------------------------------- ds41_generate.cpp
patch(os.path.join(D, "ds41_generate.cpp"), [("""    std::unique_ptr<Ds41Pager> pager;
""", """    std::unique_ptr<Ds41Pager> pager;
    // larry (STRATA_DENSE_LAZY_KV=1, --serve): card 0's KV state mapped as the context grows (Ds41KvMap); declared
    // before the dense half too
    std::unique_ptr<Ds41KvMap> kvmap;
    int64_t kv_pos = 0;   // positions the map covers
"""), ("""    Ds41Dense dense;
    std::string err;
    const double t_load = now_ms();
    if (!dense.init(a.model, dc, err)) { std::fprintf(stderr, "ds41_generate: dense init: %s\\n", err.c_str()); return 1; }
""", """#if defined(DS41_TP)
    if (a.serve && !ggml_backend_is_cpu(be) && std::getenv("STRATA_DENSE_LAZY_KV") && std::getenv("STRATA_DENSE_LAZY_KV")[0] == '1') {
        kvmap = std::make_unique<Ds41KvMap>();
        std::string kerr;
        if (!kvmap->init(ggml_backend_sycl_get_queue(be), kerr)) {
            std::fprintf(stderr, "ds41_generate: STRATA_DENSE_LAZY_KV: %s - off\\n", kerr.c_str());
            kvmap.reset();
        } else {
            dc.kv_alloc = [&kvmap](const std::vector<size_t> & ub, const std::vector<size_t> & sp, std::vector<size_t> & uo) -> ggml_backend_buffer_t {
                size_t tot = 0;
                void * va = kvmap->alloc(ub, sp, uo, tot);
                return va ? ggml_backend_sycl_buffer_from_ptr(0, va, tot) : nullptr;
            };
        }
    }
#endif
    Ds41Dense dense;
    std::string err;
    const double t_load = now_ms();
    if (!dense.init(a.model, dc, err)) { std::fprintf(stderr, "ds41_generate: dense init: %s\\n", err.c_str()); return 1; }
#if defined(DS41_TP)
    if (kvmap && dense.kv_lazy()) {   // the first STRATA_DENSE_KV_INIT positions now (the VRAM the tier sees as taken)
        const int64_t init = std::min<int64_t>(ctx, std::getenv("STRATA_DENSE_KV_INIT") ? std::atoll(std::getenv("STRATA_DENSE_KV_INIT")) : 262144);
        std::vector<size_t> kb;
        dense.kv_bytes_for(init, kb);
        if (!kvmap->grow(kb, err)) { std::fprintf(stderr, "ds41_generate: KV map: %s\\n", err.c_str()); return 1; }
        kv_pos = init;
        std::fprintf(stderr, "kv map: card 0's KV state in virtual memory - %lld of %lld positions mapped (%.2f of %.2f GiB)\\n",
                     (long long) init, (long long) ctx, kvmap->mapped() / 1073741824.0, kvmap->reserved() / 1073741824.0);
    } else if (kvmap) {
        kvmap.reset();
    }
#endif
"""), ("""            bool ok = total == reused || dh == 0 || (by_step ? feed_steps() : feed(ids.data() + reused, dh, reused, pp, nullptr));
""", """            if (!ensure_kv(total + 64)) { std::printf("ERR %s\\n", err.c_str()); std::fflush(stdout); continue; }
            bool ok = total == reused || dh == 0 || (by_step ? feed_steps() : feed(ids.data() + reused, dh, reused, pp, nullptr));
"""), ("""                    if (!pass_v(vt.data(), (int) vt.size(), p0)) { finish = "error"; break; }
""", """                    if (!ensure_kv(p0 + (int64_t) vt.size() + 64)) { ferr = err; finish = "error"; break; }
                    if (!pass_v(vt.data(), (int) vt.size(), p0)) { finish = "error"; break; }
"""), ("""                if (!step(tok, pos) || !logits_rows(0, 1, lg.data())) { finish = "error"; break; }
""", """                if (!ensure_kv(pos + 64)) { ferr = err; finish = "error"; break; }
                if (!step(tok, pos) || !logits_rows(0, 1, lg.data())) { finish = "error"; break; }
"""), ("""        const bool park_on = !(std::getenv("STRATA_SERVE_PARK") && std::getenv("STRATA_SERVE_PARK")[0] == '0');
""", """        const bool park_on = !(std::getenv("STRATA_SERVE_PARK") && std::getenv("STRATA_SERVE_PARK")[0] == '0');
        // larry (STRATA_DENSE_LAZY_KV): before a prompt or a pass reaches past the mapped positions, 32K more (at least
        // what it needs) - card 0's expert cache gives the VRAM back first (between passes: nothing reads its slabs)
        auto ensure_kv = [&](int64_t p) -> bool {
#if defined(DS41_TP)
            if (!kvmap || p <= kv_pos) return true;
            const int64_t want = std::min<int64_t>(ctx, std::max<int64_t>(p, kv_pos + 32768));
            std::vector<size_t> kb;
            dense.kv_bytes_for(want, kb);
            const size_t need = kvmap->grow_bytes(kb);
            const double t0 = now_ms();
            size_t fr = 0, tot = 0;
            ggml_backend_dev_memory(ggml_backend_get_device(be), &fr, &tot);
            const size_t margin = (size_t) 512 << 20;
            if (need + margin > fr) {
                std::string kerr;
                if (tier.shrink_cache_for((int64_t) (need + margin), kerr) < 0 && !kerr.empty())
                    std::fprintf(stderr, "kv map: card 0's cache: %s\\n", kerr.c_str());
            }
            if (!kvmap->grow(kb, err)) return false;
            std::fprintf(stderr, "kv map: %lld -> %lld positions (+%.2f GiB, %.2f GiB mapped; %.0f ms)\\n", (long long) kv_pos,
                         (long long) want, need / 1073741824.0, kvmap->mapped() / 1073741824.0, now_ms() - t0);
            kv_pos = want;
#else
            (void) p;
#endif
            return true;
        };
""")], "ensure_kv")
