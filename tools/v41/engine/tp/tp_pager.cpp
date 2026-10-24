// tools/v41/engine/tp/tp_pager.cpp - larry, engine1: card 0's copy of the dense weights the TP cards also hold,
// paged (tp_pager.hpp)
#include "tp_pager.hpp"
#include "tp_core.hpp"

#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace strata::v41 {

namespace {

constexpr int64_t QK = 256;
constexpr size_t Q6B = 210;
// Q6_K planar (ggml-sycl's reorder, the TP core's shards): ql | qh | scales | d planes over the blocks, row-major
constexpr size_t PO[4] = { 0, 128, 192, 208 }, PB[4] = { 128, 64, 16, 2 };

double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
bool env_on(const char * k, bool d) { const char * v = std::getenv(k); return v && *v ? v[0] != '0' : d; }
int env_i(const char * k, int d) { const char * v = std::getenv(k); return v && *v ? std::atoi(v) : d; }
size_t up_to(size_t v, size_t a) { return (v + a - 1) / a * a; }

}  // namespace

struct Ds41Pager::Impl {
    std::string err;
    sycl::queue * q0 = nullptr;      // the dense backend's queue (card 0): the fills' local part, the scatters
    sycl::context ctx;               // its context (a copy: the queue may be gone when this is destroyed)
    ze_context_handle_t zc = nullptr;
    ze_device_handle_t zd = nullptr;
    size_t page = (size_t) 2 << 20;
    uint8_t * va = nullptr;
    size_t va_bytes = 0;
    bool on = false;
    size_t freed_bytes = 0, slot_bytes = 0;
    // a fill copy: rank `rank`'s shard bytes -> the unit (dst) or the slot's staging (stg_off)
    struct Op { int rank; const uint8_t * src; uint8_t * dst; size_t bytes; bool stg; size_t stg_off; };
    // a K-split rank's blocks into every row of the tensor (from its shard on card 0, or from the staging)
    struct Sc { const uint8_t * src; bool stg; size_t stg_off; uint8_t * dst; int64_t N, NBR, a0, nbk; };
    struct Unit {
        size_t off = 0, bytes = 0, stg = 0;                     // in va (page-rounded), staging bytes its fill needs
        ze_physical_mem_handle_t load = nullptr;                // the load-time pages (until enable)
        int slot = -1;                                          // the slot whose pages it has (-1: unmapped)
        bool filled = false;                                    // its fill's local part queued
        std::vector<Op> ops;
        std::vector<Sc> sc;
        std::vector<std::pair<uint8_t *, size_t>> tens;         // its tensors (check)
    };
    std::vector<Unit> units;
    struct Slot {
        ze_physical_mem_handle_t ph = nullptr;
        uint8_t * stg = nullptr;
        int unit = -1;
        sycl::event used;                  // after the last work queued on its unit (fill included)
        std::vector<sycl::event> p2p;      // its fill's card-to-card copies
        uint64_t tick = 0;
    };
    std::vector<Slot> slots;
    std::vector<sycl::queue *> tq;        // the TP core's queues, by rank (rank 0's must be on the dense card)
    std::unique_ptr<sycl::queue> qc;      // card 0's copy queue: the fills' card-to-card copies (copies_far)
    uint64_t clock = 0;
    // debug: STRATA_TP_PAGE_KEEP=1 keeps every unit's load-time pages (nothing given back) and refills a unit in place
    // at each pass's first need (the fills' traffic and timing without the remapping); STRATA_TP_PAGE_PREFETCH=0: no
    // next-layer copies ahead
    bool keep = false, prefetch = true;
    int * dummy = nullptr;
    int last_need = -1;
    uint8_t * keep_stg = nullptr;
    int64_t n_fill = 0, n_pref = 0, n_wait = 0, n_miss = 0;
    double wait_ms = 0;
    size_t fill_bytes = 0;

    // (at exit: the dense backend's and the TP core's queues are gone by now - their work is long done)
    ~Impl() {
        for (auto & S : slots) for (auto & e : S.p2p) { try { e.wait(); } catch (...) {} }
        for (auto & U : units) {
            if (U.slot >= 0 && va) zeVirtualMemUnmap(zc, va + U.off, U.bytes);
            if (U.load) { zeVirtualMemUnmap(zc, va + U.off, U.bytes); zePhysicalMemDestroy(zc, U.load); }
        }
        for (auto & S : slots) {
            if (S.ph) zePhysicalMemDestroy(zc, S.ph);
            if (S.stg) { try { sycl::free(S.stg, ctx); } catch (...) {} }
        }
        if (keep_stg) { try { sycl::free(keep_stg, ctx); } catch (...) {} }
        if (va) zeVirtualMemFree(zc, va, va_bytes);
    }

    bool phys(size_t bytes, ze_physical_mem_handle_t & h) {
        ze_physical_mem_desc_t d = { ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC, nullptr, 0, bytes };
        h = nullptr;
        return zePhysicalMemCreate(zc, zd, &d, &h) == ZE_RESULT_SUCCESS && h;
    }
    bool map(void * p, size_t bytes, ze_physical_mem_handle_t h) {
        return zeVirtualMemMap(zc, p, bytes, h, 0, ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE) == ZE_RESULT_SUCCESS;
    }

    void scatter(sycl::queue & q, const Sc & s, const uint8_t * stg) {
        const uint8_t * src = s.stg ? stg + s.stg_off : s.src;
        const int64_t NBR = s.NBR, a0 = s.a0, nbk = s.nbk;
        const size_t NB = (size_t) s.N * (size_t) NBR, NBk = (size_t) s.N * (size_t) nbk;
        for (int p = 0; p < 4; ++p) {
            const uint8_t * sp = src + PO[p] * NBk;
            uint8_t * dp = s.dst + PO[p] * NB;
            const size_t pb = PB[p], seg = (size_t) nbk * pb;   // a row's bytes of this rank in this plane
            if (seg % 16 == 0) {
                using u4 = sycl::vec<uint32_t, 4>;
                const size_t nu = seg / 16;
                q.parallel_for(sycl::range<1>((size_t) s.N * nu), [=](sycl::id<1> id) {
                    const size_t i = id[0], r = i / nu, j = i % nu;
                    reinterpret_cast<u4 *>(dp + ((size_t) r * NBR + a0) * pb)[j] = reinterpret_cast<const u4 *>(sp + r * seg)[j];
                });
            } else {   // the d plane (2 bytes a block)
                const size_t nu = seg / 2;
                q.parallel_for(sycl::range<1>((size_t) s.N * nu), [=](sycl::id<1> id) {
                    const size_t i = id[0], r = i / nu, j = i % nu;
                    reinterpret_cast<uint16_t *>(dp + ((size_t) r * NBR + a0) * pb)[j] = reinterpret_cast<const uint16_t *>(sp + r * seg)[j];
                });
            }
        }
    }

    // the unit's card-to-card copies (ranks >= 1) on card 0's copy queue - card 0 pulls the shards into its own
    // mapping (peer reads); returns their events.  Never on the other cards' queues: once card 0 has remapped an
    // address (a slot's pages to another unit), their copies into it still land in the pages it had before
    // (bench/engine1/vmm_remap.cpp, 2026-10-23) - in the service that garbled other layers' slots and the freed
    // load-time pages the expert cache had taken by then
    std::vector<sycl::event> copies_far(const Unit & U, uint8_t * stg) {
        std::vector<sycl::event> ev;
        for (const Op & o : U.ops) {
            if (o.rank == 0) continue;
            ev.push_back(qc->memcpy(o.stg ? stg + o.stg_off : o.dst, o.src, o.bytes));
            fill_bytes += o.bytes;
        }
        return ev;
    }
    // the unit's local part on the dense queue, after its far copies: rank 0's copies, the scatters; the last event
    sycl::event local(const Unit & U, uint8_t * stg, const std::vector<sycl::event> & far) {
        sycl::event e = q0->ext_oneapi_submit_barrier(far);
        for (const Op & o : U.ops) {
            if (o.rank != 0) continue;
            e = q0->memcpy(o.stg ? stg + o.stg_off : o.dst, o.src, o.bytes);
            fill_bytes += o.bytes;
        }
        for (const Sc & s : U.sc) scatter(*q0, s, stg);
        return q0->ext_oneapi_submit_barrier();
    }

    void evict(Slot & S) {
        if (S.unit < 0) return;
        Unit & V = units[(size_t) S.unit];
        for (auto & e : S.p2p) e.wait();   // a prefetch's copies still writing into it
        S.p2p.clear();
        S.used.wait();
        zeVirtualMemUnmap(zc, va + V.off, V.bytes);
        V.slot = -1;
        V.filled = false;
        S.unit = -1;
    }
    // map unit u into a slot and start its card-to-card copies; block = wait for a busy slot (else give up).  Never
    // the slot `keep` (the unit the dense half is about to queue work on: its last use is not recorded yet).
    bool start(int u, bool block, int keep = -1) {
        Unit & U = units[(size_t) u];
        if (U.slot >= 0) return true;
        int best = -1;
        for (size_t s = 0; s < slots.size(); ++s) {
            Slot & S = slots[s];
            if ((int) s == keep) continue;
            if (S.unit >= 0 && !block) {   // a prefetch only takes a slot whose work is done
                bool done = S.used.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::complete;
                for (auto & e : S.p2p) done = done && e.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::complete;
                if (!done) continue;
            }
            if (best < 0 || (S.unit < 0 && slots[(size_t) best].unit >= 0) ||
                ((S.unit < 0) == (slots[(size_t) best].unit < 0) && S.tick < slots[(size_t) best].tick)) best = (int) s;
        }
        if (best < 0) return false;
        Slot & S = slots[(size_t) best];
        if (S.unit >= 0) {
            const double t = now_ms();
            const bool busy = S.used.get_info<sycl::info::event::command_execution_status>() != sycl::info::event_command_status::complete;
            evict(S);
            if (busy) { ++n_wait; wait_ms += now_ms() - t; }
        }
        if (!map(va + U.off, U.bytes, S.ph)) { err = "tp pager: map failed"; return false; }
        U.slot = best;
        U.filled = false;
        S.unit = u;
        S.tick = ++clock;
        S.p2p = copies_far(U, S.stg);
        return true;
    }
    void finish(int u) {
        Unit & U = units[(size_t) u];
        Slot & S = slots[(size_t) U.slot];
        S.used = local(U, S.stg, S.p2p);
        S.p2p.clear();
        U.filled = true;
        ++n_fill;
    }
};

Ds41Pager::Ds41Pager() : p_(std::make_unique<Impl>()) {}
Ds41Pager::~Ds41Pager() {
    if (p_ && p_->on && p_->n_fill) std::fprintf(stderr, "%s\n", report().c_str());
}
bool Ds41Pager::enabled() const { return p_->on; }
size_t Ds41Pager::freed() const { return p_->freed_bytes > p_->slot_bytes ? p_->freed_bytes - p_->slot_bytes : 0; }
const std::string & Ds41Pager::last_error() const { return p_->err; }

bool Ds41Pager::init(void * queue, std::string & err) {
    Impl & im = *p_;
    im.q0 = static_cast<sycl::queue *>(queue);
    if (!im.q0) { err = "tp pager: no queue"; return false; }
    im.ctx = im.q0->get_context();
    im.zc = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(im.ctx);
    im.zd = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(im.q0->get_device());
    im.qc = std::make_unique<sycl::queue>(im.ctx, im.q0->get_device(), sycl::property::queue::in_order());
    size_t pg = 0;
    if (zeVirtualMemQueryPageSize(im.zc, im.zd, (size_t) 1 << 30, &pg) != ZE_RESULT_SUCCESS || !pg) { err = "tp pager: page size query failed"; return false; }
    im.page = std::max<size_t>(pg, (size_t) 2 << 20);
    return true;
}

void * Ds41Pager::alloc(const std::vector<size_t> & unit_bytes, std::vector<size_t> & unit_off, size_t & total) {
    Impl & im = *p_;
    total = 0;
    im.units.assign(unit_bytes.size(), {});
    unit_off.assign(unit_bytes.size(), 0);
    size_t off = 0;
    for (size_t i = 0; i < unit_bytes.size(); ++i) {
        Impl::Unit & U = im.units[i];
        U.off = off;
        U.bytes = up_to(unit_bytes[i], im.page);
        unit_off[i] = off;
        off += U.bytes;
    }
    if (!off) return nullptr;
    void * p = nullptr;
    if (zeVirtualMemReserve(im.zc, nullptr, off, &p) != ZE_RESULT_SUCCESS || !p) { im.err = "tp pager: reserve failed"; return nullptr; }
    im.va = (uint8_t *) p;
    im.va_bytes = off;
    for (auto & U : im.units) {
        if (!U.bytes) continue;
        if (!im.phys(U.bytes, U.load) || !im.map(im.va + U.off, U.bytes, U.load)) { im.err = "tp pager: load-time pages failed"; return nullptr; }
    }
    total = im.va_bytes;
    std::fprintf(stderr, "tp pager: %.2f GiB of dense weights in %zu layer units (%zu KiB pages)\n", off / 1073741824.0,
                 unit_bytes.size(), im.page >> 10);
    return im.va;
}

bool Ds41Pager::enable(const Ds41Tp & tp, const std::vector<Ds41PagedTensor> & tensors, std::string & err) {
    Impl & im = *p_;
    if (!im.va) { err = "tp pager: nothing allocated"; return false; }
    for (void * q : tp.queues()) im.tq.push_back(static_cast<sycl::queue *>(q));
    if (im.tq.empty() || im.tq[0]->get_device() != im.q0->get_device()) { err = "tp pager: the TP core's rank 0 is not the dense card"; return false; }
    const int N = tp.n_cards();
    // ---- the fill plans: every rank's shard of every tensor, covering it exactly
    for (const auto & T : tensors) {
        if (T.layer < 0 || (size_t) T.layer >= im.units.size() || !T.data || T.K % QK) { err = "tp pager: bad tensor " + T.name; return false; }
        Impl::Unit & U = im.units[(size_t) T.layer];
        uint8_t * dst = (uint8_t *) T.data;
        if (dst < im.va + U.off || dst + T.bytes > im.va + U.off + U.bytes) { err = "tp pager: " + T.name + " outside its unit"; return false; }
        const int64_t bpr = T.K / QK, NB = T.N * bpr;
        if (T.bytes != (size_t) NB * Q6B) { err = "tp pager: " + T.name + " is not plain Q6_K"; return false; }
        int64_t cover = 0;
        for (int k = 0; k < N; ++k) {
            Ds41TpShard d;
            if (!tp.shard_desc(T.name, k, d)) { err = "tp pager: rank " + std::to_string(k) + " has no own shard of " + T.name; return false; }
            const uint8_t * src = (const uint8_t *) d.p;
            switch (d.kind) {
            case 0: {
                const int64_t b0 = d.a0 * bpr, nbk = (d.a1 - d.a0) * bpr;
                for (int p = 0; p < 4; ++p)
                    U.ops.push_back({ k, src + PO[p] * (size_t) nbk, dst + PO[p] * (size_t) NB + PB[p] * (size_t) b0, PB[p] * (size_t) nbk, false, 0 });
                cover += d.a1 - d.a0;
                break;
            }
            case 1: {   // [gate rows | up rows] of the rank -> rows a0.. of each half of the fused tensor
                if (T.N != 2 * d.half) { err = "tp pager: " + T.name + " is not a stacked pair"; return false; }
                const int64_t nh = (d.a1 - d.a0) * bpr, nbk = 2 * nh;
                for (int p = 0; p < 4; ++p) {
                    U.ops.push_back({ k, src + PO[p] * (size_t) nbk, dst + PO[p] * (size_t) NB + PB[p] * (size_t) (d.a0 * bpr), PB[p] * (size_t) nh, false, 0 });
                    U.ops.push_back({ k, src + PO[p] * (size_t) nbk + PB[p] * (size_t) nh,
                                      dst + PO[p] * (size_t) NB + PB[p] * (size_t) ((d.half + d.a0) * bpr), PB[p] * (size_t) nh, false, 0 });
                }
                cover += 2 * (d.a1 - d.a0);
                break;
            }
            case 2: {   // K blocks a0..a1 of every row: through the staging (rank 0: from its shard on this card)
                const int64_t nbk = d.a1 - d.a0;
                const size_t sb = (size_t) T.N * (size_t) nbk * Q6B;
                if (k == 0) U.sc.push_back({ src, false, 0, dst, T.N, bpr, d.a0, nbk });
                else {
                    U.ops.push_back({ k, src, nullptr, sb, true, U.stg });
                    U.sc.push_back({ nullptr, true, U.stg, dst, T.N, bpr, d.a0, nbk });
                    U.stg += up_to(sb, 256);
                }
                cover += nbk;
                break;
            }
            case 3:
                U.ops.push_back({ k, src, dst + (size_t) d.a0 * d.slice, (size_t) (d.a1 - d.a0) * d.slice, false, 0 });
                cover += (int64_t) ((size_t) (d.a1 - d.a0) * d.slice / Q6B);
                break;
            default:
                err = "tp pager: shard kind";
                return false;
            }
        }
        // coverage: rows (kind 0 / 1), K blocks (2), blocks (3)
        Ds41TpShard d0;
        tp.shard_desc(T.name, 0, d0);
        const int64_t need = d0.kind == 2 ? bpr : d0.kind == 3 ? NB : T.N;
        if (cover != need) { err = "tp pager: the ranks' shards do not cover " + T.name; return false; }
        U.tens.emplace_back(dst, T.bytes);
    }
    size_t max_u = 0, max_s = 0;
    for (auto & U : im.units) { max_u = std::max(max_u, U.bytes); max_s = std::max(max_s, U.stg); }
    // ---- STRATA_TP_PAGE_CHECK=1: every unit refilled in place from the shards and compared with what was loaded
    if (env_on("STRATA_TP_PAGE_CHECK", false)) {
        uint8_t * stg = max_s ? sycl::malloc_device<uint8_t>(max_s, *im.q0) : nullptr;
        int64_t bad = 0;
        for (size_t u = 0; u < im.units.size(); ++u) {
            Impl::Unit & U = im.units[u];
            if (U.tens.empty()) continue;
            std::vector<uint8_t> a, b;
            for (auto & t : U.tens) {
                const size_t o = a.size();
                a.resize(o + t.second);
                im.q0->memcpy(a.data() + o, t.first, t.second);
            }
            im.q0->wait();
            for (auto & t : U.tens) im.q0->memset(t.first, 0, t.second);
            im.q0->wait();
            const auto far = im.copies_far(U, stg);
            im.local(U, stg, far).wait();
            b.resize(a.size());
            size_t o = 0;
            for (auto & t : U.tens) { im.q0->memcpy(b.data() + o, t.first, t.second); o += t.second; }
            im.q0->wait();
            int64_t d = 0;
            for (size_t i = 0; i < a.size(); ++i) d += a[i] != b[i];
            if (d) std::fprintf(stderr, "tp pager: layer %zu: %lld bytes differ after the refill\n", u, (long long) d);
            bad += d;
        }
        if (stg) sycl::free(stg, *im.q0);
        std::fprintf(stderr, "tp pager: check %s\n", bad ? "FAILED - paging off" : "ok (every unit refilled bit for bit)");
        if (bad) { err = "tp pager: refill check failed"; return false; }
        im.fill_bytes = 0;
    }
    im.keep = env_on("STRATA_TP_PAGE_KEEP", false);
    im.prefetch = env_on("STRATA_TP_PAGE_PREFETCH", true);
    if (im.keep) {   // debug: the pages stay; units refill in place (one staging)
        if (max_s && !(im.keep_stg = sycl::malloc_device<uint8_t>(max_s, *im.q0))) { err = "tp pager: staging failed"; return false; }
        im.on = true;
        std::fprintf(stderr, "tp pager: KEEP mode (debug) - every unit stays mapped and refills in place each pass\n");
        return true;
    }
    // ---- the load-time pages back, the slots.  The driver frees them behind the calls: wait (<= 10 s) until the card's
    // free VRAM shows them, so what is measured next (the dense card's expert budget) has them
    im.q0->wait();
    const sycl::device dev = im.q0->get_device();
    auto free_now = [&]() -> int64_t { return dev.has(sycl::aspect::ext_intel_free_memory) ? (int64_t) dev.get_info<sycl::ext::intel::info::device::free_memory>() : -1; };
    const int64_t f0 = free_now();
    for (auto & U : im.units) {
        if (!U.load) continue;
        if (zeVirtualMemUnmap(im.zc, im.va + U.off, U.bytes) != ZE_RESULT_SUCCESS || zePhysicalMemDestroy(im.zc, U.load) != ZE_RESULT_SUCCESS) {
            err = "tp pager: giving the load-time pages back failed";
            return false;
        }
        U.load = nullptr;
        im.freed_bytes += U.bytes;
    }
    const double tw = now_ms();
    int64_t f1 = free_now();
    while (f0 >= 0 && f1 - f0 < (int64_t) (im.freed_bytes * 9 / 10) && now_ms() - tw < 10000) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        f1 = free_now();
    }
    if (f0 >= 0) std::fprintf(stderr, "tp pager: card 0 free VRAM %.2f -> %.2f GiB (%.0f ms after the release)\n", f0 / 1073741824.0,
                              f1 / 1073741824.0, now_ms() - tw);
    const int ns = std::max(2, env_i("STRATA_TP_PAGE_SLOTS", 3));
    im.slots.assign((size_t) ns, {});
    for (auto & S : im.slots) {
        if (!im.phys(max_u, S.ph)) { err = "tp pager: slot pages failed"; return false; }
        if (max_s && !(S.stg = sycl::malloc_device<uint8_t>(max_s, *im.q0))) { err = "tp pager: staging failed"; return false; }
        im.slot_bytes += max_u + max_s;
    }
    im.on = true;
    std::fprintf(stderr, "tp pager: on - %.2f GiB given back, %d slots of %.0f + %.0f MiB (staging)\n", im.freed_bytes / 1073741824.0,
                 ns, max_u / 1048576.0, max_s / 1048576.0);
    return true;
}

void Ds41Pager::need(int layer) {
    Impl & im = *p_;
    static const int dummy_us = env_i("STRATA_TP_PAGE_DUMMY", 0);
    if (!im.on) {   // (not enabled: the debug timing hooks only)
        if (dummy_us <= 0 || !im.q0 || layer == im.last_need) return;
        im.last_need = layer;
        const int64_t spins = (int64_t) dummy_us * 400;
        int * sink = im.dummy ? im.dummy : (im.dummy = sycl::malloc_device<int>(1, *im.q0));
        im.q0->single_task([=] { int v = 0; for (int64_t k = 0; k < spins; ++k) v = v * 1103515245 + 12345; *sink = v; });
        return;
    }
    if (!im.on || layer < 0 || (size_t) layer >= im.units.size() || !im.units[(size_t) layer].bytes) return;
    Impl::Unit & U = im.units[(size_t) layer];
    if (im.keep) {   // debug: refill in place at a pass's first need of the unit
        if (layer < im.last_need) for (auto & V : im.units) V.filled = false;
        im.last_need = layer;
        if (!U.filled) {
            const auto far = im.copies_far(U, im.keep_stg);
            im.local(U, im.keep_stg, far);
            U.filled = true;
            ++im.n_fill;
        }
        return;
    }
    if (U.slot < 0) {
        ++im.n_miss;
        if (!im.start(layer, true)) { std::fprintf(stderr, "tp pager: %s\n", im.err.c_str()); std::abort(); }
    }
    im.slots[(size_t) U.slot].tick = ++im.clock;
    if (!U.filled) im.finish(layer);
    for (size_t l = (size_t) layer + 1; im.prefetch && l < im.units.size(); ++l) {   // the next layer with a unit: its copies start now
        if (!im.units[l].bytes) continue;
        if (im.units[l].slot < 0 && im.start((int) l, false, U.slot)) ++im.n_pref;
        break;
    }
}

void Ds41Pager::touch(int layer) {
    Impl & im = *p_;
    if (!im.on || layer < 0 || (size_t) layer >= im.units.size()) return;
    Impl::Unit & U = im.units[(size_t) layer];
    if (!im.keep && U.slot >= 0 && U.filled) im.slots[(size_t) U.slot].used = im.q0->ext_oneapi_submit_barrier();
}

// ---------------------------------------------------------------- Ds41KvMap
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
        if (!im.map(U.off + U.tail, U.bytes - U.tail, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return nullptr; }
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

std::string Ds41Pager::report() const {
    const Impl & im = *p_;
    char b[256];
    std::snprintf(b, sizeof b, "tp pager: %lld fills (%lld prefetched, %lld on demand), %.2f GB copied, %lld waits for a busy slot (%.1f ms)",
                  (long long) im.n_fill, (long long) im.n_pref, (long long) im.n_miss, im.fill_bytes / 1e9, (long long) im.n_wait, im.wait_ms);
    return b;
}

}  // namespace strata::v41
