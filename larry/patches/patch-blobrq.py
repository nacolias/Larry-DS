#!/usr/bin/env python3
# patch-blobrq.py [XOLOTL_DIR] - larry, 2026-10-29 (optimization #4: fewer bytes per cold expert): STRATA_BLOB_RQ=<file>
# - experts that ds41_requant (patch-requant-tool.py) requantized to a smaller class read from that file.  Ds41FileBlobs
# gives each of them its new class (gate / up and down types from the file's index: Q3_K / Q2_0 / PTQ1_0 pairs, a class
# appended if the model has none like it) and its blob's offsets there, so every reader - the arena's build, the decode
# misses' read_part, the prefetch's read_staged, advise - takes the requantized bytes, never the model file's or the
# mirror's.  A file built from another model file (its size) or out of range is an error at open.  Idempotent.
import os
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"
H = os.path.join(X, "tools/v41/engine/ds41_moe.hpp")
C = os.path.join(X, "tools/v41/engine/ds41_moe.cpp")
h, c = open(H).read(), open(C).read()
if "patch-blobrq.py" in c:
    print("already patched")
    sys.exit(0)


def rep(s, old, new, what):
    if s.count(old) != 1:
        sys.exit("%s: anchor x%d: %s" % (what, s.count(old), old[:90]))
    return s.replace(old, new)


h = rep(h, """    int64_t mirrored() const { return mirrored_; }""", """    int64_t mirrored() const { return mirrored_; }
    /// larry (patch-blobrq.py, STRATA_BLOB_RQ=<file>): experts read from ds41_requant's file, in their smaller class
    int64_t requantized() const { return requantized_; }""", "hpp public")
h = rep(h, """    bool open_mirror(const char* path, std::string& err);
    int mfd_ = -1;                        ///< the mirror (O_DIRECT), -1 = none""", """    bool open_mirror(const char* path, std::string& err);
    /// larry (patch-blobrq.py): the requant file's index - classes and offsets of its experts (open, before the sizes)
    bool open_rq(const char* path, std::string& err);
    int rfd_ = -1, rdfd_ = -1;            ///< the requant file (buffered / O_DIRECT), -1 = none
    std::vector<uint8_t> rq_;             ///< per (layer, expert): 1 = its blob is in the requant file
    int64_t requantized_ = 0;
    int fd_of(int64_t l, int64_t e) const { return !rq_.empty() && rq_[(size_t) (l * g_.n_experts + e)] ? rfd_ : fd_; }
    int dfd_of(int64_t l, int64_t e) const { return !rq_.empty() && rq_[(size_t) (l * g_.n_experts + e)] ? rdfd_ : dfd_; }
    int mfd_ = -1;                        ///< the mirror (O_DIRECT), -1 = none""", "hpp private")

c = rep(c, """    if (fd_ >= 0) ::close(fd_);
    if (dfd_ >= 0) ::close(dfd_);
    if (mfd_ >= 0) ::close(mfd_);""", """    if (fd_ >= 0) ::close(fd_);
    if (dfd_ >= 0) ::close(dfd_);
    if (mfd_ >= 0) ::close(mfd_);
    if (rfd_ >= 0) ::close(rfd_);   // larry (patch-blobrq.py)
    if (rdfd_ >= 0) ::close(rdfd_);""", "destructor")
c = rep(c, """    for (size_t c = 0; c < g_.cls_gu.size(); ++c)
        max_ = std::max<size_t>(max_, (size_t) (2 * (uint64_t) gu_rb_[c] * (uint64_t) FF + (uint64_t) d_rb_[c] * (uint64_t) D));""",
        """    // larry (patch-blobrq.py, STRATA_BLOB_RQ): the requantized experts' classes and offsets, before the sizes below
    if (const char* rqp = std::getenv("STRATA_BLOB_RQ"); rqp && rqp[0])
        if (!open_rq(rqp, err)) return false;
    for (size_t c = 0; c < g_.cls_gu.size(); ++c)
        max_ = std::max<size_t>(max_, (size_t) (2 * (uint64_t) gu_rb_[c] * (uint64_t) FF + (uint64_t) d_rb_[c] * (uint64_t) D));""",
        "open")
c = rep(c, """// The mirror file: a 4096-byte-aligned header, then the blobs, each [gate | up | down] padded to 4096.""",
        """// larry (patch-blobrq.py) - the requant file (ds41_requant): a 4096-aligned header {"S41RQNT1", u64 model file size,
// u64 n, u64 data start}, n x {i32 layer, i32 expert, i32 gate/up type, i32 down type, u64 offset}, the blobs.
namespace {
struct RqHead { char magic[8]; uint64_t src_size, n, data; };
struct RqEnt { int32_t l, e, gu, d; uint64_t off; };
}

bool Ds41FileBlobs::open_rq(const char* path, std::string& err) {
    const int fd = ::open(path, O_RDONLY);
    if (fd < 0) { err = std::string("STRATA_BLOB_RQ: cannot open ") + path; return false; }
    auto fail = [&](const std::string& m) { ::close(fd); err = std::string("STRATA_BLOB_RQ ") + path + ": " + m; return false; };
    struct stat st{}, ss{};
    if (::fstat(fd, &st) != 0 || ::stat(path_.c_str(), &ss) != 0) return fail("stat");
    RqHead Hh{};
    if (::pread(fd, &Hh, sizeof Hh, 0) != (ssize_t) sizeof Hh) return fail("short header");
    if (std::memcmp(Hh.magic, "S41RQNT1", 8) != 0) return fail("bad magic");
    if (Hh.src_size != (uint64_t) ss.st_size) return fail("built from a different model file (size)");
    const uint64_t NE = (uint64_t) (g_.n_layers * g_.n_experts);
    if (Hh.n == 0 || Hh.n > NE || Hh.data < sizeof Hh + Hh.n * sizeof(RqEnt) || (Hh.data & 4095)) return fail("bad header");
    std::vector<RqEnt> ents((size_t) Hh.n);
    if (::pread(fd, ents.data(), ents.size() * sizeof(RqEnt), (off_t) sizeof Hh) != (ssize_t) (ents.size() * sizeof(RqEnt)))
        return fail("short index");
    auto rb = [](int t, int64_t n) {
        int be = 0, bb = 0;
        block_geometry((uint32_t) t, be, bb);
        return (uint64_t) (n / be) * (uint64_t) bb;
    };
    const int64_t D = g_.n_embd, FF = g_.n_ff;
    rq_.assign((size_t) NE, 0);
    int added = 0;
    for (const RqEnt& x : ents) {
        if (x.l < 0 || x.l >= g_.n_layers || x.e < 0 || x.e >= g_.n_experts) return fail("bad index entry");
        for (int t : {x.gu, x.d})
            if (t != 11 && t != 42 && t != 143) return fail("type " + std::to_string(t) + " is not an expert tier");
        int cl = -1;
        for (size_t k = 0; k < g_.cls_gu.size(); ++k)
            if (g_.cls_gu[k] == x.gu && g_.cls_d[k] == x.d) cl = (int) k;
        if (cl < 0) {
            cl = (int) g_.cls_gu.size();
            g_.cls_gu.push_back(x.gu);
            g_.cls_d.push_back(x.d);
            gu_rb_.push_back((uint32_t) rb(x.gu, D));
            d_rb_.push_back((uint32_t) rb(x.d, FF));
            ++added;
        }
        const uint64_t gb = (uint64_t) gu_rb_[(size_t) cl] * (uint64_t) FF, db = (uint64_t) d_rb_[(size_t) cl] * (uint64_t) D;
        if ((x.off & 4095) || x.off < Hh.data || x.off + 2 * gb + db > (uint64_t) st.st_size) return fail("blob out of range");
        const size_t i = (size_t) (x.l * g_.n_experts + x.e);
        g_.eclass[i] = (int16_t) cl;
        gate_off_[i] = x.off;
        up_off_[i] = x.off + gb;
        down_off_[i] = x.off + 2 * gb;
        rq_[i] = 1;
    }
    rfd_ = fd;
    rdfd_ = ::open(path, O_RDONLY | O_DIRECT);
    ::posix_fadvise(rfd_, 0, 0, POSIX_FADV_RANDOM);
    requantized_ = (int64_t) ents.size();
    std::fprintf(stderr, "blob requant: %s - %lld experts in smaller classes (%.1f GiB), %d class(es) added\\n", path,
                 (long long) ents.size(), (double) ((uint64_t) st.st_size - Hh.data) / 1073741824.0, added);
    return true;
}

// The mirror file: a 4096-byte-aligned header, then the blobs, each [gate | up | down] padded to 4096.""", "open_rq")
# readers: the requant file's fds for its experts, never the mirror
c = rep(c, """bool Ds41FileBlobs::read_direct(uint64_t off, size_t n, uint8_t* dst, uint8_t* scratch, size_t scratch_bytes, int fd) const {
    if (fd < 0) fd = dfd_;""", """bool Ds41FileBlobs::read_direct(uint64_t off, size_t n, uint8_t* dst, uint8_t* scratch, size_t scratch_bytes, int fd) const {
    if (fd == -2) return false;   // larry (patch-blobrq.py): a requant file without an O_DIRECT twin
    if (fd < 0) fd = dfd_;""", "read_direct")
c = rep(c, """void Ds41FileBlobs::read_staged(int64_t layer, int64_t expert, uint8_t* dst, uint8_t* scratch, size_t scratch_bytes) const {
    if (dfd_ < 0 || scratch_bytes < scratch_) {
        read(layer, expert, dst);
        return;
    }
    Span s[3];
    spans(layer, expert, s);
    uint64_t at = 0;
    for (const Span& x : s) {
        if (!read_direct(x.off, (size_t) x.bytes, dst + at, scratch, scratch_bytes)) {""",
        """void Ds41FileBlobs::read_staged(int64_t layer, int64_t expert, uint8_t* dst, uint8_t* scratch, size_t scratch_bytes) const {
    const int dfd = dfd_of(layer, expert);   // larry (patch-blobrq.py)
    if (dfd < 0 || scratch_bytes < scratch_) {
        read(layer, expert, dst);
        return;
    }
    Span s[3];
    spans(layer, expert, s);
    uint64_t at = 0;
    for (const Span& x : s) {
        if (!read_direct(x.off, (size_t) x.bytes, dst + at, scratch, scratch_bytes, dfd)) {""", "read_staged")
c = rep(c, """void Ds41FileBlobs::advise(int64_t layer, int64_t expert) const {
    if (fd_ < 0) return;
    Span s[3];
    spans(layer, expert, s);
    for (const Span& x : s)
        if (x.bytes) ::posix_fadvise(fd_, (off_t) x.off, (off_t) x.bytes, POSIX_FADV_WILLNEED);""",
        """void Ds41FileBlobs::advise(int64_t layer, int64_t expert) const {
    const int fd = fd_of(layer, expert);   // larry (patch-blobrq.py)
    if (fd < 0) return;
    Span s[3];
    spans(layer, expert, s);
    for (const Span& x : s)
        if (x.bytes) ::posix_fadvise(fd, (off_t) x.off, (off_t) x.bytes, POSIX_FADV_WILLNEED);""", "advise")
c = rep(c, """void Ds41FileBlobs::read(int64_t layer, int64_t expert, uint8_t* dst) const {
    Span s[3];
    spans(layer, expert, s);
    uint64_t at = 0;
    for (const Span& x : s) {
        pread_all(fd_, dst + at, x.off, x.bytes);""", """void Ds41FileBlobs::read(int64_t layer, int64_t expert, uint8_t* dst) const {
    Span s[3];
    spans(layer, expert, s);
    const int fd = fd_of(layer, expert);   // larry (patch-blobrq.py)
    uint64_t at = 0;
    for (const Span& x : s) {
        pread_all(fd, dst + at, x.off, x.bytes);""", "read")
c = rep(c, """    uint64_t at = 0;
    uint8_t* bb = part_direct_ && dfd_ >= 0 ? take_bounce() : nullptr;""", """    uint64_t at = 0;
    const bool rqe = !rq_.empty() && rq_[(size_t) (layer * g_.n_experts + expert)];   // larry (patch-blobrq.py)
    const int fd = rqe ? rfd_ : fd_, dfd = rqe ? (rdfd_ >= 0 ? rdfd_ : -2) : dfd_;
    uint8_t* bb = part_direct_ && (rqe ? rdfd_ : dfd_) >= 0 ? take_bounce() : nullptr;""", "read_part head")
c = rep(c, """    if (mfd_ >= 0 && bb && nparts > 1 && ((part + 1) * mshare_) / 1000 != (part * mshare_) / 1000) {""",
        """    if (!rqe && mfd_ >= 0 && bb && nparts > 1 && ((part + 1) * mshare_) / 1000 != (part * mshare_) / 1000) {""",
        "read_part mirror")
c = rep(c, """        if (a < b && !(bb && read_direct(x.off + (a - at), (size_t) (b - a), dst + a, bb, scratch_)))
            pread_all(fd_, dst + a, x.off + (a - at), b - a);""", """        if (a < b && !(bb && read_direct(x.off + (a - at), (size_t) (b - a), dst + a, bb, scratch_, dfd)))
            pread_all(fd, dst + a, x.off + (a - at), b - a);""", "read_part spans")
open(H, "w").write(h)
open(C, "w").write(c)
print("patched", H, C)
