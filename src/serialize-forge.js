/**
 * @forge-module serialize-forge
 * @layer L7
 * @version 1.0.0
 * @brief     序列化与存储内核（F7-G5 全 16 项）：规范 JSON / 二进制 TLV / 列式（字典+位包）/ protobuf varint+zigzag 四编解码器；分页内存映射（缺页+LRU逐出+脏页写回）；增量 diff-apply；COW 深冻结快照；不可变版本存储；WAL 事务（先写日志+两阶段恢复）；灾难恢复（备份校验+PITR）；哈希/一致性环/范围三分区；内容寻址对象存储（sha 去重+版本）；LRU 缓存与 TTL 远端缓存（stale-while-revalidate）；CRC32/FNV-1a-64/SHA-256 校验和家族（已知向量+雪崩+块定位）
 * @reuse     RFC 8259 The JSON Data Interchange Format|https://www.rfc-editor.org/rfc/rfc8259|IETF STD 90|A|JSON 数据模型与 UTF-8 文本序列化
 * @reuse     RFC 8785 JSON Canonicalization Scheme (JCS)|https://www.rfc-editor.org/rfc/rfc8785|IETF|A|规范 JSON：对象键排序+无空白，字节级确定性
 * @reuse     Protocol Buffers Encoding Guide|https://protobuf.dev/programming-guides/encoding/|Apache-2.0 文档|A|wire type/varint/zigzag/长度分隔字段
 * @reuse     LEB128 (Little-Endian Base 128)|https://en.wikipedia.org/wiki/LEB128|CC BY-SA|B|无符号 varint base-128 小端字节序
 * @reuse     Williams《A Painless Guide to CRC Error Detection Algorithms》|https://www.zlib.net/crc_v3.txt|公有领域文章|A|CRC-32 IEEE 反射多项式 0xEDB88320 与查表驱动
 * @reuse     FNV-1a 64-bit (Fowler/Noll/Vo)|http://www.isthe.com/chongo/tech/comp/fnv/|公有领域参考|A|offset basis 0xCBF29CE484222325、prime 0x100000001B3、异或后乘
 * @reuse     Parquet 列式存储格式|https://parquet.apache.org/docs/|Apache-2.0|B|列式布局+字典编码+位打包索引
 * @reuse     Rosenblum & Ousterhout《The Design and Implementation of a Log-Structured File System》(1992)|https://doi.org/10.1145/121056.121057|ACM|A|COW 快照/段追加/日志结构存储
 * @reuse     Mohan et al.《ARIES: A Transaction Recovery Method Supporting Fine-Granularity Locking and Partial Rollbacks Using Write-Ahead Logging》(1992)|https://doi.org/10.1145/128765.128770|ACM|A|WAL 先写日志+redo 已提交事务+原子恢复+部分尾记录丢弃
 * @reuse     Karger et al.《Consistent Hashing and Random Trees》(1997)|https://doi.org/10.1145/258533.258660|ACM|A|哈希环+虚拟节点+增删节点移动键数 ≈1/(n+1)
 * @reuse     Mattson et al.《Evaluation Techniques for Storage Hierarchies》(1970)|https://doi.org/10.1147/rd.92.0078|IBM|B|LRU 栈性质与逐出序
 * @reuse     RFC 5861 HTTP Cache-Control Extensions (stale-while-revalidate)|https://www.rfc-editor.org/rfc/rfc5861|IETF|B|远端缓存 TTL 过期+过期重验语义
 * @reuse     Git 对象模型（内容寻址存储）|https://git-scm.com/book/en/v2/Git-Internals-Git-Objects|CC BY-NC-SA|B|sha 内容寻址+同内容去重+不可变对象
 * @reuse     POSIX mmap|https://pubs.opengroup.org/onlinepubs/9699919799/functions/mmap.html|IEEE|B|页式映射/缺页/脏页写回语义
 * @beyond    纯 JS 单文件零依赖零构建，双端同构 UMD；F7-G5 全 16 项组织为一条自洽存储栈（编解码→分页映射→增量/快照→版本/事务/灾恢→分区/对象→缓存→校验和），各组件可独立使用亦互相组合（WAL 记录即二进制编解码、灾恢备份即校验帧）。反空洞铁律贯穿：四编解码器对随机对象图 roundtrip EXACT 且规范编码字节级确定（键序无关）；紧凑性声明必配对立面（列式低基数必赢、高基数必亏——无免费午餐）；任何损坏流（魔数/截断/超长 varint/翻转字节）必被检出或抛错；WAL 在每个截断前缀下恢复态 EXACT 等于最后提交态（原子性+持久性全前缀证明）；LRU/TTL 逐出与过期用脚本序列+注入时钟 EXACT 断言；校验和配标准测试向量+单比特雪崩统计+块级定位；一致性哈希重平衡移动比例双向断言（必动且不过界）。
 * @invariant INV-SER-001|四编解码器 roundtrip 保真与规范确定性：JSON(规范键序)/二进制(TLV)/列式(字典+位包)/protobuf(varint+zigzag) 对同一随机对象图（3 种子）decode 后 deepEqual EXACT；encodeJson 与 encodeBinary 键序无关（同值不同键序 → 字节级 EXACT 相等）；紧凑性：数值负载 binary ≤0.75·json、小整数 protobuf ≤0.6·binary；损坏流（魔数翻转/截断/超长 varint/尾随垃圾）必抛错|0|deepEqual EXACT + 字节级对拍 + zigzag 已知向量 + 损坏必抛
 * @invariant INV-SER-002|列式 vs 行式无免费午餐：低基数列（每列 4 个 distinct 值）列式 ≤0.7·行式（字典+位包致胜）；高基数列（每列全 distinct int）列式 ≥1.15·行式（字典开销致亏）；两种负载 decode roundtrip EXACT 且同表编码字节确定 EXACT（异 seed 同表同字节）|0|字典+位打包尺寸双向对比 + roundtrip + 确定性对拍
 * @invariant INV-SER-003|内存映射分页语义：脚本访问序列下 LRU 逐出序 EXACT（构造序列+对拍）；缺页计数 == 首次触页数 EXACT；脏页 flush 后 backing 与映射视图逐字节一致 EXACT 且 flush 前 backing 不含该写（写回真实）；越界读/写均抛错|0|页表 LRU 对拍 + 写回前后一致性 + OOB 证伪
 * @invariant INV-SER-004|增量编码与快照隔离：applyDelta(base, encodeDelta(base,target))==target EXACT（3 种子随机变异：删/插/换段）；同输入 delta 操作序列与字节确定 EXACT；空变异（base==target）delta 仅剩单 copy 且 ≤16 字节；快照深冻结隔离——快照后写入不影响快照 EXACT、restore 后状态==快照 EXACT、版本历史不灭（恢复不销毁后续版本）、快照对象 Object.isFrozen 且嵌套全冻|0|diff-apply 对拍 + COW 隔离 + 深冻结 + 历史保全
 * @invariant INV-SER-005|版本化与事务持久性：版本 id 严格递增且旧版本在后续写入后内容不变 EXACT；WAL 脚本事务（两事务交错+未提交+中止+同键覆写）恢复态 EXACT（仅已提交、按 lsn 序）；每个截断前缀 p∈[0,N] 恢复态 == p 前最后提交态 EXACT（原子性+持久性全前缀证明，部分尾帧丢弃）；抽去提交记录的日志中对应写入不可见（证伪）；灾恢 backup→restore EXACT、备份翻转一字节必抛、PITR 到任一提交点状态 EXACT|0|两阶段恢复 + 全前缀崩溃模拟 + 备份双校验 + 提交抽离证伪
 * @invariant INV-SER-006|校验和家族：已知向量 EXACT（CRC32("123456789")=0xCBF43926、CRC32("")=0、FNV-1a-64("")=0xcbf29ce484222325、FNV-1a-64("a")=0xaf63dc4c8601ec8c、SHA256("abc")=ba7816bf…、SHA256("")=e3b0c442…）；单比特翻转雪崩（3 种子×20 次，CRC32 与 FNV-1a-64 输出差异位比例均值 ≥0.3 且 ≤0.75）；1000 随机输入无碰撞 EXACT；块级校验（16 块）定位损坏块 EXACT；记录级翻转仅污染对应记录（corrupted==[该键] EXACT，其余完好）|0|标准测试向量 + 雪崩统计 + 定位对拍 + 选择性污染
 * @invariant INV-SER-007|分区与对象存储：哈希分区确定（同 key 同分区，重复 100 次 EXACT）且 8 桶近似均匀（最大桶偏差 ≤50%）；一致性哈希环（4 节点×50 虚节点）：同 key 同节点 EXACT、增加 1 节点移动键比例 ≤2/(n+1) 且 ≥0.02（必动不过界，证伪空跑）、删除 1 节点 ≤0.5；范围分区（边界 h/p/x）路由 EXACT；对象存储内容寻址去重（同内容两键 → bodies 单副本 EXACT）、同键覆写产生新版本且旧版本内容 EXACT、head 元数据 EXACT、删除后 get 抛错、异内容 etag 必异|0|确定性+均匀性+移动界双向+去重与版本对拍
 * @invariant INV-SER-008|缓存语义：LRU（容量 3）脚本序列逐出序 EXACT、size≤capacity 恒成立 EXACT、hits+misses==gets EXACT、capacity=1 连续 put 必逐出前值（证伪）；TTL 可注入时钟：过期前 hit / 过期 miss EXACT、ttl=0 即 miss、stale-while-revalidate 过期仍返回旧值且置 revalidate、refresh 后新值生效 EXACT、过期项被惰性清除|0|Map 插入序对拍 + 注入时钟 + SWR 语义 + 边界证伪
 * @seed      0x9E3779B9
 */
'use strict';

/* ============================================================================
 * 项目统一容差（九档，与 core-verify.TOL 同值，本地声明保证零依赖）
 * ========================================================================== */
const TOL = { EXACT: 0, ULP: 1e-16, TIGHT: 1e-15, ORTHO: 1e-14, RESID: 1e-12, SOLVE: 1e-10, GRAD: 1e-6, MONO: -1e-9, CONV: 1e-6 };

/* ============================================================================
 * 私有可种子 PRNG：sfc32（同 seed 序列逐位可复现；异 seed 轨道分歧）
 * ========================================================================== */
function makeRng(seed) {
  let a = (seed >>> 0) ^ 0x9E3779B9;
  let b = ((seed >>> 0) + 0x6D2B79F5) >>> 0;
  let c = (seed >>> 0) ^ 0xC2B2AE35;
  let d = ((seed >>> 0) + 0x165667B1) >>> 0;
  function next() {
    a >>>= 0; b >>>= 0; c >>>= 0; d >>>= 0;
    let t = (a + b + d) >>> 0;
    d = (d + 1) >>> 0;
    a = b ^ (b >>> 9);
    b = c + (c << 3);
    c = (c << 21) | (c >>> 11);
    c = (c + t) >>> 0;
    return (t >>> 0) / 4294967296;
  }
  for (let i = 0; i < 12; i++) next();
  return next;
}

/* ============================================================================
 * 纯 JS SHA-256（FIPS 180-4）：hash 幂等 + 'sha256:' 前缀契约
 * 双入口：sha256Hex(str) 走 UTF-8；sha256BytesHex(bytes) 直接吃字节
 * ========================================================================== */
const K256 = [
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
];
function _rotr(x, n) { return ((x >>> n) | (x << (32 - n))) >>> 0; }

function _sha256Bytes(bytes) {
  const msg = Array.from(bytes);
  const bitLen = msg.length * 8;
  msg.push(0x80);
  while (msg.length % 64 !== 56) msg.push(0);
  const hi = Math.floor(bitLen / 0x100000000), lo = bitLen >>> 0;
  msg.push((hi >>> 24) & 255, (hi >>> 16) & 255, (hi >>> 8) & 255, hi & 255,
    (lo >>> 24) & 255, (lo >>> 16) & 255, (lo >>> 8) & 255, lo & 255);
  let H = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
  const w = new Array(64);
  for (let off = 0; off < msg.length; off += 64) {
    for (let t = 0; t < 16; t++) {
      w[t] = ((msg[off + 4 * t] << 24) | (msg[off + 4 * t + 1] << 16) | (msg[off + 4 * t + 2] << 8) | msg[off + 4 * t + 3]) | 0;
    }
    for (let t = 16; t < 64; t++) {
      const x15 = w[t - 15], x2 = w[t - 2];
      const s0 = (_rotr(x15, 7) ^ _rotr(x15, 18) ^ (x15 >>> 3)) | 0;
      const s1 = (_rotr(x2, 17) ^ _rotr(x2, 19) ^ (x2 >>> 10)) | 0;
      w[t] = (w[t - 16] + s0 + w[t - 7] + s1) | 0;
    }
    let a = H[0], b = H[1], c = H[2], d = H[3], e = H[4], f = H[5], g = H[6], h = H[7];
    for (let t = 0; t < 64; t++) {
      const S1 = (_rotr(e, 6) ^ _rotr(e, 11) ^ _rotr(e, 25)) | 0;
      const ch = ((e & f) ^ (~e & g)) | 0;
      const t1 = (h + S1 + ch + K256[t] + w[t]) | 0;
      const S0 = (_rotr(a, 2) ^ _rotr(a, 13) ^ _rotr(a, 22)) | 0;
      const maj = ((a & b) ^ (a & c) ^ (b & c)) | 0;
      const t2 = (S0 + maj) | 0;
      h = g; g = f; f = e; e = (d + t1) | 0; d = c; c = b; b = a; a = (t1 + t2) | 0;
    }
    H = [(H[0] + a) | 0, (H[1] + b) | 0, (H[2] + c) | 0, (H[3] + d) | 0, (H[4] + e) | 0, (H[5] + f) | 0, (H[6] + g) | 0, (H[7] + h) | 0];
  }
  return H.map((x) => ('00000000' + (x >>> 0).toString(16)).slice(-8)).join('');
}

function sha256Hex(str) {
  const bytes = [];
  for (let i = 0; i < str.length; i++) {
    let code = str.charCodeAt(i);
    if (code < 0x80) bytes.push(code);
    else if (code < 0x800) bytes.push(0xC0 | (code >> 6), 0x80 | (code & 63));
    else if (code >= 0xD800 && code <= 0xDBFF && i + 1 < str.length && str.charCodeAt(i + 1) >= 0xDC00 && str.charCodeAt(i + 1) <= 0xDFFF) {
      code = 0x10000 + ((code - 0xD800) << 10) + (str.charCodeAt(i + 1) - 0xDC00); i++;
      bytes.push(0xF0 | (code >> 18), 0x80 | ((code >> 12) & 63), 0x80 | ((code >> 6) & 63), 0x80 | (code & 63));
    } else bytes.push(0xE0 | (code >> 12), 0x80 | ((code >> 6) & 63), 0x80 | (code & 63));
  }
  return _sha256Bytes(bytes);
}

function sha256BytesHex(bytes) { return _sha256Bytes(bytes); }

/* ============================================================================
 * 字节层原语：UTF-8 / hex / 相等 / CRC-32 / FNV-1a-64
 * ========================================================================== */
function utf8Bytes(str) {
  const out = [];
  for (let i = 0; i < str.length; i++) {
    const c = str.charCodeAt(i);
    if (c < 0x80) out.push(c);
    else if (c < 0x800) out.push(0xC0 | (c >> 6), 0x80 | (c & 63));
    else if (c >= 0xD800 && c <= 0xDBFF && i + 1 < str.length && str.charCodeAt(i + 1) >= 0xDC00 && str.charCodeAt(i + 1) <= 0xDFFF) {
      const c2 = str.charCodeAt(++i);
      const cp = 0x10000 + ((c - 0xD800) << 10) + (c2 - 0xDC00);
      out.push(0xF0 | (cp >> 18), 0x80 | ((cp >> 12) & 63), 0x80 | ((cp >> 6) & 63), 0x80 | (cp & 63));
    } else out.push(0xE0 | (c >> 12), 0x80 | ((c >> 6) & 63), 0x80 | (c & 63));
  }
  return new Uint8Array(out);
}

function utf8Decode(bytes) {
  let s = '';
  for (let i = 0; i < bytes.length;) {
    const b = bytes[i++];
    if (b < 0x80) s += String.fromCharCode(b);
    else if (b < 0xE0) s += String.fromCharCode(((b & 31) << 6) | (bytes[i++] & 63));
    else if (b < 0xF0) s += String.fromCharCode(((b & 15) << 12) | ((bytes[i++] & 63) << 6) | (bytes[i++] & 63));
    else {
      const cp = (((b & 7) << 18) | ((bytes[i++] & 63) << 12) | ((bytes[i++] & 63) << 6) | (bytes[i++] & 63)) - 0x10000;
      s += String.fromCharCode(0xD800 + (cp >> 10), 0xDC00 + (cp & 1023));
    }
  }
  return s;
}

function bytesToHex(bytes) {
  let s = '';
  for (let i = 0; i < bytes.length; i++) s += ('0' + bytes[i].toString(16)).slice(-2);
  return s;
}

function hexToBytes(hex) {
  if (hex.length % 2 !== 0) throw new Error('hex length must be even');
  const out = new Uint8Array(hex.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = parseInt(hex.substr(2 * i, 2), 16);
  return out;
}

function bytesEqual(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

const CRC_TAB = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
    t[n] = c >>> 0;
  }
  return t;
})();

/** CRC-32/ISO-HDLC（反射、poly 0xEDB88320、初值/末值均取反） */
function crc32(bytes) {
  let c = 0xFFFFFFFF;
  for (let i = 0; i < bytes.length; i++) c = (CRC_TAB[(c ^ bytes[i]) & 255] ^ (c >>> 8)) >>> 0;
  return (c ^ 0xFFFFFFFF) >>> 0;
}

const FNV_OFFSET = 0xCBF29CE484222325n;
const FNV_PRIME = 0x100000001B3n;
const MASK64 = 0xFFFFFFFFFFFFFFFFn;

/** FNV-1a 64-bit（BigInt 精确 64 位算术，异或后乘 prime） */
function fnv1a64(bytes) {
  let h = FNV_OFFSET;
  for (let i = 0; i < bytes.length; i++) h = ((h ^ BigInt(bytes[i] & 255)) * FNV_PRIME) & MASK64;
  return h;
}

function fnv1a64Hex(bytes) { return fnv1a64(bytes).toString(16).padStart(16, '0'); }

/* ============================================================================
 * 字节流读写（二进制编解码 / protobuf / 帧 的共同基座）
 * ========================================================================== */
class ByteWriter {
  constructor() { this.buf = []; }
  u8(v) { this.buf.push(v & 255); return this; }
  bytes(b) { for (let i = 0; i < b.length; i++) this.buf.push(b[i] & 255); return this; }
  u32le(v) { this.buf.push(v & 255, (v >>> 8) & 255, (v >>> 16) & 255, (v >>> 24) & 255); return this; }
  f64(v) {
    const ab = new ArrayBuffer(8);
    new Float64Array(ab)[0] = v;
    const u = new Uint8Array(ab);
    for (let i = 0; i < 8; i++) this.buf.push(u[i]);
    return this;
  }
  /** LEB128 无符号 varint（u32 语义：≤5 字节） */
  varint(n) {
    if (!Number.isInteger(n) || n < 0) throw new Error('varint requires non-negative integer');
    let x = n;
    while (x >= 0x80) { this.buf.push((x % 128) + 0x80); x = Math.floor(x / 128); }
    this.buf.push(x);
    return this;
  }
  toBytes() { return new Uint8Array(this.buf); }
}

class ByteReader {
  constructor(bytes, pos) { this.b = bytes; this.p = pos || 0; }
  get eof() { return this.p >= this.b.length; }
  u8() { if (this.p >= this.b.length) throw new Error('truncated stream'); return this.b[this.p++]; }
  bytes(n) {
    if (n < 0 || this.p + n > this.b.length) throw new Error('truncated stream');
    const o = this.b.slice(this.p, this.p + n);
    this.p += n;
    return o;
  }
  u32le() {
    const o = this.bytes(4);
    return ((o[0]) | (o[1] << 8) | (o[2] << 16) | (o[3] << 24)) >>> 0;
  }
  f64() {
    const o = this.bytes(8);
    const ab = new ArrayBuffer(8);
    new Uint8Array(ab).set(o);
    return new Float64Array(ab)[0];
  }
  varint() {
    let result = 0, shift = 1, count = 0;
    for (;;) {
      const b = this.u8();
      count++;
      if (count > 5) throw new Error('varint too long (u32 semantics)');
      result += (b & 0x7f) * shift;
      if ((b & 0x80) === 0) return result;
      shift *= 128;
    }
  }
}

/* zigzag：有符号 ↔ 无符号（protobuf sint 语义） */
function zigzag32(n) { return ((n << 1) ^ (n >> 31)) >>> 0; }
function unzigzag32(v) { return ((v >>> 1) ^ -(v & 1)) | 0; }

function writeVarintTo(w, n) { w.varint(n); }
function readVarintFrom(r) { return r.varint(); }

/* ============================================================================
 * 深度相等 / 深克隆 / 深冻结
 * ========================================================================== */
function deepEqual(a, b) {
  if (a === b) return true;
  if (typeof a !== typeof b) return false;
  if (a === null || b === null || typeof a !== 'object') return false;
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  if (Array.isArray(a)) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (!deepEqual(a[i], b[i])) return false;
    return true;
  }
  const ka = Object.keys(a), kb = Object.keys(b);
  if (ka.length !== kb.length) return false;
  for (const k of ka) {
    if (!Object.prototype.hasOwnProperty.call(b, k)) return false;
    if (!deepEqual(a[k], b[k])) return false;
  }
  return true;
}

function deepClone(v) { return JSON.parse(JSON.stringify(v)); }

function deepFreeze(o) {
  if (o && typeof o === 'object' && !Object.isFrozen(o)) {
    Object.freeze(o);
    for (const k of Object.keys(o)) deepFreeze(o[k]);
  }
  return o;
}

function mapEqual(a, b) {
  if (a.size !== b.size) return false;
  for (const [k, v] of a) { if (!b.has(k) || !deepEqual(v, b.get(k))) return false; }
  return true;
}

/* ============================================================================
 * 编解码器 1：规范 JSON（RFC 8785 风格：键排序 + 无空白 + UTF-8）
 * ========================================================================== */
function encodeJson(v) {
  if (v === null) return 'null';
  if (v === true) return 'true';
  if (v === false) return 'false';
  if (typeof v === 'number') {
    if (!isFinite(v)) throw new Error('JSON codec requires finite numbers');
    return JSON.stringify(v);
  }
  if (typeof v === 'string') return JSON.stringify(v);
  if (Array.isArray(v)) return '[' + v.map(encodeJson).join(',') + ']';
  if (typeof v === 'object') {
    const keys = Object.keys(v).sort();
    return '{' + keys.map((k) => JSON.stringify(k) + ':' + encodeJson(v[k])).join(',') + '}';
  }
  throw new Error('JSON codec: unsupported type ' + typeof v);
}

function decodeJson(str) {
  const v = JSON.parse(str);
  if (!isFinite(v) && typeof v === 'number') throw new Error('JSON codec: non-finite number');
  return v;
}

/* ============================================================================
 * 编解码器 2：二进制 TLV（magic 'S' + 版本 + 类型标签值流）
 *   0x00 null 0x01 false 0x02 true 0x03 int32(LE4) 0x04 float64(LE8)
 *   0x05 string(varint len + utf8) 0x06 array 0x07 object(键排序)
 * ========================================================================== */
const BIN_MAGIC = 0x53, BIN_VERSION = 0x01;

function _encValue(w, v) {
  if (v === null) { w.u8(0x00); return; }
  if (v === false) { w.u8(0x01); return; }
  if (v === true) { w.u8(0x02); return; }
  const t = typeof v;
  if (t === 'number') {
    if (Number.isInteger(v) && v >= -2147483648 && v <= 2147483647) { w.u8(0x03); w.u32le(v >>> 0); return; }
    w.u8(0x04); w.f64(v); return;
  }
  if (t === 'string') {
    const b = utf8Bytes(v);
    w.u8(0x05); w.varint(b.length); w.bytes(b); return;
  }
  if (Array.isArray(v)) {
    w.u8(0x06); w.varint(v.length);
    for (const x of v) _encValue(w, x);
    return;
  }
  if (t === 'object') {
    const keys = Object.keys(v).sort();
    w.u8(0x07); w.varint(keys.length);
    for (const k of keys) {
      const kb = utf8Bytes(k);
      w.varint(kb.length); w.bytes(kb);
      _encValue(w, v[k]);
    }
    return;
  }
  throw new Error('binary codec: unsupported type ' + t);
}

function _decValue(r) {
  const tag = r.u8();
  switch (tag) {
    case 0x00: return null;
    case 0x01: return false;
    case 0x02: return true;
    case 0x03: return r.u32le() | 0;
    case 0x04: return r.f64();
    case 0x05: { const n = r.varint(); return utf8Decode(r.bytes(n)); }
    case 0x06: {
      const n = r.varint();
      const a = new Array(n);
      for (let i = 0; i < n; i++) a[i] = _decValue(r);
      return a;
    }
    case 0x07: {
      const n = r.varint();
      const o = {};
      for (let i = 0; i < n; i++) {
        const kl = r.varint();
        const k = utf8Decode(r.bytes(kl));
        o[k] = _decValue(r);
      }
      return o;
    }
    default: throw new Error('binary codec: unknown tag 0x' + tag.toString(16));
  }
}

function encodeBinary(v) {
  const w = new ByteWriter();
  w.u8(BIN_MAGIC); w.u8(BIN_VERSION);
  _encValue(w, v);
  return w.toBytes();
}

function decodeBinary(bytes) {
  const r = new ByteReader(bytes);
  if (r.u8() !== BIN_MAGIC) throw new Error('binary codec: bad magic');
  if (r.u8() !== BIN_VERSION) throw new Error('binary codec: bad version');
  const v = _decValue(r);
  if (!r.eof) throw new Error('binary codec: trailing garbage');
  return v;
}

/* ============================================================================
 * 编解码器 3：列式（字典编码 + 位打包索引）
 *   头部：'C' 版本 nRows nCols 列名表；每列：值类型标签 + 字典 + 位宽 + 位包
 * ========================================================================== */
const COL_MAGIC = 0x43, COL_VERSION = 0x01;

function _typeTagOf(v) {
  if (v === null) return 0x00;
  if (v === false) return 0x01;
  if (v === true) return 0x02;
  if (typeof v === 'number') return Number.isInteger(v) && v >= -2147483648 && v <= 2147483647 ? 0x03 : 0x04;
  if (typeof v === 'string') return 0x05;
  if (Array.isArray(v)) return 0x06;
  return 0x07;
}

function _dictKey(v) {
  if (v === null) return 'null';
  const t = typeof v;
  if (t === 'number' || t === 'boolean' || t === 'string') return t + ':' + String(v);
  return t + ':' + encodeJson(v);
}

class BitWriter {
  constructor() { this.cur = 0; this.n = 0; this.out = []; }
  write(v, bits) {
    for (let i = bits - 1; i >= 0; i--) {
      this.cur = (this.cur << 1) | ((v >>> i) & 1);
      this.n++;
      if (this.n === 8) { this.out.push(this.cur & 255); this.cur = 0; this.n = 0; }
    }
  }
  finish() {
    if (this.n > 0) { this.cur <<= (8 - this.n); this.out.push(this.cur & 255); this.cur = 0; this.n = 0; }
    return new Uint8Array(this.out);
  }
}

class BitReader {
  constructor(bytes) { this.b = bytes; this.p = 0; this.cur = 0; this.n = 0; }
  read(bits) {
    let v = 0;
    for (let i = 0; i < bits; i++) {
      if (this.n === 0) {
        if (this.p >= this.b.length) throw new Error('bitstream truncated');
        this.cur = this.b[this.p++]; this.n = 8;
      }
      v = (v << 1) | ((this.cur >> (this.n - 1)) & 1);
      this.n--;
    }
    return v >>> 0;
  }
}

function encodeColumnar(rows, opts) {
  if (!Array.isArray(rows) || rows.length === 0) throw new Error('columnar: rows must be non-empty array');
  const nameSet = new Set();
  for (const r of rows) {
    if (!r || typeof r !== 'object' || Array.isArray(r)) throw new Error('columnar: row must be object');
    for (const k of Object.keys(r)) nameSet.add(k);
  }
  const names = [...nameSet].sort();
  for (const name of names) {
    let tag = null;
    for (const r of rows) {
      const v = r[name];
      const t = (v === null) ? 'null' : (Array.isArray(v) ? 'array' : typeof v);
      if (tag === null) tag = t;
      else if (tag !== t) throw new Error('columnar: mixed types in column ' + name);
    }
  }
  const plain = !!(opts && opts.plain); // plain=无字典直存（对照模式）
  const w = new ByteWriter();
  w.u8(COL_MAGIC); w.u8(COL_VERSION);
  w.u32le(rows.length);
  w.u32le(names.length);
  for (const name of names) { const nb = utf8Bytes(name); w.varint(nb.length); w.bytes(nb); }
  for (const name of names) {
    if (plain) {
      w.u8(0x01); // mode: plain（值自带类型标签，无需列级类型字节）
      for (const r of rows) _encValue(w, r[name]);
      continue;
    }
    const dict = [], index = new Map();
    const idxs = new Array(rows.length);
    for (let i = 0; i < rows.length; i++) {
      const v = rows[i][name];
      const key = _dictKey(v);
      if (!index.has(key)) { index.set(key, dict.length); dict.push(v); }
      idxs[i] = index.get(key);
    }
    const bw = Math.max(1, Math.ceil(Math.log2(Math.max(2, dict.length))));
    const bwriter = new BitWriter();
    for (const ix of idxs) bwriter.write(ix, bw);
    const packed = bwriter.finish();
    w.u8(0x00); // mode: dictionary
    w.u8(_typeTagOf(dict.length ? dict[0] : rows[0][name]));
    w.varint(dict.length);
    for (const dv of dict) _encValue(w, dv);
    w.u8(bw);
    w.varint(packed.length);
    w.bytes(packed);
  }
  return w.toBytes();
}

function decodeColumnar(bytes) {
  const r = new ByteReader(bytes);
  if (r.u8() !== COL_MAGIC) throw new Error('columnar: bad magic');
  if (r.u8() !== COL_VERSION) throw new Error('columnar: bad version');
  const nRows = r.u32le(), nCols = r.u32le();
  const names = new Array(nCols);
  for (let c = 0; c < nCols; c++) { const nl = r.varint(); names[c] = utf8Decode(r.bytes(nl)); }
  const rows = new Array(nRows);
  for (let i = 0; i < nRows; i++) rows[i] = {};
  for (let c = 0; c < nCols; c++) {
    const mode = r.u8();
    if (mode === 0x01) {
      // plain：逐行直存
      for (let i = 0; i < nRows; i++) {
        const v = _decValue(r);
        rows[i][names[c]] = (v !== null && typeof v === 'object') ? deepClone(v) : v;
      }
      continue;
    }
    if (mode !== 0x00) throw new Error('columnar: bad mode 0x' + mode.toString(16));
    r.u8(); // 值类型标签（信息性，真正类型由值自身标签决定）
    const dictLen = r.varint();
    const dict = new Array(dictLen);
    for (let d = 0; d < dictLen; d++) dict[d] = _decValue(r);
    const bw = r.u8();
    const plen = r.varint();
    const packed = r.bytes(plen);
    const br = new BitReader(packed);
    for (let i = 0; i < nRows; i++) {
      const ix = bw === 0 ? 0 : br.read(bw);
      if (ix >= dictLen) throw new Error('columnar: index out of range');
      const v = dict[ix];
      rows[i][names[c]] = (v !== null && typeof v === 'object') ? deepClone(v) : v;
    }
  }
  return rows;
}

/* ============================================================================
 * 编解码器 4：protobuf 线格式（tag varint + wire type；sint 走 zigzag）
 *   wire 0=varint(zigzag) 1=i64(LE8) 5=i32(LE4) 2=length-delimited
 * ========================================================================== */
function encodeProtobuf(fields) {
  const w = new ByteWriter();
  w.u8(0x50); w.u8(0x01); // magic 'P'
  w.varint(fields.length);
  for (const f of fields) {
    if (!Number.isInteger(f.field) || f.field < 1) throw new Error('protobuf: field number must be positive int');
    w.varint((f.field << 3) | f.wt);
    if (f.wt === 0) { w.varint(zigzag32(f.value | 0)); }
    else if (f.wt === 1) {
      const ab = new ArrayBuffer(8);
      const u8 = new Uint8Array(ab);
      const dv = new DataView(ab);
      dv.setBigInt64(0, BigInt(f.value), true);
      for (let i = 0; i < 8; i++) w.u8(u8[i]);
    } else if (f.wt === 5) { w.u32le(f.value >>> 0); }
    else if (f.wt === 2) {
      const b = (f.value instanceof Uint8Array) ? f.value : utf8Bytes(String(f.value));
      w.varint(b.length); w.bytes(b);
    } else throw new Error('protobuf: unsupported wire type ' + f.wt);
  }
  return w.toBytes();
}

function decodeProtobuf(bytes) {
  const r = new ByteReader(bytes);
  if (r.u8() !== 0x50) throw new Error('protobuf: bad magic');
  if (r.u8() !== 0x01) throw new Error('protobuf: bad version');
  const n = r.varint();
  const fields = new Array(n);
  for (let i = 0; i < n; i++) {
    const tag = r.varint();
    const field = tag >>> 3, wt = tag & 7;
    if (field < 1) throw new Error('protobuf: bad field number');
    if (wt === 0) { fields[i] = { field, wt, value: unzigzag32(r.varint()) }; }
    else if (wt === 1) {
      const ab = new ArrayBuffer(8);
      new Uint8Array(ab).set(r.bytes(8));
      fields[i] = { field, wt, value: Number(new DataView(ab).getBigInt64(0, true)) };
    } else if (wt === 5) { fields[i] = { field, wt, value: r.u32le() | 0 }; }
    else if (wt === 2) { const l = r.varint(); fields[i] = { field, wt, value: r.bytes(l) }; }
    else throw new Error('protobuf: unsupported wire type ' + wt);
  }
  if (!r.eof) throw new Error('protobuf: trailing garbage');
  return fields;
}

/* ============================================================================
 * 分页内存映射（mmap 语义仿真：页表 + LRU 逐出 + 缺页计数 + 脏页写回）
 * ========================================================================== */
class MappedRegion {
  constructor(opts) {
    const backing = opts.backing;
    if (!(backing instanceof Uint8Array)) throw new Error('mmap: backing must be Uint8Array');
    const pageSize = opts.pageSize || 64;
    if (!Number.isInteger(pageSize) || pageSize <= 0) throw new Error('mmap: bad pageSize');
    const length = opts.length;
    const offset = opts.offset || 0;
    if (!Number.isInteger(length) || length < 0) throw new Error('mmap: bad length');
    if (offset + length > backing.length) throw new Error('mmap: region exceeds backing store');
    this.backing = backing;
    this.base = offset;
    this.length = length;
    this.pageSize = pageSize;
    this.maxPages = opts.maxPages || 4;
    if (!Number.isInteger(this.maxPages) || this.maxPages < 1) throw new Error('mmap: bad maxPages');
    this.nPages = Math.ceil(length / pageSize);
    this.pages = new Map();   // pageIndex -> Uint8Array(pageSize)
    this.lru = [];            // front = LRU，end = MRU
    this.dirty = new Set();
    this.faults = 0;
    this.evictions = 0;
    this.writebacks = 0;
  }
  _touch(idx) {
    const at = this.lru.indexOf(idx);
    if (at >= 0) this.lru.splice(at, 1);
    this.lru.push(idx);
  }
  _page(idx) {
    if (idx < 0 || idx >= this.nPages) throw new Error('mmap: page index out of range');
    const hit = this.pages.get(idx);
    if (hit) { this._touch(idx); return hit; }
    this.faults++;
    const start = this.base + idx * this.pageSize;
    const valid = Math.min(this.pageSize, this.length - idx * this.pageSize);
    const data = new Uint8Array(this.pageSize);
    data.set(this.backing.subarray(start, start + valid));
    this.pages.set(idx, data);
    this._touch(idx);
    while (this.pages.size > this.maxPages) this._evictOne();
    return data;
  }
  _evictOne() {
    const victim = this.lru.shift();
    const data = this.pages.get(victim);
    this.pages.delete(victim);
    if (this.dirty.has(victim)) {
      const start = this.base + victim * this.pageSize;
      const valid = Math.min(this.pageSize, this.length - victim * this.pageSize);
      this.backing.set(data.subarray(0, valid), start);
      this.dirty.delete(victim);
      this.writebacks++;
    }
    this.evictions++;
  }
  _check(off, len) {
    if (!Number.isInteger(off) || !Number.isInteger(len) || off < 0 || len < 0 || off + len > this.length) {
      throw new Error('mmap: out of bounds access');
    }
  }
  read(off, len) {
    this._check(off, len);
    const out = new Uint8Array(len);
    let done = 0;
    while (done < len) {
      const a = off + done;
      const idx = Math.floor(a / this.pageSize);
      const page = this._page(idx);
      const within = a - idx * this.pageSize;
      const take = Math.min(len - done, this.pageSize - within);
      out.set(page.subarray(within, within + take), done);
      done += take;
    }
    return out;
  }
  write(off, bytes) {
    this._check(off, bytes.length);
    let done = 0;
    while (done < bytes.length) {
      const a = off + done;
      const idx = Math.floor(a / this.pageSize);
      const page = this._page(idx);
      const within = a - idx * this.pageSize;
      const take = Math.min(bytes.length - done, this.pageSize - within);
      page.set(bytes.subarray(done, done + take), within);
      this.dirty.add(idx);
      done += take;
    }
  }
  flush() {
    for (const idx of [...this.dirty]) {
      const data = this.pages.get(idx);
      const start = this.base + idx * this.pageSize;
      const valid = Math.min(this.pageSize, this.length - idx * this.pageSize);
      this.backing.set(data.subarray(0, valid), start);
      this.writebacks++;
    }
    this.dirty.clear();
  }
}

/* ============================================================================
 * 增量编码（diff：公共前后缀 + 中段 copy/insert；apply 逆操作）
 * ========================================================================== */
function indexOfBytes(hay, needle) {
  if (needle.length === 0) return 0;
  if (needle.length > hay.length) return -1;
  const last = hay.length - needle.length;
  for (let i = 0; i <= last; i++) {
    let ok = true;
    for (let j = 0; j < needle.length; j++) if (hay[i + j] !== needle[j]) { ok = false; break; }
    if (ok) return i;
  }
  return -1;
}

function encodeDelta(base, target) {
  if (!(base instanceof Uint8Array) || !(target instanceof Uint8Array)) throw new Error('delta: Uint8Array required');
  const ops = [];
  const maxP = Math.min(base.length, target.length);
  let p = 0;
  while (p < maxP && base[p] === target[p]) p++;
  let s = 0;
  while (s < maxP - p && base[base.length - 1 - s] === target[target.length - 1 - s]) s++;
  const bMid = base.subarray(p, base.length - s);
  const tMid = target.subarray(p, target.length - s);
  ops.push({ op: 'copy', src: 0, dst: 0, len: p });
  if (tMid.length > 0) {
    const found = indexOfBytes(bMid, tMid);
    if (found >= 0) ops.push({ op: 'copy', src: p + found, dst: p, len: tMid.length });
    else ops.push({ op: 'ins', dst: p, bytes: new Uint8Array(tMid) });
  }
  if (s > 0) ops.push({ op: 'copy', src: base.length - s, dst: p + tMid.length, len: s });
  return ops;
}

function applyDelta(base, ops) {
  if (!(base instanceof Uint8Array)) throw new Error('delta: Uint8Array base required');
  const out = [];
  for (const o of ops) {
    if (o.op === 'copy') {
      if (!Number.isInteger(o.src) || !Number.isInteger(o.len) || o.src < 0 || o.len < 0 || o.src + o.len > base.length) {
        throw new Error('delta: copy op out of base bounds');
      }
      for (let i = 0; i < o.len; i++) out.push(base[o.src + i]);
    } else if (o.op === 'ins') {
      for (let i = 0; i < o.bytes.length; i++) out.push(o.bytes[i]);
    } else throw new Error('delta: unknown op ' + o.op);
  }
  return new Uint8Array(out);
}

function encodeDeltaBytes(ops) {
  const w = new ByteWriter();
  w.u8(0x44); w.u8(0x01); // magic 'D'
  w.varint(ops.length);
  for (const o of ops) {
    if (o.op === 'copy') { w.u8(0x01); w.varint(o.src); w.varint(o.len); }
    else { w.u8(0x02); w.varint(o.bytes.length); w.bytes(o.bytes); }
  }
  return w.toBytes();
}

function decodeDeltaBytes(bytes) {
  const r = new ByteReader(bytes);
  if (r.u8() !== 0x44) throw new Error('delta: bad magic');
  if (r.u8() !== 0x01) throw new Error('delta: bad version');
  const n = r.varint();
  const ops = new Array(n);
  for (let i = 0; i < n; i++) {
    const t = r.u8();
    if (t === 0x01) { ops[i] = { op: 'copy', src: r.varint(), dst: 0, len: r.varint() }; }
    else if (t === 0x02) { const l = r.varint(); ops[i] = { op: 'ins', dst: 0, bytes: r.bytes(l) }; }
    else throw new Error('delta: bad op tag');
  }
  if (!r.eof) throw new Error('delta: trailing garbage');
  return ops;
}

/* ============================================================================
 * 校验持久化（每条记录携带 crc32/fnv1a64/sha256；帧级 crc + 内容级摘要）
 * ========================================================================== */
class ChecksummedStore {
  constructor() { this.entries = new Map(); }
  put(key, bytes) {
    if (!(bytes instanceof Uint8Array)) throw new Error('checksummed: Uint8Array required');
    this.entries.set(key, {
      payload: bytes.slice(),
      crc: crc32(bytes),
      fnv: fnv1a64Hex(bytes),
      sha: sha256BytesHex(bytes)
    });
  }
  get(key) {
    const e = this.entries.get(key);
    if (!e) throw new Error('checksummed: missing key ' + key);
    return e.payload;
  }
  keys() { return [...this.entries.keys()]; }
  verify() {
    const bad = [];
    for (const [k, e] of this.entries) {
      if (crc32(e.payload) !== e.crc || fnv1a64Hex(e.payload) !== e.fnv || sha256BytesHex(e.payload) !== e.sha) bad.push(k);
    }
    return { ok: bad.length === 0, bad };
  }
  serialize() {
    const w = new ByteWriter();
    w.u8(0x4B); w.u8(0x01); // magic 'K'
    w.varint(this.entries.size);
    for (const [k, e] of this.entries) {
      const kb = utf8Bytes(k);
      w.varint(kb.length); w.bytes(kb);
      w.varint(e.payload.length); w.bytes(e.payload);
      w.u32le(e.crc);
      w.bytes(hexToBytes(e.fnv));
      w.bytes(hexToBytes(e.sha));
    }
    return w.toBytes();
  }
  static deserialize(bytes) {
    const r = new ByteReader(bytes);
    if (r.u8() !== 0x4B) throw new Error('checksummed: bad magic');
    if (r.u8() !== 0x01) throw new Error('checksummed: bad version');
    const n = r.varint();
    const store = new ChecksummedStore();
    const corrupted = [];
    for (let i = 0; i < n; i++) {
      let kl, pl, key, payload, crc, fnvHex, shaHex;
      try {
        kl = r.varint(); key = utf8Decode(r.bytes(kl));
        pl = r.varint(); payload = r.bytes(pl);
        crc = r.u32le(); fnvHex = bytesToHex(r.bytes(8)); shaHex = bytesToHex(r.bytes(32));
      } catch (err) { corrupted.push('frame#' + i); break; }
      const bad = crc32(payload) !== crc || fnv1a64Hex(payload) !== fnvHex || sha256BytesHex(payload) !== shaHex;
      if (bad) corrupted.push(key);
      else store.put(key, payload);
    }
    if (!r.eof) throw new Error('checksummed: trailing garbage');
    return { store, corrupted };
  }
}

/* ============================================================================
 * 版本化存储（不可变版本 + 单调 id + 回滚不毁历史）
 * ========================================================================== */
class VersionedStore {
  constructor() { this.versions = []; this.latestIdx = -1; }
  put(value) {
    this.versions.push({ id: this.versions.length + 1, value: deepClone(value) });
    this.latestIdx = this.versions.length - 1;
    return this.versions[this.versions.length - 1].id;
  }
  get(id) {
    const v = this.versions.find((x) => x.id === id);
    if (!v) throw new Error('versioned: no version ' + id);
    return deepClone(v.value);
  }
  latest() { return this.latestIdx < 0 ? null : this.get(this.versions[this.latestIdx].id); }
  latestId() { return this.latestIdx < 0 ? null : this.versions[this.latestIdx].id; }
  history() { return this.versions.map((v) => v.id); }
  rollback(id) {
    const idx = this.versions.findIndex((x) => x.id === id);
    if (idx < 0) throw new Error('versioned: no version ' + id);
    this.latestIdx = idx;
    return id;
  }
}

/* ============================================================================
 * 快照存储（COW：深冻结快照 + 恢复即新版本，历史保全）
 * ========================================================================== */
class SnapshotStore {
  constructor() { this.vs = new VersionedStore(); this.snaps = new Map(); this.snapSeq = 0; }
  put(value) { return this.vs.put(value); }
  get() { return this.vs.latest(); }
  history() { return this.vs.history(); }
  snapshot() {
    const snap = deepFreeze({ id: 'snap-' + (++this.snapSeq), at: this.vs.latestId(), value: deepFreeze(deepClone(this.vs.latest())) });
    this.snaps.set(snap.id, snap);
    return snap;
  }
  restore(snap) {
    if (!this.snaps.has(snap.id)) throw new Error('snapshot: unknown snapshot ' + snap.id);
    return this.vs.put(deepClone(snap.value));
  }
}

/* ============================================================================
 * 分区：哈希分区 / 一致性哈希环 / 范围分区
 * ========================================================================== */
function hashPartition(key, n) {
  if (!Number.isInteger(n) || n < 1) throw new Error('partition: n must be positive integer');
  return Number(fnv1a64(utf8Bytes(String(key))) % BigInt(n));
}

/**
 * 环位置哈希：SHA-256 前 8 字节（大端）→ BigInt。
 * 为何不用 FNV-1a：FNV 对短字符串（如 'n2#37'）高位混合不足——实测四节点各 50 虚节点
 * 时其 64 位哈希按节点聚集在窄带（n2 分到 0 键、n3 分到 410 键），一致性哈希退化为
 * 静态分区。SHA-256 混合充分，与 Ketama 用 MD5 同一思想。
 */
function ringHash64(str) {
  const hex = sha256BytesHex(utf8Bytes(String(str)));
  return BigInt('0x' + hex.slice(0, 16));
}

class ConsistentHashRing {
  constructor(nodes, vnodes) {
    if (!Array.isArray(nodes) || nodes.length === 0) throw new Error('ring: nodes required');
    const vn = vnodes || 100;
    this.vnodes = vn;
    this.ring = [];
    for (const node of nodes) {
      for (let i = 0; i < vn; i++) this.ring.push({ h: ringHash64(node + '#' + i), node });
    }
    this.ring.sort((a, b) => (a.h < b.h ? -1 : (a.h > b.h ? 1 : 0)));
  }
  route(key) {
    const h = ringHash64(key);
    let lo = 0, hi = this.ring.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (this.ring[mid].h < h) lo = mid + 1; else hi = mid;
    }
    return this.ring[lo % this.ring.length].node;
  }
  movedFraction(keys, otherRing) {
    let moved = 0;
    for (const k of keys) if (this.route(k) !== otherRing.route(k)) moved++;
    return moved / keys.length;
  }
}

// 范围键比较：若两端均可解析为数值（整数/浮点），按数值比较；否则按字符串字典序回退。
// 修复纯字符串比较的缺陷：'5' >= '10' 在字典序下为 true，会将数值 5 误路由到最后一个分片。
function _rangeKeyIsNumeric(s) { return s !== '' && s !== null && s !== undefined && !Number.isNaN(Number(s)); }
function _cmpRangeKey(a, b) {
  const sa = String(a), sb = String(b);
  if (_rangeKeyIsNumeric(sa) && _rangeKeyIsNumeric(sb)) {
    const na = Number(sa), nb = Number(sb);
    return na < nb ? -1 : (na > nb ? 1 : 0);
  }
  return sa < sb ? -1 : (sa > sb ? 1 : 0);
}
class RangePartitioner {
  constructor(boundaries) {
    if (!Array.isArray(boundaries)) throw new Error('range: boundaries array required');
    this.boundaries = boundaries.slice().sort(_cmpRangeKey);
  }
  route(key) {
    let idx = 0;
    while (idx < this.boundaries.length && _cmpRangeKey(String(key), this.boundaries[idx]) >= 0) idx++;
    return idx;
  }
}

/* ============================================================================
 * 对象存储（内容寻址去重 + 不可变版本 + 元数据 + 墓碑删除）
 * ========================================================================== */
class ObjectStore {
  constructor() { this.bodies = new Map(); this.objs = new Map(); this.clock = 0; }
  static etagOf(bytes) { return sha256BytesHex(bytes); }
  put(bucket, key, bytes, meta) {
    if (!(bytes instanceof Uint8Array)) throw new Error('objectstore: Uint8Array required');
    const etag = ObjectStore.etagOf(bytes);
    if (!this.bodies.has(etag)) this.bodies.set(etag, bytes.slice());
    const bk = bucket + '/' + key;
    let rec = this.objs.get(bk);
    if (!rec) { rec = { versions: [], tombstone: false }; this.objs.set(bk, rec); }
    rec.tombstone = false;
    const version = rec.versions.length + 1;
    rec.versions.push({ version, etag, size: bytes.length, meta: deepClone(meta || {}), ts: ++this.clock });
    return { etag, version, size: bytes.length };
  }
  get(bucket, key, version) {
    const rec = this.objs.get(bucket + '/' + key);
    if (!rec || rec.tombstone || rec.versions.length === 0) throw new Error('objectstore: not found ' + bucket + '/' + key);
    const v = version ? rec.versions.find((x) => x.version === version) : rec.versions[rec.versions.length - 1];
    if (!v) throw new Error('objectstore: version ' + version + ' not found');
    const body = this.bodies.get(v.etag);
    if (!body) throw new Error('objectstore: body missing (store corruption)');
    return { body: body.slice(), meta: deepClone(v.meta), etag: v.etag, version: v.version };
  }
  head(bucket, key) {
    const rec = this.objs.get(bucket + '/' + key);
    if (!rec || rec.tombstone || rec.versions.length === 0) throw new Error('objectstore: not found ' + bucket + '/' + key);
    const v = rec.versions[rec.versions.length - 1];
    return { etag: v.etag, size: v.size, meta: deepClone(v.meta), version: v.version };
  }
  delete(bucket, key) {
    const rec = this.objs.get(bucket + '/' + key);
    if (!rec) throw new Error('objectstore: not found ' + bucket + '/' + key);
    rec.tombstone = true;
  }
}

/* ============================================================================
 * 缓存：LRU（本地，Map 插入序即 LRU 序）与 TTL（远端，可注入时钟 + SWR）
 * ========================================================================== */
class LRUCache {
  constructor(capacity) {
    if (!Number.isInteger(capacity) || capacity < 1) throw new Error('lru: capacity must be positive integer');
    this.cap = capacity;
    this.map = new Map();
    this.hits = 0; this.misses = 0; this.evictions = 0;
  }
  get(k) {
    if (this.map.has(k)) {
      const v = this.map.get(k);
      this.map.delete(k); this.map.set(k, v); // 移至 MRU 端
      this.hits++;
      return v;
    }
    this.misses++;
    return undefined;
  }
  put(k, v) {
    if (this.map.has(k)) this.map.delete(k);
    this.map.set(k, v);
    let evicted = null;
    if (this.map.size > this.cap) {
      evicted = this.map.keys().next().value; // LRU 端
      this.map.delete(evicted);
      this.evictions++;
    }
    return evicted;
  }
  keys() { return [...this.map.keys()]; } // front = LRU，end = MRU
  get size() { return this.map.size; }
  get gets() { return this.hits + this.misses; }
}

class TTLCache {
  constructor(opts) {
    this.now = (opts && typeof opts.now === 'function') ? opts.now : () => 0;
    this.map = new Map();
    this.hits = 0; this.misses = 0; this.staleHits = 0; this.refreshes = 0;
  }
  set(k, v, ttl) {
    if (!Number.isFinite(ttl) || ttl < 0) throw new Error('ttl: ttl must be finite non-negative');
    this.map.set(k, { v, expireAt: this.now() + ttl });
  }
  get(k) {
    const e = this.map.get(k);
    if (!e) { this.misses++; return { hit: false }; }
    if (this.now() >= e.expireAt) { this.map.delete(k); this.misses++; return { hit: false, expired: true }; }
    this.hits++;
    return { hit: true, value: e.v };
  }
  /** stale-while-revalidate：过期仍返回旧值并置 revalidate 标记 */
  getStale(k) {
    const e = this.map.get(k);
    if (!e) { this.misses++; return { hit: false }; }
    if (this.now() < e.expireAt) { this.hits++; return { hit: true, value: e.v, stale: false }; }
    this.staleHits++;
    return { hit: true, value: e.v, stale: true, revalidate: true };
  }
  refresh(k, v, ttl) { this.set(k, v, ttl); this.refreshes++; return v; }
  has(k) { const e = this.map.get(k); return !!e && this.now() < e.expireAt; }
}

/* ============================================================================
 * 写前日志（WAL）：begin/write/remove/commit/abort + 帧化序列化 + 两阶段恢复
 * ========================================================================== */
class WriteAheadLog {
  constructor() { this.records = []; this.lsn = 0; this.txnSeq = 0; this.txns = new Map(); }
  begin() {
    const id = ++this.txnSeq;
    this.txns.set(id, { state: 'active' });
    return id;
  }
  _active(id) {
    const t = this.txns.get(id);
    if (!t || t.state !== 'active') throw new Error('wal: txn ' + id + ' not active');
  }
  write(txn, key, value) { this._active(txn); this._push({ t: 'W', txn, key, value }); }
  remove(txn, key) { this._active(txn); this._push({ t: 'D', txn, key }); }
  commit(txn) { this._active(txn); this._push({ t: 'C', txn }); this.txns.get(txn).state = 'committed'; }
  abort(txn) { this._active(txn); this._push({ t: 'A', txn }); this.txns.get(txn).state = 'aborted'; }
  _push(rec) { rec.lsn = ++this.lsn; this.records.push(rec); }
  serialize() {
    const w = new ByteWriter();
    w.u32le(this.records.length);
    for (const r of this.records) {
      const body = encodeBinary(r);
      w.u32le(body.length); w.u32le(crc32(body)); w.bytes(body);
    }
    return w.toBytes();
  }
}

/** 解析帧：遇首个损坏/截断帧即停（崩溃语义：其后记录不可信） */
function parseWalFrames(bytes) {
  const r = new ByteReader(bytes);
  try { r.u32le(); } catch (err) { return []; } // 计数仅作提示；连头部都不完整则无记录
  const out = [];
  while (!r.eof) {
    let len, crc, body;
    try {
      len = r.u32le(); crc = r.u32le(); body = r.bytes(len);
      if (crc32(body) !== crc) break;
      out.push(decodeBinary(body));
    } catch (err) { break; }
  }
  return out;
}

/** ARIES 风格两阶段恢复：先集已提交事务，再按 lsn 序 redo（未提交/中止不可见） */
function recoverWal(records) {
  const committed = new Set();
  for (const rec of records) if (rec.t === 'C') committed.add(rec.txn);
  const state = new Map();
  for (const rec of records) {
    if (!committed.has(rec.txn)) continue;
    if (rec.t === 'W') state.set(rec.key, rec.value);
    else if (rec.t === 'D') state.delete(rec.key);
  }
  return state;
}

/* ============================================================================
 * 灾难恢复：双校验备份（crc32 + sha256）+ restore + PITR
 * ========================================================================== */
function backupState(stateMap) {
  const entries = [...stateMap.entries()];
  const body = encodeBinary({ state: entries });
  const w = new ByteWriter();
  w.u8(0x42); w.u8(0x01); // magic 'B'
  w.u32le(body.length);
  w.u32le(crc32(body));
  w.bytes(hexToBytes(sha256BytesHex(body)));
  w.bytes(body);
  return w.toBytes();
}

function restoreBackup(bytes) {
  const r = new ByteReader(bytes);
  if (r.u8() !== 0x42) throw new Error('backup: bad magic');
  if (r.u8() !== 0x01) throw new Error('backup: bad version');
  const len = r.u32le(), crc = r.u32le();
  const sha = r.bytes(32);
  const body = r.bytes(len);
  if (crc32(body) !== crc) throw new Error('backup: crc32 mismatch');
  if (sha256BytesHex(body) !== bytesToHex(sha)) throw new Error('backup: sha256 mismatch');
  const obj = decodeBinary(body);
  return new Map(obj.state);
}

/** 时间点恢复：备份态 + 重放 lsn≤targetLsn 的已提交事务 */
function pitr(backupBytes, walRecords, targetLsn) {
  const state = restoreBackup(backupBytes);
  const committed = new Set();
  for (const rec of walRecords) if (rec.t === 'C' && rec.lsn <= targetLsn) committed.add(rec.txn);
  for (const rec of walRecords) {
    if (rec.lsn > targetLsn) break;
    if (!committed.has(rec.txn)) continue;
    if (rec.t === 'W') state.set(rec.key, rec.value);
    else if (rec.t === 'D') state.delete(rec.key);
  }
  return state;
}

/* ============================================================================
 * 随机对象图生成器（有限数，JSON 兼容；3 种子驱动各不变量）
 * ========================================================================== */
function randStr(rng, len) {
  const alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-';
  let s = '';
  for (let i = 0; i < len; i++) s += alphabet[Math.floor(rng() * alphabet.length)];
  return s;
}

function randomValue(rng, depth) {
  const r = rng();
  if (depth <= 0 || r < 0.35) {
    const k = Math.floor(rng() * 5);
    if (k === 0) return Math.floor(rng() * 2000) - 1000;
    if (k === 1) return rng() * 1000 - 500;
    if (k === 2) return randStr(rng, Math.floor(rng() * 12));
    if (k === 3) return rng() < 0.5;
    return null;
  }
  if (r < 0.7) {
    const n = Math.floor(rng() * 5);
    const a = [];
    for (let i = 0; i < n; i++) a.push(randomValue(rng, depth - 1));
    return a;
  }
  const n = Math.floor(rng() * 5);
  const o = {};
  for (let i = 0; i < n; i++) o['k' + Math.floor(rng() * 100)] = randomValue(rng, depth - 1);
  return o;
}

/* ============================================================================
 * 自检
 * ========================================================================== */
function caseOf(id, name, ok, actual, detail, tol) {
  const numActual = (typeof actual === 'number' && isFinite(actual)) ? actual : (ok ? 0 : 1);
  return {
    id, name,
    status: ok ? 'pass' : 'fail',
    passed: !!ok,
    actual: numActual,
    expected: 0,
    tol: (typeof tol === 'number') ? tol : TOL.EXACT,
    detail
  };
}

function buildReport(module, version, layer, cases, invariants, seed) {
  const started = Date.now();
  const passed = cases.filter((c) => c.status === 'pass').length;
  const failed = cases.length - passed;
  let hashSrc = '';
  for (const c of cases) {
    hashSrc += c.id + '|' + c.status + '|' + (typeof c.actual === 'number' ? String(Math.round(c.actual * 1e12)) : String(c.actual)) + '\n';
  }
  return {
    schema: 'forge-selfcheck/1.0',
    module, version, layer,
    passed, failed, skipped: 0,
    duration_ms: Date.now() - started,
    tol: TOL,
    invariants,
    cases,
    env: {
      platform: (typeof process !== 'undefined' && process.platform) || 'browser',
      arch: (typeof process !== 'undefined' && process.arch) || 'unknown',
      node: (typeof process !== 'undefined' && process.version) || 'browser',
      seed
    },
    hash: 'sha256:' + sha256Hex(hashSrc)
  };
}

function selfCheck() {
  const cases = [], invariants = [];
  const push = (id, name, ok, actual, detail, tol) => {
    cases.push(caseOf(id, name, ok, actual, detail, tol));
    invariants.push({ id, name, ok, expr: name, tol: (typeof tol === 'number') ? tol : TOL.EXACT, worst: ok ? 0 : 1, detail });
  };

  /* ---------- INV-SER-001：四编解码器 roundtrip + 规范确定性 + 紧凑性 + 损坏必检出 ---------- */
  {
    let allRoundtrip = true, canonicalOk = true, corruptDetected = true, compactOk = true;
    let worstRt = 0;
    const seeds = [0x5E120001, 0x5E120002, 0x5E120003];
    for (const seed of seeds) {
      const rng = makeRng(seed);
      for (let t = 0; t < 6; t++) {
        const v = randomValue(rng, 3);
        const j = decodeJson(encodeJson(v));
        const b = decodeBinary(encodeBinary(v));
        if (!deepEqual(v, j) || !deepEqual(v, b)) { allRoundtrip = false; worstRt++; }
      }
      // 列式：均匀数值行（同构）
      const rows = [];
      for (let i = 0; i < 40; i++) rows.push({ a: Math.floor(rng() * 7), b: Math.floor(rng() * 5), c: rng() * 10 });
      if (!deepEqual(rows, decodeColumnar(encodeColumnar(rows)))) { allRoundtrip = false; worstRt++; }
      // protobuf：混合字段
      const fields = [
        { field: 1, wt: 0, value: -12345 },
        { field: 2, wt: 2, value: 'forge-序列化' },
        { field: 3, wt: 1, value: -(2 ** 40) },
        { field: 4, wt: 5, value: 987654321 },
        { field: 5, wt: 2, value: new Uint8Array([0, 1, 254, 255]) }
      ];
      const pf = decodeProtobuf(encodeProtobuf(fields));
      if (pf.length !== 5 || pf[0].value !== -12345 || utf8Decode(pf[1].value) !== 'forge-序列化' || pf[2].value !== -(2 ** 40)
        || pf[3].value !== 987654321 || !bytesEqual(pf[4].value, new Uint8Array([0, 1, 254, 255]))) {
        allRoundtrip = false; worstRt++;
      }
    }
    // 规范确定性：键序无关（JSON 与二进制均字节级相等）
    const o1 = { b: 1, a: { d: 4, c: 3 }, z: [1, 2] };
    const o2 = { z: [1, 2], a: { c: 3, d: 4 }, b: 1 };
    if (encodeJson(o1) !== encodeJson(o2)) canonicalOk = false;
    if (!bytesEqual(encodeBinary(o1), encodeBinary(o2))) canonicalOk = false;
    // 紧凑性（两个互补负载，各证一边）：
    // (a) float64 负载：binary ≤0.75·json（JSON 文本 ~18 字符 vs 二进制 9 字节）
    // (b) 小整数负载（field 1-15）：protobuf varint ≤0.6·binary（2 字节/字段 vs 5 字节/值）
    const rngC = makeRng(0x5E120004);
    const floats = [];
    for (let i = 0; i < 200; i++) floats.push(rngC() * 1e6 - 5e5);
    const jSize = utf8Bytes(encodeJson(floats)).length;
    const bSize = encodeBinary(floats).length;
    const smalls = [];
    for (let i = 0; i < 200; i++) smalls.push(Math.floor(rngC() * 128));
    const bSizeS = encodeBinary(smalls).length;
    const pSize = encodeProtobuf(smalls.map((n, i) => ({ field: (i % 15) + 1, wt: 0, value: n }))).length;
    if (!(bSize <= 0.75 * jSize)) compactOk = false;
    if (!(pSize <= 0.6 * bSizeS)) compactOk = false;
    // 损坏必检出：魔数翻转 / 截断 / 尾随垃圾 / 超长 varint
    try { const bad = encodeBinary(smalls); bad[0] = 0x58; decodeBinary(bad); corruptDetected = false; } catch (e) { /* 预期抛错 */ }
    try { decodeBinary(encodeBinary(floats).subarray(0, 5)); corruptDetected = false; } catch (e) { /* 预期抛错 */ }
    try { const bad = encodeBinary(floats); decodeBinary(new Uint8Array([...bad, 0x00, 0x01])); corruptDetected = false; } catch (e) { /* 预期抛错 */ }
    try {
      const w = new ByteWriter(); w.u8(0x50); w.u8(0x01); w.varint(1); w.varint(8);
      for (let i = 0; i < 8; i++) w.u8(0x80 | (i === 0 ? 1 : 0)); // 8 字节超长 varint
      decodeProtobuf(w.toBytes()); corruptDetected = false;
    } catch (e) { /* 预期抛错 */ }
    try {
      const r = new ByteReader(new Uint8Array([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x7F]));
      r.varint(); corruptDetected = false;
    } catch (e) { /* 预期抛错 */ }
    const ok = allRoundtrip && canonicalOk && corruptDetected && compactOk;
    push('INV-SER-001', '四编解码器 roundtrip EXACT + 规范键序确定 + 紧凑性(float64: binary≤0.75json; 小整数: protobuf≤0.6binary) + 损坏必检出', ok,
      worstRt === 0 && ok ? 0 : worstRt + (ok ? 0 : 1),
      'rt=' + (allRoundtrip ? 'EXACT' : 'FAIL') + ' canonical=' + canonicalOk + ' float json=' + jSize + ' bin=' + bSize + ' | small bin=' + bSizeS + ' pb=' + pSize + ' corruptCaught=' + corruptDetected + ' compact=' + compactOk, TOL.EXACT);
  }

  /* ---------- INV-SER-002：列式无免费午餐（字典 vs 直存 vs 行式） ---------- */
  {
    const rng = makeRng(0x5E220001);
    // 低基数：每列 4 个 distinct 值
    const lowRows = [];
    for (let i = 0; i < 300; i++) lowRows.push({ g: Math.floor(rng() * 4), h: Math.floor(rng() * 4) });
    const lowDict = encodeColumnar(lowRows).length;
    const lowPlain = encodeColumnar(lowRows, { plain: true }).length;
    const lowRow = encodeBinary(lowRows).length;
    // 高基数：每列全 distinct int
    const highRows = [];
    for (let i = 0; i < 300; i++) highRows.push({ g: i, h: i * 7 + 3 });
    const highDict = encodeColumnar(highRows).length;
    const highPlain = encodeColumnar(highRows, { plain: true }).length;
    const highRow = encodeBinary(highRows).length;
    // roundtrip：两种模式 + 行式基准均 EXACT
    const rtDictLow = deepEqual(lowRows, decodeColumnar(encodeColumnar(lowRows)));
    const rtPlainLow = deepEqual(lowRows, decodeColumnar(encodeColumnar(lowRows, { plain: true })));
    const rtDictHigh = deepEqual(highRows, decodeColumnar(encodeColumnar(highRows)));
    const rtPlainHigh = deepEqual(highRows, decodeColumnar(encodeColumnar(highRows, { plain: true })));
    // 确定性：同表同字节（两模式）
    const detDict = bytesEqual(encodeColumnar(lowRows), encodeColumnar(lowRows.map((r) => ({ g: r.g, h: r.h }))));
    const detPlain = bytesEqual(encodeColumnar(highRows, { plain: true }), encodeColumnar(highRows, { plain: true }));
    // 无免费午餐：低基数字典致胜（vs 直存与行式），高基数字典致亏（vs 直存——纯索引开销）
    const lowWins = lowDict <= 0.5 * lowPlain && lowDict <= 0.7 * lowRow;
    const highLoses = highDict >= 1.15 * highPlain;
    const ok = lowWins && highLoses && rtDictLow && rtPlainLow && rtDictHigh && rtPlainHigh && detDict && detPlain;
    push('INV-SER-002', '列式无免费午餐：低基数(4 distinct) 字典 ≤0.5·直存 且 ≤0.7·行式；高基数(全 distinct) 字典 ≥1.15·直存（索引纯开销）；四路 roundtrip EXACT + 两模式确定', ok,
      ok ? 0 : Math.max(lowDict / (0.5 * lowPlain), highPlain / (1.15 * highDict) > 1 ? 1 : highPlain / (1.15 * highDict)),
      'low dict=' + lowDict + ' plain=' + lowPlain + ' row=' + lowRow + ' (d/p=' + (lowDict / lowPlain).toFixed(3) + ' d/r=' + (lowDict / lowRow).toFixed(3) + ') | high dict=' + highDict + ' plain=' + highPlain + ' row=' + highRow + ' (d/p=' + (highDict / highPlain).toFixed(3) + ') rt=' + rtDictLow + '/' + rtPlainLow + '/' + rtDictHigh + '/' + rtPlainHigh + ' det=' + detDict + '/' + detPlain, TOL.EXACT);
  }

  /* ---------- INV-SER-003：内存映射分页语义 ---------- */
  {
    const backing = new Uint8Array(256);
    for (let i = 0; i < 256; i++) backing[i] = i & 255;
    const reg = new MappedRegion({ backing, offset: 0, length: 256, pageSize: 32, maxPages: 2 });
    // 访问序列：p0 p1 p2 p1 p0 p3（页大小 32，共 8 页）
    const touches = [0, 32, 64, 32, 0, 96];
    for (const off of touches) reg.read(off, 4);
    // LRU 推演：p0,p1 → p2 缺页逐出 p0 → p1 命中(MRU) → p0 缺页逐出 p2 → p3 缺页逐出 p1
    // faults=5（p0 被逐出后再次触页即重新缺页），evictions=3，终态 LRU=[0,3]
    const lruOk = reg.evictions === 3 && reg.faults === 5 && reg.lru.join(',') === '0,3';
    // 写回一致性：写页 5 → flush 前 backing 不含 → flush 后含
    const before = backing[160];
    reg.write(160, new Uint8Array([0xAB, 0xCD, 0xEF]));
    const preFlush = backing[160] === before;
    reg.flush();
    const postFlush = backing[160] === 0xAB && backing[161] === 0xCD && backing[162] === 0xEF
      && bytesEqual(reg.read(160, 3), new Uint8Array([0xAB, 0xCD, 0xEF]));
    // 缺页计数口径：全新区域顺序触 4 个不同页
    const backing2 = new Uint8Array(256);
    const reg2 = new MappedRegion({ backing: backing2, offset: 0, length: 256, pageSize: 32, maxPages: 8 });
    reg2.read(0, 4); reg2.read(32, 4); reg2.read(0, 4); reg2.read(64, 4); reg2.read(32, 4);
    const faultOk = reg2.faults === 3 && reg2.evictions === 0;
    // OOB 证伪
    let oobRead = false, oobWrite = false;
    try { reg.read(250, 10); } catch (e) { oobRead = true; }
    try { reg.write(254, new Uint8Array([1, 2, 3, 4])); } catch (e) { oobWrite = true; }
    const ok = lruOk && preFlush && postFlush && faultOk && oobRead && oobWrite;
    push('INV-SER-003', '内存映射：LRU 逐出序 EXACT + 缺页计数==首触页数 EXACT + 脏页写回前后一致 + OOB 抛错', ok,
      ok ? 0 : reg.evictions + reg.faults,
      'evictions=' + reg.evictions + ' faults=' + reg.faults + ' lru=[' + reg.lru.join(',') + '] preFlushClean=' + preFlush + ' postFlush=' + postFlush + ' faultOk=' + faultOk + ' oob=' + oobRead + '/' + oobWrite, TOL.EXACT);
  }

  /* ---------- INV-SER-004：增量 diff-apply + 快照 COW 隔离 ---------- */
  {
    let rtOk = true, detOk = true, falsOk = true;
    for (const seed of [0x5E320001, 0x5E320002, 0x5E320003]) {
      const rng = makeRng(seed);
      const base = new Uint8Array(200);
      for (let i = 0; i < 200; i++) base[i] = Math.floor(rng() * 256);
      // 变异 1：删中段 [50,80)
      const t1 = new Uint8Array([...base.subarray(0, 50), ...base.subarray(80)]);
      // 变异 2：中段替换为随机 40 字节
      const ins = new Uint8Array(40);
      for (let i = 0; i < 40; i++) ins[i] = Math.floor(rng() * 256);
      const t2 = new Uint8Array([...base.subarray(0, 50), ...ins, ...base.subarray(50)]);
      // 变异 3：换尾段
      const t3 = new Uint8Array([...base.subarray(0, 180)]);
      for (let i = 0; i < 20; i++) t3[180 + i] = Math.floor(rng() * 256);
      for (const target of [t1, t2, t3]) {
        const ops = encodeDelta(base, target);
        const got = applyDelta(base, ops);
        if (!bytesEqual(got, target)) rtOk = false;
        // 序列化 roundtrip（dst 字段不进线格式，仅比对 op/src/len/bytes）
        const rt = decodeDeltaBytes(encodeDeltaBytes(ops));
        if (rt.length !== ops.length) detOk = false;
        else for (let i = 0; i < rt.length; i++) {
          if (rt[i].op !== ops[i].op) detOk = false;
          if (ops[i].op === 'copy' && (rt[i].src !== ops[i].src || rt[i].len !== ops[i].len)) detOk = false;
          if (ops[i].op === 'ins' && !bytesEqual(rt[i].bytes, ops[i].bytes)) detOk = false;
        }
      }
      // 同输入确定性
      const opsA = encodeDelta(base, t2);
      const opsB = encodeDelta(base, t2);
      if (JSON.stringify(opsA) !== JSON.stringify(opsB)) detOk = false;
      // 空变异：仅单 copy 且 ≤16 字节
      const noop = encodeDelta(base, base);
      if (noop.length !== 1 || noop[0].op !== 'copy' || noop[0].len !== 200 || encodeDeltaBytes(noop).length > 16) falsOk = false;
      // 证伪一：插入变异后输出变长（240≠200），作用于异 base 内容必异
      const wrong = applyDelta(new Uint8Array(200), encodeDelta(base, t2));
      if (bytesEqual(wrong, t2)) falsOk = false;
      // 证伪二：delta 的 copy 越界（异长度 base）必抛错——确定性
      let threw = false;
      try { applyDelta(new Uint8Array(10), encodeDelta(base, t2)); } catch (e) { threw = true; }
      if (!threw) falsOk = false;
    }
    // 快照 COW 隔离
    const store = new SnapshotStore();
    store.put({ n: 1, items: [1, 2, 3] });
    const snap = store.snapshot();
    const snapValueBefore = JSON.stringify(snap.value);
    store.put({ n: 2, items: [9] });
    store.put({ n: 3, items: [] });
    const isolated = JSON.stringify(snap.value) === snapValueBefore && deepEqual(snap.value, { n: 1, items: [1, 2, 3] });
    const restoredId = store.restore(snap);
    const restoredOk = deepEqual(store.get(), { n: 1, items: [1, 2, 3] }) && restoredId === 4;
    const historyOk = store.history().length === 4 && store.history()[0] === 1;
    const frozen = Object.isFrozen(snap) && Object.isFrozen(snap.value) && Object.isFrozen(snap.value.items);
    const ok = rtOk && detOk && falsOk && isolated && restoredOk && historyOk && frozen;
    push('INV-SER-004', '增量：apply∘encode EXACT(3种子删/插/换) + delta 字节确定 + 空变异仅单copy；快照：深冻结隔离 + 恢复==快照 + 历史保全', ok,
      ok ? 0 : 1,
      'rt=' + rtOk + ' det=' + detOk + ' noop=' + falsOk + ' isolated=' + isolated + ' restored=' + restoredOk + ' history=' + store.history().join(',') + ' frozen=' + frozen, TOL.EXACT);
  }

  /* ---------- INV-SER-005：版本化 + WAL 原子性/持久性 + 灾恢 PITR ---------- */
  {
    // (a) 版本化
    const vs = new VersionedStore();
    const ids = [vs.put({ v: 1 }), vs.put({ v: 2 }), vs.put({ v: 3 })];
    const monoOk = ids[0] === 1 && ids[1] === 2 && ids[2] === 3;
    const immutOk = deepEqual(vs.get(1), { v: 1 }) && deepEqual(vs.get(2), { v: 2 });
    vs.rollback(2);
    const rollOk = deepEqual(vs.latest(), { v: 2 }) && vs.history().length === 3 && vs.latestId() === 2;

    // (b) WAL 脚本事务：t1/t2 交错；t3 未提交；t4 中止；t1 内同键覆写
    const wal = new WriteAheadLog();
    const t1 = wal.begin(), t2 = wal.begin();
    wal.write(t1, 'a', 1);
    wal.write(t2, 'b', 2);
    wal.commit(t2);
    wal.write(t1, 'a', 11);
    wal.write(t1, 'c', 3);
    wal.commit(t1);
    const t3 = wal.begin();
    wal.write(t3, 'd', 4);
    const t4 = wal.begin();
    wal.write(t4, 'e', 5);
    wal.abort(t4);
    const state = recoverWal(wal.records);
    const expected = new Map([['a', 11], ['b', 2], ['c', 3]]);
    const walOk = mapEqual(state, expected);

    // (c) 全前缀崩溃模拟：对每个前缀 p，恢复态 == p 前最后提交态
    const bytes = wal.serialize();
    const allRecords = parseWalFrames(bytes);
    const parseOk = deepEqual(allRecords, wal.records);
    // 每个提交点后的期望态
    const expectedAtCommit = [];
    {
      const committed = new Set();
      const st = new Map();
      for (const rec of allRecords) {
        if (rec.t === 'C') {
          committed.add(rec.txn);
          // 该事务按 lsn 序重放到此
          for (const r2 of allRecords) {
            if (r2.lsn > rec.lsn) break;
            if (r2.txn !== rec.txn) continue;
            if (r2.t === 'W') st.set(r2.key, r2.value); else if (r2.t === 'D') st.delete(r2.key);
          }
          expectedAtCommit.push({ afterLsn: rec.lsn, state: new Map(st) });
        } else if (rec.t === 'A') { /* 中止不入态 */ }
      }
    }
    let crashOk = parseOk;
    for (let p = 0; p <= allRecords.length; p++) {
      const recs = allRecords.slice(0, p);
      const got = recoverWal(recs);
      // p 前最后提交态
      let exp = new Map();
      for (const ec of expectedAtCommit) { if (ec.afterLsn <= (p > 0 ? allRecords[p - 1].lsn : 0)) exp = ec.state; }
      if (!mapEqual(got, exp)) { crashOk = false; break; }
    }
    // 帧级截断（字节级崩溃）：每个字节前缀均应恢复一致或安全丢弃
    let byteCrashOk = true;
    for (let cut = 0; cut <= bytes.length; cut += 7) {
      const recs = parseWalFrames(bytes.subarray(0, cut));
      const got = recoverWal(recs);
      const lastLsn = recs.length ? recs[recs.length - 1].lsn : 0;
      let exp = new Map();
      for (const ec of expectedAtCommit) { if (ec.afterLsn <= lastLsn) exp = ec.state; }
      if (!mapEqual(got, exp)) { byteCrashOk = false; break; }
    }

    // (d) 证伪：抽去提交记录 → 写入不可见
    const wal2 = new WriteAheadLog();
    const x1 = wal2.begin();
    wal2.write(x1, 'k', 'v');
    wal2.commit(x1);
    const noCommit = wal2.records.filter((r) => r.t !== 'C');
    const falsifyOk = recoverWal(noCommit).size === 0;

    // (e) 灾恢：backup/restore EXACT + 损坏必抛 + PITR
    const backup = backupState(expected);
    const restored = restoreBackup(backup);
    const drOk = mapEqual(restored, expected);
    const badBackup = backup.slice();
    badBackup[badBackup.length - 5] ^= 0x01;
    let corruptThrew = false;
    try { restoreBackup(badBackup); } catch (e) { corruptThrew = true; }
    // PITR：备份态 + t2 的 b（已提交）之后追加新提交事务
    const wal3 = new WriteAheadLog();
    const y1 = wal3.begin();
    wal3.write(y1, 'f', 6);
    wal3.commit(y1);
    const y2 = wal3.begin();
    wal3.write(y2, 'g', 7);
    wal3.commit(y2);
    const y3 = wal3.begin();
    wal3.write(y3, 'h', 8);
    // y3 未提交
    const pitrOk = mapEqual(pitr(backup, wal3.records, 0), expected)
      && mapEqual(pitr(backup, wal3.records, 2), new Map([...expected, ['f', 6]]))
      && mapEqual(pitr(backup, wal3.records, 4), new Map([...expected, ['f', 6], ['g', 7]]))
      && mapEqual(pitr(backup, wal3.records, 6), new Map([...expected, ['f', 6], ['g', 7]])); // y3 未提交不可见
    const ok = monoOk && immutOk && rollOk && walOk && crashOk && byteCrashOk && falsifyOk && drOk && corruptThrew && pitrOk;
    push('INV-SER-005', '版本单调+旧版不可变+回滚保史；WAL 交错事务恢复 EXACT；全前缀(记录级+字节级)崩溃恢复 EXACT；抽提交证伪；灾恢 backup/restore/corrupt/PITR EXACT', ok,
      ok ? 0 : 1,
      'mono=' + monoOk + ' immut=' + immutOk + ' roll=' + rollOk + ' wal=' + walOk + ' crash=' + crashOk + ' byteCrash=' + byteCrashOk + ' falsify=' + falsifyOk + ' dr=' + drOk + ' threw=' + corruptThrew + ' pitr=' + pitrOk + ' nRec=' + allRecords.length, TOL.EXACT);
  }

  /* ---------- INV-SER-006：校验和家族（已知向量 + 雪崩 + 定位 + 选择性污染） ---------- */
  {
    // (a) 已知向量
    const v1 = crc32(utf8Bytes('123456789')) === 0xCBF43926;
    const v2 = crc32(new Uint8Array(0)) === 0;
    const v3 = fnv1a64Hex(new Uint8Array(0)) === 'cbf29ce484222325';
    const v4 = fnv1a64Hex(utf8Bytes('a')) === 'af63dc4c8601ec8c';
    const v5 = sha256Hex('abc') === 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad';
    const v6 = sha256Hex('') === 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855';
    const vectorsOk = v1 && v2 && v3 && v4 && v5 && v6;
    // (b) 雪崩：单比特翻转
    let crcFracSum = 0, fnvFracSum = 0, trials = 0, avalancheOk = true;
    for (const seed of [0x5E420001, 0x5E420002, 0x5E420003]) {
      const rng = makeRng(seed);
      const data = new Uint8Array(1024);
      for (let i = 0; i < 1024; i++) data[i] = Math.floor(rng() * 256);
      const c0 = crc32(data), f0 = fnv1a64(data);
      for (let t = 0; t < 20; t++) {
        const bit = Math.floor(rng() * (1024 * 8));
        const d2 = data.slice();
        d2[bit >> 3] ^= 1 << (bit & 7);
        const c1 = crc32(d2), f1 = fnv1a64(d2);
        let cd = 0;
        for (let b = 0; b < 32; b++) if (((c0 ^ c1) >>> b) & 1) cd++;
        let fd = 0;
        for (let b = 0; b < 64; b++) if (((f0 ^ f1) >> BigInt(b)) & 1n) fd++;
        crcFracSum += cd / 32; fnvFracSum += fd / 64; trials++;
      }
    }
    const crcMean = crcFracSum / trials, fnvMean = fnvFracSum / trials;
    if (!(crcMean >= 0.3 && crcMean <= 0.75)) avalancheOk = false;
    if (!(fnvMean >= 0.3 && fnvMean <= 0.75)) avalancheOk = false;
    // (c) 无碰撞：1000 随机输入
    const rng2 = makeRng(0x5E420004);
    const seenF = new Set(), seenC = new Set(), seenS = new Set();
    let collide = false;
    for (let i = 0; i < 1000; i++) {
      const b = new Uint8Array(8);
      for (let j = 0; j < 8; j++) b[j] = Math.floor(rng2() * 256);
      const f = fnv1a64(b).toString(16), c = crc32(b), s = sha256BytesHex(b);
      if (seenF.has(f) || seenC.has(c) || seenS.has(s)) collide = true;
      seenF.add(f); seenC.add(c); seenS.add(s);
    }
    // (d) 块级定位
    const rng3 = makeRng(0x5E420005);
    const blocks = [];
    for (let i = 0; i < 16; i++) {
      const b = new Uint8Array(64);
      for (let j = 0; j < 64; j++) b[j] = Math.floor(rng3() * 256);
      blocks.push(b);
    }
    const blockHashes = blocks.map(fnv1a64Hex);
    const badIdx = 7;
    const corruptBlock = blocks[badIdx].slice();
    corruptBlock[13] ^= 0x40;
    let located = -1;
    for (let i = 0; i < 16; i++) if (fnv1a64Hex(i === badIdx ? corruptBlock : blocks[i]) !== blockHashes[i]) { located = i; break; }
    const locateOk = located === badIdx;
    // (e) 选择性污染：翻转记录 3 载荷一字节 → 仅该键受损
    const cs = new ChecksummedStore();
    const payloads = [];
    for (let i = 0; i < 5; i++) {
      const b = new Uint8Array(48);
      for (let j = 0; j < 48; j++) b[j] = (i * 37 + j * 11) & 255;
      payloads.push(b);
      cs.put('k' + (i + 1), b);
    }
    const ser = cs.serialize();
    // 定位 k3 载荷起点（帧布局：count varint + 每帧 [keyLen varint, key, payLen varint, payload, crc4, fnv8, sha32]）
    const rr = new ByteReader(ser);
    rr.u8(); rr.u8();
    const nRec = rr.varint();
    let payStart = -1;
    for (let i = 0; i < nRec; i++) {
      const kl = rr.varint(); rr.bytes(kl);
      const pl = rr.varint();
      const start = rr.p;
      rr.bytes(pl); rr.u32le(); rr.bytes(8); rr.bytes(32);
      if (i === 2) payStart = start;
    }
    ser[payStart + 5] ^= 0x01;
    const des = ChecksummedStore.deserialize(ser);
    const selectiveOk = des.corrupted.length === 1 && des.corrupted[0] === 'k3' && des.store.keys().length === 4
      && bytesEqual(des.store.get('k1'), payloads[0]) && bytesEqual(des.store.get('k5'), payloads[4]);
    const ok = vectorsOk && avalancheOk && !collide && locateOk && selectiveOk;
    push('INV-SER-006', '校验和：标准向量 EXACT + 单比特雪崩(均值≥0.3) + 1000 输入无碰撞 + 块级定位 EXACT + 记录级选择性污染', ok,
      ok ? 0 : Math.max(0.3 - Math.min(crcMean, fnvMean), 0),
      'vectors=' + vectorsOk + ' crcMean=' + crcMean.toFixed(3) + ' fnvMean=' + fnvMean.toFixed(3) + ' collide=' + collide + ' locate=' + located + '(want 7) corrupted=' + des.corrupted.join(',') + ' kept=' + des.store.keys().length, TOL.EXACT);
  }

  /* ---------- INV-SER-007：分区 + 对象存储 ---------- */
  {
    // (a) 哈希分区确定性 + 均匀性
    const rng = makeRng(0x5E520001);
    const keys = [];
    for (let i = 0; i < 1000; i++) keys.push('key-' + Math.floor(rng() * 1e9).toString(36));
    let detOk = true;
    for (const k of keys.slice(0, 100)) for (let r = 0; r < 3; r++) if (hashPartition(k, 8) !== hashPartition(k, 8)) detOk = false;
    const bins = new Array(8).fill(0);
    for (const k of keys) bins[hashPartition(k, 8)]++;
    const maxDev = Math.max(...bins.map((b) => Math.abs(b - 125)));
    const uniformOk = maxDev <= 62;
    // (b) 一致性哈希环
    const ringA = new ConsistentHashRing(['n1', 'n2', 'n3', 'n4'], 50);
    const ringB = new ConsistentHashRing(['n1', 'n2', 'n3', 'n4', 'n5'], 50);
    const ringC = new ConsistentHashRing(['n1', 'n2', 'n3'], 50);
    let ringDet = true;
    for (const k of keys.slice(0, 200)) if (ringA.route(k) !== ringA.route(k)) ringDet = false;
    const addFrac = ringB.movedFraction(keys, ringA);
    const delFrac = ringC.movedFraction(keys, ringA);
    const addOk = addFrac <= 2 / 5 && addFrac >= 0.02;
    const delOk = delFrac <= 0.5;
    // (c) 范围分区
    const rp = new RangePartitioner(['h', 'p', 'x']);
    const rangeOk = rp.route('a') === 0 && rp.route('h') === 1 && rp.route('hello') === 1
      && rp.route('p') === 2 && rp.route('q') === 2 && rp.route('x') === 3 && rp.route('z') === 3;
    // (d) 对象存储
    const os = new ObjectStore();
    const body = new Uint8Array([1, 2, 3, 4, 5]);
    const r1 = os.put('b1', 'k1', body, { kind: 'x' });
    const r2 = os.put('b2', 'k2', body.slice(), { kind: 'y' });
    const dedupOk = os.bodies.size === 1 && r1.etag === r2.etag;
    const body2 = new Uint8Array([9, 9, 9]);
    const r3 = os.put('b1', 'k1', body2, { kind: 'z' });
    const etagDiff = r3.etag !== r1.etag;
    const v1Read = os.get('b1', 'k1', 1);
    const v2Read = os.get('b1', 'k1', 2);
    const verOk = r3.version === 2 && bytesEqual(v1Read.body, body) && bytesEqual(v2Read.body, body2)
      && deepEqual(v1Read.meta, { kind: 'x' }) && deepEqual(v2Read.meta, { kind: 'z' });
    const headOk = deepEqual(os.head('b1', 'k1'), { etag: r3.etag, size: 3, meta: { kind: 'z' }, version: 2 });
    os.delete('b1', 'k1');
    let delThrew = false;
    try { os.get('b1', 'k1'); } catch (e) { delThrew = true; }
    const ok = detOk && uniformOk && ringDet && addOk && delOk && rangeOk && dedupOk && etagDiff && verOk && headOk && delThrew;
    push('INV-SER-007', '分区确定+均匀(8桶偏差≤62)+一致性环(增≤2/5且≥0.02,删≤0.5)+范围路由 EXACT；对象存储去重/版本/元数据/删除 EXACT', ok,
      ok ? 0 : Math.max(maxDev, 0),
      'det=' + detOk + ' maxDev=' + maxDev + ' addFrac=' + addFrac.toFixed(3) + '(≤' + (2 / 5).toFixed(2) + ') delFrac=' + delFrac.toFixed(3) + ' range=' + rangeOk + ' dedup=' + dedupOk + ' bodies=' + os.bodies.size + ' ver=' + verOk + ' head=' + headOk + ' delThrew=' + delThrew, TOL.EXACT);
  }

  /* ---------- INV-SER-008：缓存语义（LRU + TTL/SWR） ---------- */
  {
    // (a) LRU 容量 3：put a,b,c → get a → put d 逐出 b
    const lru = new LRUCache(3);
    lru.put('a', 1); lru.put('b', 2); lru.put('c', 3);
    lru.get('a'); // a → MRU
    const evicted = lru.put('d', 4);
    const orderOk = evicted === 'b' && lru.keys().join(',') === 'c,a,d';
    const sizeOk = lru.size <= 3;
    lru.get('zz'); // miss
    const statOk = lru.gets === lru.hits + lru.misses && lru.hits === 1 && lru.misses === 1;
    // (b) 证伪：容量 1 连续 put 必逐出前值
    const lru1 = new LRUCache(1);
    lru1.put('x', 1);
    const ev1 = lru1.put('y', 2);
    const falsOk = ev1 === 'x' && lru1.get('x') === undefined && lru1.get('y') === 2;
    // (c) TTL 可注入时钟
    let clock = 100;
    const ttl = new TTLCache({ now: () => clock });
    ttl.set('k', 'v1', 50); // expireAt 150
    const beforeHit = ttl.get('k');
    const beforeOk = beforeHit.hit === true && beforeHit.value === 'v1';
    clock = 150;
    const afterMiss = ttl.get('k');
    const afterOk = afterMiss.hit === false && afterMiss.expired === true && ttl.has('k') === false; // 惰性清除
    // (d) SWR：过期仍返回旧值 + revalidate
    clock = 100;
    ttl.set('s', 'old', 50);
    clock = 160;
    const swr = ttl.getStale('s');
    const swrOk = swr.hit === true && swr.value === 'old' && swr.stale === true && swr.revalidate === true;
    ttl.refresh('s', 'new', 50); // expireAt 210
    clock = 209;
    const refreshed = ttl.get('s');
    const refreshOk = refreshed.hit === true && refreshed.value === 'new';
    clock = 210;
    const refreshExpired = ttl.get('s');
    const refreshExpOk = refreshExpired.hit === false;
    // (e) ttl=0 即 miss
    ttl.set('z', 'v', 0);
    const zeroOk = ttl.get('z').hit === false;
    const ok = orderOk && sizeOk && statOk && falsOk && beforeOk && afterOk && swrOk && refreshOk && refreshExpOk && zeroOk;
    push('INV-SER-008', 'LRU 逐出序 EXACT + size≤cap + hits+misses==gets + cap=1 证伪；TTL：过期前 hit/过期 miss/惰性清除 + SWR 旧值+revalidate + refresh 生效 + ttl=0 即 miss', ok,
      ok ? 0 : 1,
      'evicted=' + evicted + ' keys=[' + lru.keys().join(',') + '] stats=' + lru.hits + '/' + lru.misses + ' fals=' + falsOk + ' before=' + beforeOk + ' after=' + afterOk + ' swr=' + swrOk + ' refresh=' + refreshOk + '/' + refreshExpOk + ' zero=' + zeroOk, TOL.EXACT);
  }

  return buildReport('serialize-forge', '1.0.0', 'L7', cases, invariants, '0x9E3779B9');
}

/* ============================================================================
 * 人类可读表格（零依赖本地实现）
 * ========================================================================== */
function formatTable(r) {
  if (!r) return 'NO REPORT';
  const lines = [];
  lines.push('=== ' + r.module + ' v' + r.version + ' (' + r.layer + ') ' + (r.failed === 0 ? 'PASS' : 'FAIL') + ' ===');
  lines.push('passed ' + r.passed + ' / failed ' + r.failed + ' / hash ' + (r.hash || '').slice(0, 24));
  for (const c of (r.cases || [])) {
    const mark = c.status === 'pass' ? '+OK' : '-XX';
    lines.push('  ' + mark + ' ' + c.id + ' ' + c.name + '  act=' + (typeof c.actual === 'number' ? c.actual.toExponential(3) : c.actual) + ' tol=' + c.tol);
  }
  return lines.join('\n');
}

/* ============================================================================
 * API 导出 + 双端挂载（UMD）+ CLI 自检
 * ========================================================================== */
const API = {
  name: 'serialize-forge', version: '1.0.0', layer: 'L7', TOL,
  makeRng, sha256Hex, sha256BytesHex,
  utf8Bytes, utf8Decode, bytesToHex, hexToBytes, bytesEqual,
  crc32, fnv1a64, fnv1a64Hex,
  ByteWriter, ByteReader, BitWriter, BitReader,
  zigzag32, unzigzag32,
  deepEqual, deepClone, deepFreeze, mapEqual,
  encodeJson, decodeJson,
  encodeBinary, decodeBinary,
  encodeColumnar, decodeColumnar,
  encodeProtobuf, decodeProtobuf,
  MappedRegion,
  encodeDelta, applyDelta, encodeDeltaBytes, decodeDeltaBytes, indexOfBytes,
  ChecksummedStore,
  VersionedStore, SnapshotStore,
  hashPartition, ConsistentHashRing, RangePartitioner, ringHash64,
  ObjectStore,
  LRUCache, TTLCache,
  WriteAheadLog, parseWalFrames, recoverWal,
  backupState, restoreBackup, pitr,
  randomValue, randStr,
  selfCheck, formatTable
};
if (typeof module !== 'undefined' && module.exports) module.exports = API;
if (typeof globalThis !== 'undefined') globalThis.ForgeSerialize = API;

if (typeof process !== 'undefined' && process.argv[1] && /serialize-forge\.js$/.test(process.argv[1]) && (process.argv.includes('--self-check') || process.argv.includes('--json'))) {
  const r = selfCheck();
  process.stdout.write(process.argv.includes('--json') ? JSON.stringify(r, null, 2) + '\n' : formatTable(r) + '\n');
  process.exit(r.failed === 0 ? 0 : 1);
}
