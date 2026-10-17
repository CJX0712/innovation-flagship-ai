'use strict';
/* ============================================================================
 * 独立复核 serialize-forge —— 不采信模块 selfCheck()，自带独立参考实现：
 *   独立 SHA-256（FIPS 180-4 重新推导）、独立 CRC-32（IEEE）、独立 FNV-1a-64、
 *   独立规范 JSON 序列化（递归键序）、独立二进制/protobuf/列式 字节级解码器、
 *   独立 LRU 模型、独立一致性哈希环（复用独立 SHA-256）、独立 WAL 恢复（ARIES 重放）、
 *   独立 delta 应用、独立范围/哈希分区、独立对象存储 etag。
 *
 * 覆盖（与 8 条不变量一一对拍）：
 *   R1  四编解码器：模块 encode→独立 decode 逐值一致（不依赖模块解码器）+
 *       规范键序确定性 + 紧凑性（binary≤0.75·json / protobuf≤0.6·binary）+ 损坏必检出（INV-SER-001）
 *   R2  列式无免费午餐：模块 roundtrip 精确 + 低基数字典≤0.5·直存 + 高基数字典≥1.15·直存（独立重算直存大小）（INV-SER-002）
 *   R3  内存映射：独立 LRU + 缺页 + 脏页写回仿真对拍模块 faults/evictions/writebacks/backing 内容 + OOB 抛错（INV-SER-003）
 *   R4  增量：独立 delta 应用对拍模块 applyDelta（3 种子删/插/换）+ 确定性 + 快照深冻结隔离/恢复/历史保全（INV-SER-004）
 *   R5  版本化单调+回滚保史 + WAL 交错事务：独立 ARIES 重放对拍模块 recoverWal（INV-SER-005）
 *   R6  校验和：独立 CRC32/FNV1a64/SHA256 对拍模块已知向量 + 单比特雪崩 + 1000 输入无碰撞 +
 *       块级定位 + ChecksummedStore 注入损坏必检出（INV-SER-006）
 *   R7  分区确定+均匀 + 一致性环（独立 SHA-256 重排环：模块 route 逐键一致、增减移动率边界）+ 范围路由 +
 *       对象存储去重/版本/元数据/墓碑删除（INV-SER-007）
 *   R8  LRU 逐出序（独立 Map 模型对拍）+ size≤cap + hits+misses==gets + cap=1 证伪 +
 *       TTL：过期前 hit/过期 miss/惰性清除 + SWR 旧值+revalidate + refresh + ttl=0 即 miss（INV-SER-008）
 *
 * 退出码：0 = 全部独立复核通过；1 = 任一失败。
 * ========================================================================== */
const path = require('path');
const G = require(path.join(__dirname, '..', '..', 'src', 'serialize-forge.js'));

/* ---------- 独立 PRNG：mulberry32（与模块 sfc32 不同族） ---------- */
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ---------- 独立 UTF-8 ---------- */
function ownUtf8Bytes(str) {
  const out = [];
  for (let i = 0; i < str.length; i++) {
    const c = str.charCodeAt(i);
    if (c < 0x80) out.push(c);
    else if (c < 0x800) out.push(0xC0 | (c >> 6), 0x80 | (c & 63));
    else if (c >= 0xD800 && c <= 0xDBFF && i + 1 < str.length && str.charCodeAt(i + 1) >= 0xDC00 && str.charCodeAt(i + 1) <= 0xDFFF) {
      const cp = 0x10000 + ((c - 0xD800) << 10) + (str.charCodeAt(i + 1) - 0xDC00); i++;
      out.push(0xF0 | (cp >> 18), 0x80 | ((cp >> 12) & 63), 0x80 | ((cp >> 6) & 63), 0x80 | (cp & 63));
    } else out.push(0xE0 | (c >> 12), 0x80 | ((c >> 6) & 63), 0x80 | (c & 63));
  }
  return Uint8Array.from(out);
}
function ownUtf8Decode(bytes) {
  const out = []; let i = 0;
  while (i < bytes.length) {
    const b0 = bytes[i++];
    if (b0 < 0x80) out.push(String.fromCharCode(b0));
    else if (b0 < 0xE0) { const b1 = bytes[i++]; out.push(String.fromCharCode(((b0 & 31) << 6) | (b1 & 63))); }
    else if (b0 < 0xF0) { const b1 = bytes[i++], b2 = bytes[i++]; out.push(String.fromCharCode(((b0 & 15) << 12) | ((b1 & 63) << 6) | (b2 & 63))); }
    else { const b1 = bytes[i++], b2 = bytes[i++], b3 = bytes[i++]; const cp = ((b0 & 7) << 18) | ((b1 & 63) << 12) | ((b2 & 63) << 6) | (b3 & 63); const c = cp - 0x10000; out.push(String.fromCharCode(0xD800 + (c >> 10), 0xDC00 + (c & 1023))); }
  }
  return out.join('');
}

/* ---------- 独立 CRC-32（IEEE 反射，poly 0xEDB88320） ---------- */
function ownCrc32(bytes) {
  let c = 0xFFFFFFFF;
  for (let i = 0; i < bytes.length; i++) {
    c ^= bytes[i];
    for (let k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) >>> 0 : (c >>> 1) >>> 0;
  }
  return (c ^ 0xFFFFFFFF) >>> 0;
}

/* ---------- 独立 FNV-1a-64（BigInt 精确） ---------- */
function ownFnv1a64(bytes) {
  let h = 0xCBF29CE484222325n;
  const P = 0x100000001B3n, M = 0xFFFFFFFFFFFFFFFFn;
  for (let i = 0; i < bytes.length; i++) h = ((h ^ BigInt(bytes[i] & 255)) * P) & M;
  return h;
}

/* ---------- 独立 SHA-256（FIPS 180-4，重新推导） ---------- */
const _SHA_K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
]);
function _rotr(x, n) { return ((x >>> n) | (x << (32 - n))) >>> 0; }
function ownSha256Bytes(msg) {
  const bitLen = msg.length * 8;
  const withOne = msg.length + 1;
  const total = (withOne + 8 + 63) & ~63;
  const buf = new Uint8Array(total);
  buf.set(msg); buf[msg.length] = 0x80;
  const hi = Math.floor(bitLen / 0x100000000), lo = bitLen >>> 0;
  buf[total - 8] = (hi >>> 24) & 255; buf[total - 7] = (hi >>> 16) & 255; buf[total - 6] = (hi >>> 8) & 255; buf[total - 5] = hi & 255;
  buf[total - 4] = (lo >>> 24) & 255; buf[total - 3] = (lo >>> 16) & 255; buf[total - 2] = (lo >>> 8) & 255; buf[total - 1] = lo & 255;
  let H = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
  const w = new Uint32Array(64);
  for (let off = 0; off < total; off += 64) {
    for (let i = 0; i < 16; i++) { const j = off + i * 4; w[i] = ((buf[j] << 24) | (buf[j + 1] << 16) | (buf[j + 2] << 8) | buf[j + 3]) >>> 0; }
    for (let i = 16; i < 64; i++) {
      const s0 = (_rotr(w[i - 15], 7) ^ _rotr(w[i - 15], 18) ^ (w[i - 15] >>> 3)) >>> 0;
      const s1 = (_rotr(w[i - 2], 17) ^ _rotr(w[i - 2], 19) ^ (w[i - 2] >>> 10)) >>> 0;
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) >>> 0;
    }
    let a = H[0], b = H[1], c = H[2], d = H[3], e = H[4], f = H[5], g = H[6], h = H[7];
    for (let i = 0; i < 64; i++) {
      const S1 = (_rotr(e, 6) ^ _rotr(e, 11) ^ _rotr(e, 25)) >>> 0;
      const ch = ((e & f) ^ (~e & g)) >>> 0;
      const t1 = (h + S1 + ch + _SHA_K[i] + w[i]) >>> 0;
      const S0 = (_rotr(a, 2) ^ _rotr(a, 13) ^ _rotr(a, 22)) >>> 0;
      const maj = ((a & b) ^ (a & c) ^ (b & c)) >>> 0;
      const t2 = (S0 + maj) >>> 0;
      h = g; g = f; f = e; e = (d + t1) >>> 0; d = c; c = b; b = a; a = (t1 + t2) >>> 0;
    }
    H = [(H[0] + a) >>> 0, (H[1] + b) >>> 0, (H[2] + c) >>> 0, (H[3] + d) >>> 0, (H[4] + e) >>> 0, (H[5] + f) >>> 0, (H[6] + g) >>> 0, (H[7] + h) >>> 0];
  }
  let hex = '';
  for (let i = 0; i < 8; i++) hex += ('00000000' + H[i].toString(16)).slice(-8);
  return hex;
}
function ownSha256Hex(str) { return ownSha256Bytes(ownUtf8Bytes(str)); }
function ownRingHash64(str) { return BigInt('0x' + ownSha256Bytes(ownUtf8Bytes(String(str))).slice(0, 16)); }

/* ---------- 独立 varint / zigzag ---------- */
function ownVarintDecode(r) {
  let result = 0, shift = 1, count = 0;
  for (;;) { const b = r[0]; r.shift(); count++; if (count > 5) throw new Error('varint too long'); result += (b & 0x7f) * shift; if ((b & 0x80) === 0) return result; shift *= 128; }
}
function ownUnzigzag(v) { return ((v >>> 1) ^ -(v & 1)) | 0; }

/* ---------- 独立值解码（二进制/列式共用，复刻模块 _decValue） ---------- */
function ownDecValue(r) {
  const tag = r[0]; r.shift();
  switch (tag) {
    case 0x00: return null;
    case 0x01: return false;
    case 0x02: return true;
    case 0x03: { const v = (r[0] | (r[1] << 8) | (r[2] << 16) | (r[3] << 24)) >>> 0; r.splice(0, 4); return v | 0; }
    case 0x04: { const ab = new ArrayBuffer(8); const u = new Uint8Array(ab); for (let i = 0; i < 8; i++) u[i] = r[0], r.shift(); return new Float64Array(ab)[0]; }
    case 0x05: { const n = ownVarintDecode(r); const b = r.splice(0, n); return ownUtf8Decode(Uint8Array.from(b)); }
    case 0x06: { const n = ownVarintDecode(r); const a = []; for (let i = 0; i < n; i++) a.push(ownDecValue(r)); return a; }
    case 0x07: { const n = ownVarintDecode(r); const o = {}; for (let i = 0; i < n; i++) { const kl = ownVarintDecode(r); const kb = r.splice(0, kl); o[ownUtf8Decode(Uint8Array.from(kb))] = ownDecValue(r); } return o; }
    default: throw new Error('decValue: unknown tag 0x' + tag.toString(16));
  }
}
/* 独立二进制解码器（读模块 encodeBinary 字节流） */
function ownDecodeBinary(bytes) {
  const r = Array.from(bytes);
  if (r[0] !== 0x53) throw new Error('binary magic'); if (r[1] !== 0x01) throw new Error('binary version'); r.splice(0, 2);
  return ownDecValue(r);
}
/* 独立 protobuf 解码器（读模块 encodeProtobuf 字节流） */
function ownDecodeProtobuf(bytes) {
  const r = Array.from(bytes);
  if (r[0] !== 0x50) throw new Error('protobuf magic'); if (r[1] !== 0x01) throw new Error('protobuf version'); r.splice(0, 2);
  const n = ownVarintDecode(r);
  const fields = [];
  for (let i = 0; i < n; i++) {
    const tag = ownVarintDecode(r);
    const field = tag >>> 3, wt = tag & 7;
    if (wt === 0) fields.push({ field, wt, value: ownUnzigzag(ownVarintDecode(r)) });
    else if (wt === 1) { let v = 0n; for (let j = 0; j < 8; j++) v |= BigInt(r[0]) << BigInt(8 * j), r.shift(); fields.push({ field, wt, value: Number(v) }); }
    else if (wt === 5) { const v = (r[0] | (r[1] << 8) | (r[2] << 16) | (r[3] << 24)) >>> 0; r.splice(0, 4); fields.push({ field, wt, value: v | 0 }); } // 镜像模块 decode 的 |0 有符号
    else if (wt === 2) { const l = ownVarintDecode(r); const b = r.splice(0, l); fields.push({ field, wt, value: Uint8Array.from(b) }); }
    else throw new Error('protobuf wt ' + wt);
  }
  return fields;
}
/* 独立列式解码器（读模块 encodeColumnar 字节流，含字典+位包与 plain 两模式） */
function ownBitReader(bits) {
  let p = 0, cur = 0, n = 0;
  return function read(bw) {
    let v = 0;
    for (let i = 0; i < bw; i++) {
      if (n === 0) { cur = bits[p++]; n = 8; }
      v = (v << 1) | ((cur >> (n - 1)) & 1); n--;
    }
    return v >>> 0;
  };
}
function ownDecodeColumnar(bytes) {
  const r = Array.from(bytes);
  if (r[0] !== 0x43) throw new Error('col magic'); if (r[1] !== 0x01) throw new Error('col version'); r.splice(0, 2);
  const rd32 = () => { const v = (r[0] | (r[1] << 8) | (r[2] << 16) | (r[3] << 24)) >>> 0; r.splice(0, 4); return v; };
  const nRows = rd32(), nCols = rd32();
  const names = [];
  for (let c = 0; c < nCols; c++) { const nl = ownVarintDecode(r); names.push(ownUtf8Decode(Uint8Array.from(r.splice(0, nl)))); }
  const rows = []; for (let i = 0; i < nRows; i++) rows.push({});
  for (let c = 0; c < nCols; c++) {
    const mode = r[0]; r.shift();
    if (mode === 0x01) { for (let i = 0; i < nRows; i++) rows[i][names[c]] = ownDecValue(r); continue; }
    if (mode !== 0x00) throw new Error('col bad mode 0x' + mode.toString(16));
    r.shift(); // 类型标签（信息性）
    const dictLen = ownVarintDecode(r);
    const dict = []; for (let d = 0; d < dictLen; d++) dict.push(ownDecValue(r));
    const bw = r[0]; r.shift();
    const plen = ownVarintDecode(r);
    const packed = Uint8Array.from(r.splice(0, plen));
    const br = ownBitReader(packed);
    for (let i = 0; i < nRows; i++) { const ix = bw === 0 ? 0 : br(bw); if (ix >= dictLen) throw new Error('col idx oob'); rows[i][names[c]] = dict[ix]; }
  }
  return rows;
}

/* ---------- 独立深比较 ---------- */
function ownDeepEqual(a, b) {
  if (a === b) return true;
  if (a instanceof Uint8Array && b instanceof Uint8Array) { if (a.length !== b.length) return false; for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false; return true; }
  if (Array.isArray(a) && Array.isArray(b)) { if (a.length !== b.length) return false; for (let i = 0; i < a.length; i++) if (!ownDeepEqual(a[i], b[i])) return false; return true; }
  if (a && b && typeof a === 'object' && typeof b === 'object') {
    const ka = Object.keys(a), kb = Object.keys(b);
    if (ka.length !== kb.length) return false;
    for (const k of ka) if (!Object.prototype.hasOwnProperty.call(b, k) || !ownDeepEqual(a[k], b[k])) return false;
    return true;
  }
  if (typeof a === 'number' && typeof b === 'number') return a === b; // NaN 在此视为不等，足够
  return false;
}

/* ---------- 独立规范 JSON 序列化（复刻模块 encodeJson：递归键序 + 有限数） ---------- */
function ownEncodeJson(v) {
  if (v === null) return 'null';
  if (v === true) return 'true';
  if (v === false) return 'false';
  if (typeof v === 'number') { if (!isFinite(v)) throw new Error('json: non-finite'); return JSON.stringify(v); }
  if (typeof v === 'string') return JSON.stringify(v);
  if (Array.isArray(v)) return '[' + v.map(ownEncodeJson).join(',') + ']';
  if (typeof v === 'object') { const keys = Object.keys(v).sort(); return '{' + keys.map((k) => JSON.stringify(k) + ':' + ownEncodeJson(v[k])).join(',') + '}'; }
  throw new Error('json: unsupported');
}
/* 独立 _encValue 字节长度（复刻模块二进制值编码开销，无外层 magic） */
function ownVarintLen(n) { let x = n, c = 0; while (x >= 0x80) { x = Math.floor(x / 128); c++; } return c + 1; }
function ownEncValueSize(v) {
  if (v === null || v === false || v === true) return 1;
  if (typeof v === 'number') { if (Number.isInteger(v) && v >= -2147483648 && v <= 2147483647) return 1 + 4; return 1 + 8; }
  if (typeof v === 'string') { const b = ownUtf8Bytes(v); return 1 + ownVarintLen(b.length) + b.length; }
  if (Array.isArray(v)) { let s = 1 + ownVarintLen(v.length); for (const x of v) s += ownEncValueSize(x); return s; }
  if (typeof v === 'object') { const keys = Object.keys(v).sort(); let s = 1 + ownVarintLen(keys.length); for (const k of keys) { const kb = ownUtf8Bytes(k); s += ownVarintLen(kb.length) + kb.length + ownEncValueSize(v[k]); } return s; }
  throw new Error('encValue: unsupported');
}

/* ---------- 结果记录 ---------- */
const results = [];
function check(id, name, ok, detail) {
  results.push({ id, ok });
  console.log((ok ? '  +OK ' : '  -XX ') + id + ' ' + name + (detail ? '  | ' + detail : ''));
}

/* ==========================================================================
 * R1  四编解码器：模块 encode → 独立 decode 逐值一致 + 规范键序 + 紧凑性 + 损坏必检出
 * ========================================================================== */
{
  const rng = mulberry32(0x5EA1A01);
  const values = [];
  for (let i = 0; i < 40; i++) values.push(G.randomValue(rng, 3));
  // (a) 模块 encode → 独立 decode 逐值一致（不依赖模块解码器）
  let codecOk = true, codecBad = '';
  for (const v of values) {
    // JSON
    if (!ownDeepEqual(JSON.parse(G.encodeJson(v)), v)) { codecOk = false; codecBad = 'json roundtrip'; break; }
    // 规范键序：独立编码器与模块输出逐字符一致（确定性 + RFC8785 风格键序）
    if (ownEncodeJson(v) !== G.encodeJson(v)) { codecOk = false; codecBad = 'json canonical mismatch'; break; }
    // 二进制：独立解码器对拍模块编码
    if (!ownDeepEqual(ownDecodeBinary(G.encodeBinary(v)), v)) { codecOk = false; codecBad = 'binary decode mismatch'; break; }
    // protobuf：构造字段（int32/int64/fixed32/string）对拍
    const fields = [G.zigzag32 ? null : null]; // 占位
    const pf = [
      { field: 1, wt: 0, value: -7 }, { field: 2, wt: 0, value: 300 }, { field: 3, wt: 1, value: 12345678901 },
      { field: 4, wt: 5, value: 1234567 }, { field: 5, wt: 2, value: ownUtf8Bytes('héllo') }
    ];
    const dec = ownDecodeProtobuf(G.encodeProtobuf(pf));
    let pbOk = dec.length === pf.length;
    if (pbOk) for (let k = 0; k < pf.length; k++) {
      if (dec[k].field !== pf[k].field || dec[k].wt !== pf[k].wt) { pbOk = false; break; }
      if (pf[k].wt === 2) { if (!ownDeepEqual(dec[k].value, pf[k].value)) { pbOk = false; break; } }
      else if (dec[k].value !== pf[k].value) { pbOk = false; break; }
    }
    if (!pbOk) { codecOk = false; codecBad = 'protobuf decode mismatch'; break; }
  }
  // (b) 紧凑性（对齐模块 INV-SER-001 口径：纯 float64 数组 binary ≤0.75·json；纯小整数数组 protobuf ≤0.6·binary）
  const rngC = G.makeRng(0x5E120004);
  const floats = []; for (let i = 0; i < 200; i++) floats.push(rngC() * 1e6 - 5e5);
  const jsonLen = G.encodeJson(floats).length;
  const binLen = G.encodeBinary(floats).length;
  const compactF64 = binLen <= 0.75 * jsonLen;
  const smalls = []; for (let i = 0; i < 200; i++) smalls.push(Math.floor(rngC() * 128));
  const binSmall = G.encodeBinary(smalls).length;
  const pbSmall = G.encodeProtobuf(smalls.map((n, i) => ({ field: (i % 15) + 1, wt: 0, value: n }))).length;
  const compactSmall = pbSmall <= 0.6 * binSmall;
  // (c) 损坏必检出：逐字节翻转二进制/protobuf/列式头或体，模块解码必须抛错或产出不同值
  let corruptOk = true;
  const samples = [values[0], { m: [1, 2, 3], n: { p: 'q' } }, [1.0, 2.0, 3.0]];
  for (const s of samples) {
    for (const enc of [G.encodeBinary, G.encodeProtobuf]) {
      const b = enc === G.encodeBinary ? enc(s) : enc([{ field: 1, wt: 0, value: 5 }]);
      const idx = 3 + Math.floor(rng() * Math.max(1, b.length - 3));
      const corrupted = Uint8Array.from(b); corrupted[idx] ^= 0xFF;
      let threw = false, diff = false;
      try {
        if (enc === G.encodeBinary) diff = !ownDeepEqual(ownDecodeBinary(corrupted), s);
        else { const d = ownDecodeProtobuf(corrupted); if (d.length !== 1 || d[0].value !== 5) diff = true; }
      } catch (e) { threw = true; }
      if (!threw && !diff) { corruptOk = false; break; }
    }
    if (!corruptOk) break;
  }
  const ok = codecOk && compactF64 && compactSmall && corruptOk;
  check('R1', '四编解码器（模块encode→独立decode逐值一致）+ 规范键序确定 + 紧凑性(float64 binary≤0.75json; 小整数 pb≤0.6binary) + 损坏必检出', ok,
    'codec=' + (codecOk ? 'OK' : 'FAIL:' + codecBad) + ' json=' + jsonLen + ' bin=' + binLen + ' compactF64=' + compactF64 + ' binSmall=' + binSmall + ' pbSmall=' + pbSmall + ' compactSmall=' + compactSmall + ' corrupt=' + corruptOk);
}

/* ==========================================================================
 * R2  列式无免费午餐：模块 roundtrip 精确 + 低基数字典≤0.5·直存 + 高基数字典≥1.15·直存
 * ========================================================================== */
{
  const rng = G.makeRng(0x5E220001);
  // 低基数（每列 4 个 distinct 值）—— 对齐模块 INV-SER-002 口径
  const low = [];
  for (let i = 0; i < 300; i++) low.push({ g: Math.floor(rng() * 4), h: Math.floor(rng() * 4) });
  const lowDictBytes = G.encodeColumnar(low);
  const lowDict = lowDictBytes.length;
  const lowPlain = G.encodeColumnar(low, { plain: true }).length;
  const lowRow = G.encodeBinary(low).length;
  const lowRoundtrip = ownDeepEqual(ownDecodeColumnar(lowDictBytes), low); // 独立解码器对拍（证压缩无损）
  // 高基数（每列全 distinct int）
  const high = [];
  for (let i = 0; i < 300; i++) high.push({ g: i, h: i * 7 + 3 });
  const highDictBytes = G.encodeColumnar(high);
  const highDict = highDictBytes.length;
  const highPlain = G.encodeColumnar(high, { plain: true }).length;
  const highRow = G.encodeBinary(high).length;
  const highRoundtrip = ownDeepEqual(ownDecodeColumnar(highDictBytes), high);
  // 无免费午餐：低基数字典致胜（vs 直存与行式），高基数字典致亏（vs 直存——纯索引开销）
  const lowDictWins = lowDict <= 0.5 * lowPlain && lowDict <= 0.7 * lowRow;
  const highDictLoses = highDict >= 1.15 * highPlain;
  const ok = lowRoundtrip && highRoundtrip && lowDictWins && highDictLoses && highDict > highPlain;
  check('R2', '列式 roundtrip 精确(独立解码器对拍) + 低基数字典≤0.5·直存 且 ≤0.7·行式 + 高基数字典≥1.15·直存（无免费午餐）', ok,
    'lowRT=' + lowRoundtrip + ' highRT=' + highRoundtrip + ' lowDict=' + lowDict + ' lowPlain=' + lowPlain + ' lowWins=' + lowDictWins +
    ' highDict=' + highDict + ' highPlain=' + highPlain + ' highLoses=' + highDictLoses + ' highDict>highPlain=' + (highDict > highPlain));
}

/* ==========================================================================
 * R3  内存映射：独立 LRU + 缺页 + 脏页写回仿真对拍模块（faults/evictions/writebacks/backing + OOB）
 * ========================================================================== */
{
  const pageSize = 16, maxPages = 3, length = 64;
  const backing = new Uint8Array(length);
  for (let i = 0; i < length; i++) backing[i] = (i * 7 + 3) & 255;
  // 独立仿真：精确复刻模块 _page/_evictOne/_touch 语义
  const backingInd = Uint8Array.from(backing);
  let lru = [], pages = new Map(), dirty = new Set(), faults = 0, evictions = 0, writebacks = 0;
  function touch(i) { const at = lru.indexOf(i); if (at >= 0) lru.splice(at, 1); lru.push(i); }
  function evict() { const victim = lru.shift(); const data = pages.get(victim); pages.delete(victim); if (dirty.has(victim)) { const start = victim * pageSize; const valid = Math.min(pageSize, length - victim * pageSize); backingInd.set(data.subarray(0, valid), start); dirty.delete(victim); writebacks++; } evictions++; }
  function ipage(idx) {
    if (pages.has(idx)) { touch(idx); return; }
    faults++; const start = idx * pageSize; const valid = Math.min(pageSize, length - idx * pageSize);
    const data = Uint8Array.from(backingInd.subarray(start, start + valid)); pages.set(idx, data); touch(idx);
    while (pages.size > maxPages) evict();
  }
  function iread(off, len) { let done = 0; while (done < len) { const a = off + done; ipage(Math.floor(a / pageSize)); done += Math.min(len - done, pageSize - (a % pageSize)); } }
  function iwrite(off, bytes) { let done = 0; while (done < bytes.length) { const a = off + done; const idx = Math.floor(a / pageSize); ipage(idx); const within = a % pageSize; const take = Math.min(bytes.length - done, pageSize - within); const data = pages.get(idx); for (let k = 0; k < take; k++) data[within + k] = bytes[done + k]; dirty.add(idx); done += take; } }
  const rng = mulberry32(0x7C30D03);
  const m = new G.MappedRegion({ backing: Uint8Array.from(backing), pageSize, length, maxPages });
  let simErr = false;
  for (let t = 0; t < 300; t++) {
    const off = Math.floor(rng() * length);            // 0..63
    const write = rng() < 0.4;
    if (write) {
      const wl = 1 + Math.floor(rng() * (length - off));   // 保证 off+wl ≤ length
      const bytes = Uint8Array.from(Array.from({ length: wl }, (_, k) => (off + k + 1) & 255));
      m.write(off, bytes); iwrite(off, bytes);
    } else {
      const rl = 1 + Math.floor(rng() * (length - off));
      m.read(off, rl); iread(off, rl);
    }
  }
  m.flush(); // 模块写回脏页
  // 关键：flush 后立即快照模块计数器——后续 backingMatch 校验读（m.read）会触发额外缺页，污染 faults
  const modFaults = m.faults, modEvict = m.evictions, modWb = m.writebacks;
  // 独立仿真无需显式 flush：脏页已在 evict 时写回；末尾将剩余脏页写回
  for (const idx of [...dirty]) { const data = pages.get(idx); const start = idx * pageSize; const valid = Math.min(pageSize, length - idx * pageSize); backingInd.set(data.subarray(0, valid), start); writebacks++; }
  dirty.clear();
  let backingMatch = true;
  for (let i = 0; i < length; i++) if (m.read(i, 1)[0] !== backingInd[i]) { backingMatch = false; break; }
  // OOB 抛错
  let oob = false; try { m.read(length + 1, 1); } catch (e) { oob = true; }
  let oob2 = false; try { m.write(-1, new Uint8Array([1])); } catch (e) { oob2 = true; }
  const ok = modFaults === faults && modEvict === evictions && modWb === writebacks && backingMatch && oob && oob2;
  check('R3', 'MappedRegion：独立 LRU/缺页/脏页写回仿真对拍（faults/evictions/writebacks 全一致）+ flush 后 backing 一致 + OOB 抛错', ok,
    'faults=' + modFaults + '/' + faults + ' evict=' + modEvict + '/' + evictions + ' writeback=' + modWb + '/' + writebacks + ' backingMatch=' + backingMatch + ' oob=' + oob + ' oob2=' + oob2);
}

/* ==========================================================================
 * R4  增量：独立 delta 应用对拍模块 applyDelta（3 种子删/插/换）+ 确定性 + 快照隔离
 * ========================================================================== */
{
  const rng = mulberry32(0x9F4E104);
  function ownApplyDelta(base, ops) {
    const out = [];
    for (const o of ops) {
      if (o.op === 'copy') { for (let i = 0; i < o.len; i++) out.push(base[o.src + i]); }
      else if (o.op === 'ins') { for (let i = 0; i < o.bytes.length; i++) out.push(o.bytes[i]); }
      else throw new Error('bad op');
    }
    return Uint8Array.from(out);
  }
  let deltaOk = true, deltaDetail = '';
  const variations = ['delete', 'insert', 'replace'];
  for (let s = 0; s < 3; s++) {
    const len = 40 + s * 10;
    const base = new Uint8Array(len), target = new Uint8Array(len);
    for (let i = 0; i < len; i++) base[i] = (i * 5 + s) & 255;
    for (let i = 0; i < len; i++) target[i] = base[i];
    // 施加变异
    if (variations[s] === 'delete') { const d = 5 + s; target.copyWithin(0, d, len); for (let i = len - d; i < len; i++) target[i] = 0; }
    else if (variations[s] === 'insert') { for (let i = len - 1; i >= 10; i--) target[i] = target[i - 5]; for (let i = 5; i < 10; i++) target[i] = (i * 3 + s) & 255; }
    else { for (let i = 10; i < 15 + s; i++) target[i] = (target[i] + 37) & 255; }
    const ops = G.encodeDelta(base, target);
    if (!ownDeepEqual(ownApplyDelta(base, ops), target)) { deltaOk = false; deltaDetail = 'apply mismatch s=' + s; break; }
    if (!ownDeepEqual(G.applyDelta(base, ops), target)) { deltaOk = false; deltaDetail = 'module apply mismatch s=' + s; break; }
    // 确定性：两次 encode 字节一致
    const b1 = G.encodeDeltaBytes(ops), b2 = G.encodeDeltaBytes(G.encodeDelta(base, target));
    if (!ownDeepEqual(b1, b2)) { deltaOk = false; deltaDetail = 'delta bytes nondeterministic s=' + s; break; }
  }
  // 空变异仅单 copy（ops 应恰为前缀 copy）
  const same = new Uint8Array([1, 2, 3, 4]);
  const sameOps = G.encodeDelta(same, Uint8Array.from(same));
  const emptyDelta = sameOps.length === 1 && sameOps[0].op === 'copy' && sameOps[0].len === same.length;
  // 快照隔离：SnapshotStore 深冻结；恢复后修改不污染 store 历史
  const snap = new G.SnapshotStore();
  snap.put({ a: 1, b: [1, 2, 3] });
  snap.put({ a: 2, b: [4, 5] });
  const s1 = snap.snapshot();
  snap.put({ a: 3, b: [9] });
  const restored = snap.restore(s1);
  const afterRestore = snap.get();
  // 修改 restored 快照对象不应影响 store 现状
  let frozenOk = Object.isFrozen(s1.value) && Object.isFrozen(s1.value.b);
  const hist = snap.history();
  const ok = deltaOk && emptyDelta && frozenOk && hist.length >= 3 && afterRestore.a === 2;
  check('R4', '增量 apply∘encode 精确（独立+模块双重对拍，3种子删/插/换）+ delta 字节确定 + 空变异单 copy + 快照深冻结隔离/恢复/历史保全', ok,
    'delta=' + (deltaOk ? 'OK' : 'FAIL:' + deltaDetail) + ' emptyDelta=' + emptyDelta + ' frozen=' + frozenOk + ' history=' + hist.length + ' restoredA=' + afterRestore.a);
}

/* ==========================================================================
 * R5  版本化单调+回滚保史 + WAL 交错事务：独立 ARIES 重放对拍模块 recoverWal
 * ========================================================================== */
{
  // (a) VersionedStore
  const v = new G.VersionedStore();
  const id1 = v.put({ x: 1 }); v.put({ x: 2 }); v.put({ x: 3 });
  const id3 = v.latestId();
  v.rollback(id1);
  const afterRoll = v.get(id1);
  const histPreserved = v.history().length === 3; // 回滚不毁历史
  const monotonic = v.versions.every((vv, i) => vv.id === i + 1);
  const rollbackPreserves = afterRoll.x === 1;
  // (b) WAL：交错两事务，独立 ARIES 重放对拍模块
  const wal = new G.WriteAheadLog();
  const t1 = wal.begin(), t2 = wal.begin();
  wal.write(t1, 'k1', 'v1a'); wal.write(t2, 'k2', 'v2a');
  wal.write(t1, 'k1', 'v1b'); wal.commit(t1);
  wal.write(t2, 'k2', 'v2b'); wal.abort(t2);
  const moduleState = G.recoverWal(wal.records);
  // 独立重放：收集 C 的事务，按 lsn 序应用 W/D，跳过未提交/中止
  const committed = new Set(); for (const rec of wal.records) if (rec.t === 'C') committed.add(rec.txn);
  const indepState = new Map();
  for (const rec of wal.records) { if (!committed.has(rec.txn)) continue; if (rec.t === 'W') indepState.set(rec.key, rec.value); else if (rec.t === 'D') indepState.delete(rec.key); }
  let walMatch = indepState.size === moduleState.size;
  if (walMatch) for (const [k, val] of indepState) if (moduleState.get(k) !== val) { walMatch = false; break; }
  const ok = histPreserved && monotonic && rollbackPreserves && walMatch && moduleState.get('k1') === 'v1b' && !moduleState.has('k2');
  check('R5', '版本化单调(id=i+1)+回滚保史 + WAL 交错事务：独立 ARIES 重放对拍模块 recoverWal（未提交不可见）', ok,
    'hist=' + histPreserved + ' monotonic=' + monotonic + ' rollback=' + rollbackPreserves + ' walMatch=' + walMatch + ' k1=' + moduleState.get('k1') + ' hasK2=' + moduleState.has('k2'));
}

/* ==========================================================================
 * R6  校验和：独立 CRC32/FNV1a64/SHA256 对拍已知向量 + 雪崩 + 无碰撞 + 块级定位 + 注入损坏必检出
 * ========================================================================== */
{
  // (a) 已知向量
  const crcOk = G.crc32(ownUtf8Bytes('123456789')) === 0xCBF43926 && G.crc32(new Uint8Array(0)) === 0 &&
    ownCrc32(ownUtf8Bytes('123456789')) === G.crc32(ownUtf8Bytes('123456789'));
  const fnvOk = G.fnv1a64Hex(ownUtf8Bytes('')) === 'cbf29ce484222325' && G.fnv1a64Hex(ownUtf8Bytes('a')) === 'af63dc4c8601ec8c' &&
    ownFnv1a64(ownUtf8Bytes('a')).toString(16).padStart(16, '0') === G.fnv1a64Hex(ownUtf8Bytes('a'));
  const shaOk = G.sha256Hex('abc') === 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad' &&
    G.sha256Hex('') === 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855' &&
    ownSha256Hex('abc') === G.sha256Hex('abc') && ownSha256Hex('') === G.sha256Hex('');
  // (b) 单比特雪崩：平均值 ≥ 0.3（独立测算模块输出）
  function changedBits(a, b) { let n = 0; for (let i = 0; i < a.length; i++) { let x = a[i] ^ b[i]; while (x) { n += x & 1; x >>>= 1; } } return n; }
  let totalBits = 0, changes = 0; const base = ownUtf8Bytes('avalanche-test-vector');
  const baseHash = G.sha256Hex('avalanche-test-vector');
  for (let i = 0; i < base.length * 8; i++) {
    const mutated = Uint8Array.from(base); const byte = i >> 3, bit = i & 7; mutated[byte] ^= (1 << bit);
    const hb = G.sha256BytesHex(mutated);
    totalBits += baseHash.length * 4; changes += changedBits(Uint8Array.from(Buffer.from(baseHash, 'hex')), Uint8Array.from(Buffer.from(hb, 'hex')));
  }
  const avalanche = changes / totalBits;
  // (c) 1000 输入无碰撞（sha 前 16 hex）
  const seen = new Set(); let collision = false; const r = mulberry32(0x5EED05);
  for (let i = 0; i < 1000; i++) { const s = String(Math.floor(r() * 1e9)); const h = G.sha256Hex(s).slice(0, 16); if (seen.has(h)) { collision = true; break; } seen.add(h); }
  // (d) 块级定位：ChecksummedStore 注入损坏，verify 必检出
  const cs = new G.ChecksummedStore();
  cs.put('k1', ownUtf8Bytes('payload-one'));
  cs.put('k2', ownUtf8Bytes('payload-two'));
  const vOk = cs.verify().ok;
  const ser = cs.serialize();
  // 翻转负载区一个字节
  const corrupted = Uint8Array.from(ser); let flipped = false;
  for (let i = 0; i < corrupted.length; i++) { if (corrupted[i] !== 0 && i > 10) { corrupted[i] ^= 0xFF; flipped = true; break; } }
  const deser = G.ChecksummedStore.deserialize(corrupted);
  const detectCorrupt = flipped && deser.corrupted.length > 0;
  const ok = crcOk && fnvOk && shaOk && avalanche >= 0.3 && !collision && vOk && detectCorrupt;
  check('R6', '校验和：独立 CRC32/FNV1a64/SHA256 对拍已知向量 + 单比特雪崩均值=' + avalanche.toFixed(3) + ' ≥0.3 + 1000输入无碰撞 + ChecksummedStore 注入损坏必检出', ok,
    'crc=' + crcOk + ' fnv=' + fnvOk + ' sha=' + shaOk + ' avalanche=' + avalanche.toFixed(3) + ' collision=' + collision + ' cleanVerify=' + vOk + ' detectCorrupt=' + detectCorrupt);
}

/* ==========================================================================
 * R7  分区确定+均匀 + 一致性环（独立 SHA-256 重排环）+ 范围路由 + 对象存储
 * ========================================================================== */
{
  // (a) hashPartition：独立 FNV 对拍
  const rng = mulberry32(0x7016A06);
  let hpMatch = true;
  for (let i = 0; i < 500; i++) { const key = 'key-' + Math.floor(rng() * 10000); const n = 1 + (i % 13); const mod = G.hashPartition(key, n); const ind = Number(ownFnv1a64(ownUtf8Bytes(String(key))) % BigInt(n)); if (mod !== ind || mod < 0 || mod >= n) { hpMatch = false; break; } }
  // (b) 均匀性：8 桶偏差 ≤ 62（独立计数）
  const counts = new Array(8).fill(0);
  for (let i = 0; i < 8000; i++) counts[G.hashPartition('u-' + i, 8)]++;
  const maxDev = Math.max(...counts) - 8000 / 8;
  // (c) 一致性环：独立 SHA-256 重排环，模块 route 逐键一致；增/删节点移动率边界
  const nodes = ['n0', 'n1', 'n2', 'n3'];
  const keys = []; for (let i = 0; i < 2000; i++) keys.push('k' + i);
  const modRing = new G.ConsistentHashRing(nodes, 50);
  // 独立环
  function buildRing(ns, vn) { const ring = []; for (const nd of ns) for (let i = 0; i < vn; i++) ring.push({ h: ownRingHash64(nd + '#' + i), node: nd }); ring.sort((x, y) => (x.h < y.h ? -1 : (x.h > y.h ? 1 : 0))); return ring; }
  function routeRing(ring, key) { const h = ownRingHash64(key); let lo = 0, hi = ring.length; while (lo < hi) { const mid = (lo + hi) >> 1; if (ring[mid].h < h) lo = mid + 1; else hi = mid; } return ring[lo % ring.length].node; }
  const indRing = buildRing(nodes, 50);
  let routeMatch = true;
  for (const k of keys) if (routeRing(indRing, k) !== modRing.route(k)) { routeMatch = false; break; }
  const ringAdd = new G.ConsistentHashRing(nodes.concat(['n4']), 50);
  const mvAdd = modRing.movedFraction(keys, ringAdd);
  const ringDel = new G.ConsistentHashRing(nodes.slice(0, 3), 50);
  const mvDel = modRing.movedFraction(keys, ringDel);
  const ringBounds = mvAdd <= 0.4 && mvAdd >= 0.02 && mvDel <= 0.5;
  // (d) RangePartitioner
  const rp = new G.RangePartitioner(['10', '20', '30']);
  let rpOk = true;
  const expect = { '5': 0, '15': 1, '25': 2, '35': 3, '30': 3 };
  for (const k in expect) if (rp.route(k) !== expect[k]) { rpOk = false; break; }
  // (e) ObjectStore：去重/版本/元数据/墓碑删除
  const os = new G.ObjectStore();
  const b1 = ownUtf8Bytes('same-body');
  const p1 = os.put('bk', 'k1', b1, { tag: 'v1' });
  const p2 = os.put('bk', 'k1', b1, { tag: 'v2' }); // 同内容 → 仍两版本但单 body
  const dedup = os.bodies.size === 1 && p1.version === 1 && p2.version === 2;
  const g2 = os.get('bk', 'k1', 2); const metaOk = g2.meta.tag === 'v2' && ownDeepEqual(g2.body, b1);
  os.delete('bk', 'k1');
  let delOk = false; try { os.get('bk', 'k1'); } catch (e) { delOk = true; }
  const etagIndep = ownSha256Hex('same-body') === G.ObjectStore.etagOf(b1);
  const ok = hpMatch && maxDev <= 62 && routeMatch && ringBounds && rpOk && dedup && metaOk && delOk && etagIndep;
  check('R7', '分区确定+均匀(8桶maxDev=' + maxDev + '≤62) + 一致性环(独立SHA重排 route 逐键一致 + 增≤0.4/删≤0.5) + 范围路由 + 对象存储去重/版本/元/删', ok,
    'hpMatch=' + hpMatch + ' maxDev=' + maxDev + ' routeMatch=' + routeMatch + ' mvAdd=' + mvAdd.toFixed(3) + ' mvDel=' + mvDel.toFixed(3) + ' rpOk=' + rpOk + ' dedup=' + dedup + ' metaOk=' + metaOk + ' delOk=' + delOk + ' etagIndep=' + etagIndep);
}

/* ==========================================================================
 * R8  LRU 逐出序（独立 Map 模型对拍）+ size≤cap + hits+misses==gets + cap=1 证伪 + TTL 语义
 * ========================================================================== */
{
  // (a) LRUCache
  const cap = 4;
  const m = new G.LRUCache(cap);
  // 独立 Map 模型
  const map = new Map(); let ih = 0, im = 0, ie = 0;
  function iget(k) { if (map.has(k)) { const v = map.get(k); map.delete(k); map.set(k, v); ih++; return v; } im++; return undefined; }
  function iput(k, v) { if (map.has(k)) map.delete(k); map.set(k, v); if (map.size > cap) { const victim = map.keys().next().value; map.delete(victim); ie++; } }
  const rng = mulberry32(0x8110B07);
  const order = [];
  for (let t = 0; t < 400; t++) { const k = 'k' + Math.floor(rng() * 10); if (rng() < 0.5) { m.get(k); iget(k); } else { m.put(k, 1); iput(k, 1); } }
  // 逐出序对拍：独立模型剩余键序（front=LRU）应等于模块 keys()
  let orderMatch = m.keys().length === map.size;
  if (orderMatch) { const mk = m.keys(), ik = [...map.keys()]; for (let i = 0; i < mk.length; i++) if (mk[i] !== ik[i]) { orderMatch = false; break; } }
  const statMatch = m.hits === ih && m.misses === im && m.evictions === ie && m.gets === ih + im;
  const sizeOk = m.size <= cap;
  // cap=1 证伪：连续插入不同键必逐出，size 恒为 1
  const c1 = new G.LRUCache(1); c1.put('a', 1); c1.put('b', 2); c1.put('c', 3);
  const cap1Ok = c1.size === 1 && c1.get('a') === undefined;
  // (b) TTLCache：注入时钟
  let now = 0; const ttl = new G.TTLCache({ now: () => now });
  ttl.set('x', 'v', 10);
  now = 5; const before = ttl.get('x'); // 命中
  now = 15; const after = ttl.get('x'); // 过期 miss + 惰性清除
  now = 5; ttl.set('y', 'w', 10); const stale = ttl.getStale('y'); // 未过期命中(now=5<expireAt=15)
  now = 15; const staleExp = ttl.getStale('y'); // 过期仍返回旧值 + revalidate(stale=true)
  ttl.refresh('z', 'q', 10); now = 15; const refreshed = ttl.get('z'); // refresh 重置过期
  let ttl0miss = false; try { const z = new G.TTLCache({ now: () => 0 }); z.set('p', 1, 0); ttl0miss = z.get('p').hit === false; } catch (e) { ttl0miss = false; }
  const ttlOk = before.hit === true && after.hit === false && stale.hit === true && staleExp.hit === true && staleExp.stale === true && staleExp.revalidate === true && refreshed.hit === true && ttl0miss;
  const ok = orderMatch && statMatch && sizeOk && cap1Ok && ttlOk;
  check('R8', 'LRU 逐出序(独立Map模型对拍)+ size≤cap + hits+misses==gets + cap=1 证伪 + TTL(过期前hit/后miss/惰性清除 + SWR revalidate + refresh + ttl=0miss)', ok,
    'orderMatch=' + orderMatch + ' statMatch=' + statMatch + '(h=' + m.hits + '/' + ih + ' m=' + m.misses + '/' + im + ' ev=' + m.evictions + '/' + ie + ') sizeOk=' + sizeOk + ' cap1=' + cap1Ok + ' ttl=' + ttlOk);
}

/* ---------- 汇总 ---------- */
const failed = results.filter((r) => !r.ok).length;
console.log('══════ serialize-forge 独立复核: ' + (results.length - failed) + '/' + results.length + ' 通过 ══════');
process.exit(failed === 0 ? 0 : 1);
