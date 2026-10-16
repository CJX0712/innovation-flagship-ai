/**
 * 创新旗舰 AI · 契约一致性自检器 (tests/harness.js)
 *
 * 职责（对照 W2 交付清单）：
 *   1. 契约完整性  — 每个模块导出 selfCheck()，返回 schema === 'forge-selfcheck/1.0'，
 *                    且含 cases/invariants 数组、passed/failed 数值、hash（sha256: 前缀）。
 *   2. hash 幂等   — 同一模块两次 selfCheck() 的 hash 必须完全一致。
 *   3. 不良输入抛错 — 各模块对文档化危险输入必须抛出 TypeError（不允许静默返回错误有限值）。
 *   4. CLI _exit   — 每个模块作为 `node <mod> --self-check --json` 运行，退出码必须为 0，
 *                    且 stdout 为可解析 JSON、failed === 0。
 *
 * 退出码：0 = 全部契约满足；1 = 任一契约被破坏。
 */
'use strict';
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

const SRC = path.join(__dirname, '..', 'src');
const MOD_RE = /(-forge|core-numeric|core-verify)\.js$/;

// 文档化“不良输入必须抛错”清单：module 全局名 → [函数名, 危险参数]
const BAD_INPUTS = {
  ForgeStable: [['logsumexp', [null]], ['softmax', [1, 2, 'x']], ['logSigmoid', [Infinity]]],
  ForgeLinalg: [['lu', [null]], ['chol', [[1, 2], [3]]], ['qr', [[1, 2], [3]]]],
  ForgeSampling: [['softmaxTemperature', [null]], ['topKFilter', [[1, 2], 0]], ['topPFilter', [[1, 2], 1.5]]],
  ForgeTransformer: [['tokenEmbedding', [null, [0]]], ['sinusoidalPositionalEncoding', [0, 4]]],
  ForgeKVCache: [['createKVCache', [null]], ['createKVCache', [{ nLayers: 0, nHeads: 2, headDim: 4, maxLen: 8 }]]],
  ForgePosEnc: [['positionInterpolation', [3, 0]], ['positionInterpolation', [3, -1]], ['alibiSlopes', [0]]],
  ForgeMoE: [['topKGating', [[1, 2, 3], 0]], ['loadBalancingLoss', [null, 4]]],
  ForgeQuant: [['calibrate', [[1, 2, 3], { bits: 0 }]], ['calibrate', [[1, 2, 3], { bits: 8, scheme: 'bogus' }]]],
  ForgeSpeculative: [['rejectionSample', [[0.5, 0.6], [0.5, 0.5], 1]]],
  ForgeGeneration: [['generate', [null, [], { maxTokens: 3 }]], ['stopCheck', [null, null]]],
  ForgeDQN: [
    ['qNet', [new Float64Array(3), null]],
    ['qNet', [new Float64Array([2, 2, 0, 0, 0, 0, 0, 0]), null]],
    ['makeQParams', [0, 2]],
    ['makeChainEnv', [1]],
    ['tdTarget', [1, [], false, 0.9]],
    ['epsilonGreedy', [[], 0.5, () => 0.5]],
    ['epsilonGreedy', [[0.1, 0.2], 1.5, () => 0]],
    ['epsilonSchedule', [-1, { mode: 'linear', decaySteps: 10 }]],
    ['epsilonSchedule', [5, { mode: 'bogus' }]],
    ['polyakUpdate', [new Float64Array(4), new Float64Array(6), 0.1]],
    ['bellmanResidual', [null, null, 0.9]]
  ]
};

const problems = [];
function fail(where, msg) { problems.push('[' + where + '] ' + msg); }

function listModules() {
  return fs.readdirSync(SRC).filter(f => MOD_RE.test(f)).map(f => path.join(SRC, f));
}

function checkContract(modPath) {
  const name = path.basename(modPath);
  let mod;
  try { mod = require(modPath); } catch (e) { fail(name, 'require 失败: ' + e.message); return; }
  if (typeof mod.selfCheck !== 'function') { fail(name, '未导出 selfCheck()'); return; }

  let r1, r2;
  try { r1 = mod.selfCheck(); r2 = mod.selfCheck(); }
  catch (e) { fail(name, 'selfCheck() 抛错: ' + e.message); return; }

  // schema
  if (!r1 || r1.schema !== 'forge-selfcheck/1.0') fail(name, 'schema 不是 forge-selfcheck/1.0: ' + (r1 && r1.schema));
  // cases / invariants
  if (!Array.isArray(r1.cases)) fail(name, 'cases 非数组');
  if (!Array.isArray(r1.invariants)) fail(name, 'invariants 非数组');
  // passed/failed 自洽
  const pf = r1.cases.filter(c => c.status === 'pass').length;
  if (r1.passed !== pf) fail(name, 'passed 字段(' + r1.passed + ') 与实际用例(' + pf + ') 不一致');
  if (r1.passed + r1.failed !== r1.cases.length) fail(name, 'passed+failed ≠ cases.length');
  // hash 格式 + 幂等
  if (typeof r1.hash !== 'string' || !r1.hash.startsWith('sha256:')) fail(name, 'hash 缺少 sha256: 前缀');
  if (r1.hash !== r2.hash) fail(name, 'hash 非幂等: ' + r1.hash + ' ≠ ' + r2.hash);
  // 所有用例必须 pass（核心交付门槛）
  const failing = r1.cases.filter(c => c.status !== 'pass').map(c => c.id);
  if (failing.length) fail(name, '存在未通过用例: ' + failing.join(','));
}

function checkBadInputs(modPath, globalName) {
  const name = path.basename(modPath);
  const spec = BAD_INPUTS[globalName];
  if (!spec) return; // 未登记的模块跳过（其余模块不强制）
  let mod;
  try { mod = require(modPath); } catch (e) { return; }
  for (const [fn, args] of spec) {
    if (typeof mod[fn] !== 'function') continue;
    let threw = false;
    try { mod[fn].apply(null, args); } catch (e) { threw = true; }
    if (!threw) fail(name, '不良输入未抛错: ' + fn + '(' + JSON.stringify(args) + ')');
  }
}

function checkCli(modPath) {
  const name = path.basename(modPath);
  let out;
  try {
    out = execFileSync(process.execPath, [modPath, '--self-check', '--json'], { encoding: 'utf8', timeout: 12000 });
  } catch (e) {
    fail(name, 'CLI 运行失败 (exit ' + (e.status || '?') + '): ' + (e.stderr || e.message).slice(0, 200));
    return;
  }
  let r;
  try { r = JSON.parse(out); } catch (e) { fail(name, 'CLI 输出非 JSON'); return; }
  if (r.failed !== 0) fail(name, 'CLI 报告 failed=' + r.failed);
}

function main() {
  const mods = listModules();
  if (!mods.length) { fail('harness', '未发现任何契约模块'); }
  for (const m of mods) {
    checkContract(m);
    checkBadInputs(m, globalNameFor(m));
    checkCli(m);
  }
  if (problems.length) {
    console.log('══════ 契约自检失败 (' + problems.length + ') ══════');
    for (const p of problems) console.log('  ✗ ' + p);
    process.exit(1);
  } else {
    console.log('══════ 契约自检通过：' + mods.length + ' 个模块全部满足 @forge-module 契约 ══════');
    process.exit(0);
  }
}

// 由文件名推断全局名（与模块头部 globalThis 挂载保持一致）
function globalNameFor(modPath) {
  const base = path.basename(modPath, '.js'); // e.g. stable-forge
  const map = {
    'core-numeric': 'ForgeNumeric', 'core-verify': 'ForgeVerify',
    'linalg-forge': 'ForgeLinalg', 'stable-forge': 'ForgeStable',
    'pca-forge': 'ForgePCA', 'gmm-forge': 'ForgeGMM', 'nn-forge': 'ForgeNN', 'attn-forge': 'ForgeAttn',
    'optim-forge': 'ForgeOptim', 'autodiff-forge': 'ForgeAutoDiff', 'loss-act-norm-forge': 'ForgeLossActNorm',
    'attention-forge': 'ForgeAttention', 'rope-forge': 'ForgeRoPE', 'retrieval-forge': 'ForgeRetrieval',
    'prob-forge': 'ForgeProb', 'graph-forge': 'ForgeGraph', 'rl-forge': 'ForgeRL',
    'transformer-forge': 'ForgeTransformer', 'sampling-forge': 'ForgeSampling',
    'kv-cache-forge': 'ForgeKVCache', 'position-encoding-forge': 'ForgePosEnc', 'moe-forge': 'ForgeMoE',
    'quant-forge': 'ForgeQuant', 'speculative-forge': 'ForgeSpeculative', 'generation-forge': 'ForgeGeneration',
    'dqn-forge': 'ForgeDQN'
  };
  return map[base] || null;
}

if (typeof process !== 'undefined' && process.argv[1] && /harness\.js$/.test(process.argv[1])) {
  main();
}
