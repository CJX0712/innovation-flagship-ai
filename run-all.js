/**
 * run-all.js — 无头聚合自检运行器（W1 基础设施）
 * 扫描 src/*-forge.js，对每个 spawn `node <mod>.js --self-check --json`，
 * 解析 JSON 报告，输出聚合表格，写 run-all-report.json，退出码 0/1。
 * 全局 110s 熔断（PRD S3 ≤120s），单模块软 5s / 硬超时 10s。
 */
'use strict';
const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const SRC = path.join(__dirname, 'src');
const GLOBAL_TIMEOUT_MS = 110000;
const MODULE_TIMEOUT_MS = 10000;

function listModules() {
  const files = fs.readdirSync(SRC).filter(f => /(-forge|core-numeric|core-verify)\.js$/.test(f));
  return files.sort();
}

function runModule(file) {
  const p = spawnSync(process.argv[0], [path.join(SRC, file), '--self-check', '--json'], {
    encoding: 'utf8', timeout: MODULE_TIMEOUT_MS, maxBuffer: 1 << 24
  });
  const result = { file, exitCode: p.status, passed: 0, failed: 0, hash: '', error: '' };
  if (p.error) { result.error = p.error.message; return result; }
  if (p.status === null) { result.error = '超时(>' + MODULE_TIMEOUT_MS + 'ms)'; return result; }
  try {
    const json = JSON.parse(p.stdout);
    result.passed = json.passed || 0;
    result.failed = json.failed || 0;
    result.hash = (json.hash || '').slice(0, 24);
  } catch (e) {
    result.error = 'stdout 非 JSON: ' + (p.stdout || '').slice(0, 120);
  }
  return result;
}

function main() {
  const only = process.argv.includes('--only=') ? process.argv.find(a => a.startsWith('--only=')).split('=')[1] : null;
  const jsonOnly = process.argv.includes('--json');
  const start = Date.now();
  let files = listModules();
  if (only) files = files.filter(f => f.indexOf(only) >= 0);

  const results = [];
  let totalPass = 0, totalFail = 0;
  for (const f of files) {
    if (Date.now() - start > GLOBAL_TIMEOUT_MS) {
      results.push({ file: f, exitCode: 66, passed: 0, failed: 0, hash: '', error: '全局熔断' });
      continue;
    }
    const r = runModule(f);
    results.push(r);
    totalPass += r.passed; totalFail += r.failed;
  }

  const report = {
    started_at: new Date(start).toISOString(),
    duration_ms: Date.now() - start,
    modules: results.length,
    total_passed: totalPass,
    total_failed: totalFail,
    results
  };
  fs.writeFileSync(path.join(__dirname, 'run-all-report.json'), JSON.stringify(report, null, 2));

  if (jsonOnly) {
    process.stdout.write(JSON.stringify(report, null, 2) + '\n');
  } else {
    const lines = [];
    lines.push('════════ 创新旗舰 AI · 聚合自检 ════════');
    lines.push('模块数 ' + results.length + ' · 通过 ' + totalPass + ' · 失败 ' + totalFail + ' · 耗时 ' + (Date.now() - start) + 'ms');
    for (const r of results) {
      const status = r.failed === 0 && !r.error ? 'PASS' : 'FAIL';
      const detail = r.error ? ('错误: ' + r.error) : ('+' + r.passed + '/-' + r.failed);
      lines.push('  [' + status + '] ' + r.file + '  ' + detail + '  ' + r.hash);
    }
    const allPass = totalFail === 0 && results.every(r => !r.error);
    lines.push('──────── ' + (allPass ? '✅ 全部通过' : '❌ 存在失败') + ' ────────');
    process.stdout.write(lines.join('\n') + '\n');
  }
  process.exit(totalFail === 0 && results.every(r => !r.error) ? 0 : 1);
}
main();
