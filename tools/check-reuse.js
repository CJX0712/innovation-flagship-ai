#!/usr/bin/env node
/**
 * tools/check-reuse.js — 复用合规校验（零重写 · 集大成）
 *
 * 校验目标（纯 Node，零外部依赖）：
 *   1. 扫描 src/*.js 中匹配 /(-forge|core-numeric|core-verify)\.js$/ 的模块；
 *   2. 解析每个模块的 @reuse 头部行，格式：@reuse NAME|URL|LICENSE|GRADE|note
 *      GRADE ∈ {A,B,C,D,X}，其中 X = 自研 / 无需外部复用；
 *   3. 每个模块【必须】声明至少一条 @reuse（零重写·集大成原则）；
 *   4. 与 docs/L10-index.json 交叉比对：任何标记为 status="done" 的 L10 微指令
 *      都必须存在对应的实现模块文件（一致性 / 孤儿检测）；
 *   5. 列出 @reuse 评级为 X 但仍可引用外部顶级实现的模块（仅 ADVISORY）。
 *
 * 退出码：0 = 无 ERROR 级问题；1 = 存在缺失 @reuse 或 done 孤儿。
 * 作者：晨星 · MIT
 */
'use strict';

const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const SRC_DIR = path.join(ROOT, 'src');
const IDX = path.join(ROOT, 'docs', 'L10-index.json');

const MODULE_RE = /(-forge|core-numeric|core-verify)\.js$/;
const REUSE_RE = /^\s*\*\s*@reuse\s+(.+?)\s*$/;
const GRADES = new Set(['A', 'B', 'C', 'D', 'X']);

/** 左对齐填充到指定显示宽度（按码点计数，CJK 近似按 2 宽度）。 */
function pad(s, n) {
  s = String(s);
  let w = 0;
  for (const ch of s) w += ch.charCodeAt(0) > 0x2e7f ? 2 : 1; // 粗略 CJK 双宽
  if (w >= n) return s;
  return s + ' '.repeat(n - w);
}

function main() {
  const errors = [];   // ERROR 级：缺失 @reuse / done 孤儿 / 非法 GRADE
  const advisories = []; // ADVISORY 级：X 评级模块提示

  if (!fs.existsSync(SRC_DIR)) {
    console.error('未找到 src/ 目录：' + SRC_DIR);
    process.exit(1);
  }

  // 1) 发现模块
  const files = fs.readdirSync(SRC_DIR).filter((f) => MODULE_RE.test(f)).sort();
  if (!files.length) {
    console.error('未在 src/ 发现匹配模块（' + MODULE_RE + '）');
    process.exit(1);
  }

  // 2) 解析每个模块的 @reuse 行
  const modules = [];
  for (const f of files) {
    const full = path.join(SRC_DIR, f);
    const src = fs.readFileSync(full, 'utf8');
    const reuses = [];
    for (const line of src.split(/\r?\n/)) {
      const m = REUSE_RE.exec(line);
      if (!m) continue;
      const parts = m[1].split('|');
      // 评级 GRADE 容错解析：在任意字段中匹配 A/B/C/D/X，
      // 兼容既有「缺 LICENSE」的 4 字段写法（NAME|URL|GRADE|note）。
      let grade = '';
      for (const p of parts) {
        const t = p.trim().toUpperCase();
        if (GRADES.has(t)) { grade = t; break; }
      }
      reuses.push({
        name: (parts[0] || '').trim(),
        url: (parts[1] || '').trim(),
        license: (parts[2] || '').trim(),
        grade: grade,
        note: (parts[parts.length - 1] || '').trim(),
        raw: m[1],
        fields: parts.length,
      });
    }
    modules.push({ file: f, module: f.replace(/\.js$/, ''), reuses });
  }

  // 3) 逐模块校验 @reuse 声明与 GRADE 合法性
  const rows = [];
  for (const mod of modules) {
    const grades = mod.reuses.map((r) => r.grade).filter(Boolean);
    let status = 'OK';
    if (mod.reuses.length === 0) {
      status = 'ERROR·缺失@reuse';
      errors.push('模块 ' + mod.file + ' 未声明任何 @reuse 行（违反「零重写·集大成」原则：每模块须引用世界顶级外部参照或标 X）');
    } else {
      for (const r of mod.reuses) {
        // 评级缺失 / X 评级 / 字段数非 5 均属 ADVISORY（不影响退出码）
        if (!r.grade) {
          advisories.push('模块 ' + mod.file + ' 的 @reuse 未找到合法评级（应为 A/B/C/D/X）：' + r.raw);
        }
        if (r.grade === 'X') {
          advisories.push('模块 ' + mod.file + ' 的 @reuse "' + r.name + '" 评级为 X（自研/无需外部复用）。若该能力存在世界顶级外部参照，建议改标 A/B/C/D 以体现集大成。');
        }
        if (r.fields !== 5) {
          advisories.push('模块 ' + mod.file + ' 的 @reuse 字段数为 ' + r.fields + '（规范应为 5：NAME|URL|LICENSE|GRADE|note）：' + r.raw);
        }
      }
    }
    rows.push({ file: mod.file, count: mod.reuses.length, grades: grades.join(',') || '-', status });
  }

  // 4) 与 L10 索引交叉比对：done 微指令须有实现模块
  let doneTotal = 0;
  let orphanCount = 0;
  if (!fs.existsSync(IDX)) {
    errors.push('未找到 ' + path.relative(ROOT, IDX) + '，请先运行 node tools/gen-l10.js');
  } else {
    const idx = JSON.parse(fs.readFileSync(IDX, 'utf8'));
    doneTotal = idx.by_status && idx.by_status.done ? idx.by_status.done : 0;
    const orphans = [];
    for (const leaf of idx.leaves || []) {
      if (leaf.status !== 'done') continue;
      const modName = String(leaf.reuse || '').trim().split(/\s+/)[0];
      const exists = !!modName && fs.existsSync(path.join(SRC_DIR, modName + '.js'));
      if (!exists) orphans.push(leaf.id + ' → 实现模块 "' + (modName || '(空)') + '" 不存在');
    }
    orphanCount = orphans.length;
    if (orphanCount) {
      errors.push('发现 ' + orphanCount + ' 个标记为 done 但无实现模块的 L10 微指令（孤儿）：');
      orphans.forEach((o) => errors.push('   - ' + o));
    }
  }

  // 5) 输出报告
  const line = '---------------------------------------------------------------';
  console.log('');
  console.log('=== 复用合规校验 (check-reuse) ===');
  console.log('');
  console.log('[1] 模块 @reuse 声明检查');
  console.log('  ' + pad('模块文件', 26) + pad('@reuse数', 10) + pad('评级', 12) + '状态');
  console.log('  ' + line);
  for (const r of rows) {
    console.log('  ' + pad(r.file, 26) + pad(r.count, 10) + pad(r.grades, 12) + r.status);
  }

  console.log('');
  console.log('[2] L10 一致性检查（docs/L10-index.json）');
  console.log('  L10 done 总数: ' + doneTotal);
  console.log('  done 孤儿（无实现模块）: ' + orphanCount);

  if (advisories.length) {
    console.log('');
    console.log('[3] ADVISORY 提示（不影响退出码）');
    advisories.forEach((a) => console.log('  • ' + a));
  }

  console.log('');
  console.log('--- 汇总 ---');
  console.log('  扫描模块数 : ' + modules.length);
  console.log('  ERROR 级   : ' + errors.length);
  console.log('  ADVISORY 级: ' + advisories.length);

  if (errors.length) {
    console.log('');
    console.log('❌ 校验失败（' + errors.length + ' 项 ERROR）：');
    errors.forEach((e) => console.log('   - ' + e));
    process.exit(1);
  }
  console.log('');
  console.log('✅ 复用合规校验通过：所有模块均声明 @reuse，且全部 done 微指令均有实现模块。');
  process.exit(0);
}

main();
