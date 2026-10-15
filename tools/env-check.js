/**
 * tools/env-check.js — 环境诊断（W1 基础设施）
 * 检测 Node 版本、Git 可用性、目录结构、已存在模块清单。输出中文诊断表 + 可执行修复建议。
 * 退出码 0 可继续 / 66 环境不满足。
 */
'use strict';
const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const ROOT = __dirname.replace(/[\\/]tools$/, '');
const SRC = path.join(ROOT, 'src');

function check(label, ok, detail, fix) {
  return { label, ok, detail: detail || '', fix: fix || '' };
}

function main() {
  const rows = [];
  // 1. Node 版本
  const nodeVer = process.version;
  const major = parseInt(nodeVer.replace(/^v/, '').split('.')[0], 10);
  rows.push(check('Node 版本 >= 18', major >= 18, nodeVer, major >= 18 ? '' : '请升级 Node 到 18 以上（nvm install 18）'));
  // 2. Git
  let gitOk = false, gitDetail = '';
  try { gitDetail = execSync('git --version', { encoding: 'utf8' }).trim(); gitOk = true; } catch (e) { gitDetail = '未安装 git'; }
  rows.push(check('Git 可用', gitOk, gitDetail, gitOk ? '' : '安装 git 并加入 PATH'));
  // 3. 目录结构
  const dirs = ['src', 'docs', 'tools', 'tests'];
  const missing = dirs.filter(d => !fs.existsSync(path.join(ROOT, d)));
  rows.push(check('目录结构完整', missing.length === 0, missing.length === 0 ? 'src/docs/tools/tests 均存在' : '缺失: ' + missing.join(','), missing.length === 0 ? '' : 'mkdir -p ' + missing.map(d => path.join(ROOT, d)).join(' ')));
  // 4. 模块清单
  const present = fs.existsSync(SRC) ? fs.readdirSync(SRC).filter(f => /(-forge|core-numeric|core-verify)\.js$/.test(f)).sort() : [];
  rows.push(check('核心模块存在', present.length >= 8, '已发现 ' + present.length + ' 个 *-forge.js', present.length >= 8 ? '' : '缺少部分核心模块（core-numeric/core-verify/linalg/stable 等）'));
  // 5. LICENSE
  rows.push(check('LICENSE(MIT) 存在', fs.existsSync(path.join(ROOT, 'LICENSE')), fs.existsSync(path.join(ROOT, 'LICENSE')) ? '存在' : '缺失', '需放置 MIT LICENSE，署名 晨星/CJX0712'));

  const allOk = rows.every(r => r.ok);
  process.stdout.write('\n════════ 环境诊断 ════════\n');
  for (const r of rows) {
    process.stdout.write('  ' + (r.ok ? '[OK]  ' : '[WARN]') + r.label + ' — ' + r.detail + '\n');
    if (!r.ok && r.fix) process.stdout.write('         ↳ 修复: ' + r.fix + '\n');
  }
  process.stdout.write('──────── ' + (allOk ? '✅ 环境就绪，可继续' : '⚠️ 见上修复项') + ' ────────\n\n');
  process.exit(allOk ? 0 : 66);
}
main();
