/**
 * tools/verify-l10.js — L10 分类法一致性校验
 * 校验：总数=1024、族=8、组=64、每组16 op、ID 唯一、字段完整、status∈{done,planned}。
 * 退出码 0=通过 / 1=失败。作者：晨星 · MIT
 */
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const IDX = path.join(ROOT, 'docs', 'L10-index.json');

function main() {
  if (!fs.existsSync(IDX)) { console.error('未找到 ' + IDX + '，请先运行 node tools/gen-l10.js'); process.exit(1); }
  const idx = JSON.parse(fs.readFileSync(IDX, 'utf8'));
  const errs = [];

  if (idx.total !== 1024) errs.push('总数应为 1024，实际 ' + idx.total);
  if (idx.families !== 8) errs.push('族应为 8，实际 ' + idx.families);
  if (idx.groups !== 64) errs.push('组应为 64，实际 ' + idx.groups);
  if (idx.ops_per_group !== 16) errs.push('每组 op 应为 16，实际 ' + idx.ops_per_group);

  const ids = new Set();
  const famCount = {}, grpCount = {};
  for (const l of idx.leaves) {
    if (ids.has(l.id)) errs.push('重复 ID：' + l.id);
    ids.add(l.id);
    famCount[l.family] = (famCount[l.family] || 0) + 1;
    grpCount[l.family + '/' + l.group] = (grpCount[l.family + '/' + l.group] || 0) + 1;
    if (!l.name || !l.layer || !l.status || !l.reuse) errs.push('字段缺失：' + l.id);
    if (l.status !== 'done' && l.status !== 'planned') errs.push('非法 status：' + l.id + '=' + l.status);
    if (!/^L10-F[0-7]-G[0-7]-O[0-9][0-9]$/.test(l.id)) errs.push('ID 格式非法：' + l.id);
  }
  for (let f = 0; f < 8; f++) { const k = 'F' + f; if ((famCount[k] || 0) !== 128) errs.push('族 ' + k + ' 应为 128 条，实际 ' + (famCount[k] || 0)); }
  for (let f = 0; f < 8; f++) for (let g = 0; g < 8; g++) { const k = 'F' + f + '/G' + g; if ((grpCount[k] || 0) !== 16) errs.push('组 ' + k + ' 应为 16 条，实际 ' + (grpCount[k] || 0)); }

  if (errs.length) {
    console.error('❌ L10 校验失败（' + errs.length + ' 项）：');
    errs.slice(0, 30).forEach(e => console.error('   - ' + e));
    if (errs.length > 30) console.error('   ... 其余 ' + (errs.length - 30) + ' 项省略');
    process.exit(1);
  }
  console.log('✅ L10 校验通过：' + idx.total + ' 项 / 8 族 / 64 组 / 每组 16 op / ID 唯一 / 字段完整');
  console.log('   状态分布：' + JSON.stringify(idx.by_status));
  process.exit(0);
}
main();
