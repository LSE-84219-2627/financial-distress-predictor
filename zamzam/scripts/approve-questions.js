#!/usr/bin/env node
// Moves approved question IDs from questions.pending.json into questions.json
// then re-generates schedule.json
//
// Usage: node scripts/approve-questions.js q-pending-001 q-pending-003

import { readFileSync, writeFileSync, existsSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';
import { execSync } from 'child_process';

const __dir = dirname(fileURLToPath(import.meta.url));
const qPath    = join(__dir, '../data/questions.json');
const pendPath = join(__dir, '../data/questions.pending.json');

const ids = process.argv.slice(2);
if (!ids.length) { console.error('Usage: approve-questions.js <id> [<id>...]'); process.exit(1); }

if (!existsSync(pendPath)) { console.error('No questions.pending.json found.'); process.exit(1); }

const { questions } = JSON.parse(readFileSync(qPath, 'utf8'));
const pending = JSON.parse(readFileSync(pendPath, 'utf8'));

const toApprove = pending.filter(q => ids.includes(q.id));
const remaining = pending.filter(q => !ids.includes(q.id));

if (!toApprove.length) {
  console.error('None of the given IDs found in pending:', ids);
  process.exit(1);
}

// Assign real IDs (q34, q35, …)
const maxNum = questions.reduce((m, q) => {
  const n = parseInt(q.id.replace(/\D/g, ''), 10);
  return isNaN(n) ? m : Math.max(m, n);
}, 0);

toApprove.forEach((q, i) => { q.id = `q${maxNum + i + 1}`; });

questions.push(...toApprove);
writeFileSync(qPath, JSON.stringify({ questions }, null, 2));
writeFileSync(pendPath, JSON.stringify(remaining, null, 2));

console.log(`✅ Approved ${toApprove.length} question(s):`, toApprove.map(q => q.id));
console.log('Regenerating schedule…');
execSync('node scripts/generate-schedule.js', { stdio: 'inherit', cwd: join(__dir, '..') });
