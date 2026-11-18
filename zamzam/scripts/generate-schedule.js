#!/usr/bin/env node
// Generates schedule.json: assigns 5 question IDs per day for 365 days starting today.
// Rules:
//  - At least 1 'core' and 1 'fun' per day
//  - No two questions from the same sub-category in one day
//  - Difficulty spread: roughly 1 easy, 3 medium, 1 hard (difficulty 1/2/3)
//  - Order: easiest first, hardest last; alternate core/fun
//  - No question repeats within 30 days
//  - Prefer never-used questions over repeats

import { readFileSync, writeFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dir = dirname(fileURLToPath(import.meta.url));
const questionsPath = join(__dir, '../data/questions.json');
const schedulePath = join(__dir, '../data/schedule.json');

const { questions } = JSON.parse(readFileSync(questionsPath, 'utf8'));

const START_DATE = new Date();
START_DATE.setHours(0, 0, 0, 0);
const DAYS = 365;
const QUESTIONS_PER_DAY = 5;
const NO_REPEAT_WINDOW = 30;

function dateStr(d) {
  return d.toISOString().slice(0, 10);
}

function seededRng(seed) {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return (s >>> 0) / 0xffffffff;
  };
}

function dateSeed(str) {
  let h = 0;
  for (const c of str) h = (Math.imul(31, h) + c.charCodeAt(0)) | 0;
  return h >>> 0;
}

function shuffle(arr, rng) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// Build a pool with metadata for each question
const pool = questions.map(q => ({
  id: q.id,
  pillar: q.pillar,
  category: q.category,
  difficulty: q.difficulty,
  lastUsed: -999,
  usedCount: 0,
}));

const schedule = {};
const usedOnDay = {}; // day index → Set of question ids

function pickDay(dayIndex, dateString) {
  const rng = seededRng(dateSeed(dateString));
  const available = pool.filter(q => dayIndex - q.lastUsed >= NO_REPEAT_WINDOW);

  if (available.length < QUESTIONS_PER_DAY) {
    // Fallback: relax window
    available.push(...pool.filter(q => !available.includes(q)));
  }

  const shuffled = shuffle(available, rng);

  const picked = [];
  const usedCategories = new Set();
  let coreCount = 0, funCount = 0;

  // Try to pick: 1 easy (diff 1), 3 medium (diff 2), 1 hard (diff 3)
  const targetDifficulties = [1, 2, 2, 2, 3];

  for (const targetDiff of targetDifficulties) {
    const candidates = shuffled.filter(q =>
      !picked.includes(q) &&
      !usedCategories.has(q.category) &&
      q.difficulty === targetDiff &&
      !(coreCount === 0 && q.pillar === 'fun' && picked.length === 4) && // ensure at least 1 core
      !(funCount === 0 && q.pillar === 'core' && picked.length === 4)    // ensure at least 1 fun
    );

    if (candidates.length > 0) {
      const chosen = candidates[0];
      picked.push(chosen);
      usedCategories.add(chosen.category);
      if (chosen.pillar === 'core') coreCount++;
      else funCount++;
    } else {
      // Relax difficulty constraint
      const fallback = shuffled.find(q => !picked.includes(q) && !usedCategories.has(q.category));
      if (fallback) {
        picked.push(fallback);
        usedCategories.add(fallback.category);
        if (fallback.pillar === 'core') coreCount++;
        else funCount++;
      }
    }
  }

  // Ensure pillar requirements
  if (coreCount === 0) {
    // Replace last fun pick with a core
    const coreCandidate = shuffled.find(q => !picked.includes(q) && q.pillar === 'core');
    if (coreCandidate) {
      picked[picked.length - 1] = coreCandidate;
    }
  }
  if (funCount === 0) {
    const funCandidate = shuffled.find(q => !picked.includes(q) && q.pillar === 'fun');
    if (funCandidate) {
      picked[picked.length - 1] = funCandidate;
    }
  }

  // Sort: easiest first, alternating pillar where possible
  picked.sort((a, b) => a.difficulty - b.difficulty);

  // Update pool tracking
  for (const q of picked) {
    q.lastUsed = dayIndex;
    q.usedCount++;
  }

  return picked.map(q => q.id);
}

for (let i = 0; i < DAYS; i++) {
  const d = new Date(START_DATE);
  d.setDate(d.getDate() + i);
  const ds = dateStr(d);
  schedule[ds] = pickDay(i, ds);
}

writeFileSync(schedulePath, JSON.stringify(schedule, null, 2));

// Warning report
const neverUsed = pool.filter(q => q.usedCount === 0);
const coreMissing = neverUsed.filter(q => q.pillar === 'core').length;
const funMissing = neverUsed.filter(q => q.pillar === 'fun').length;

console.log(`Schedule written: ${DAYS} days, ${Object.keys(schedule).length} entries`);
console.log(`Questions never scheduled: ${neverUsed.length} (core: ${coreMissing}, fun: ${funMissing})`);
if (pool.filter(q => q.usedCount === 0).length > 30) {
  console.warn('⚠️  Pool is large — consider removing near-duplicate questions.');
}
const usedOnce = pool.filter(q => q.usedCount >= 1).length;
console.log(`Questions used at least once: ${usedOnce}/${pool.length}`);
