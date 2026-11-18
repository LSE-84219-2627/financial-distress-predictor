import { matchAnswer } from './matcher.js';

// ─── Constants ────────────────────────────────────────────────────────────────
const TIER_WATER = { Common: 5, Uncommon: 10, Rare: 15, Legendary: 20 };
const TIER_EMOJI = { Common: '🟢', Uncommon: '🔵', Rare: '🟣', Legendary: '🟡', miss: '⬛' };
const TIER_CLASS = { Common: 'tier-common', Uncommon: 'tier-uncommon', Rare: 'tier-rare', Legendary: 'tier-legendary' };
const TIMER_SECONDS = 25;

// ─── State ────────────────────────────────────────────────────────────────────
let questions = [];
let schedule = {};
let state = null;
let stats = loadStats();

// ─── Timer ────────────────────────────────────────────────────────────────────
let timerInterval = null;
let timerRemaining = 0;

function startTimer() {
  stopTimer();
  timerRemaining = TIMER_SECONDS;
  updateTimerDisplay();
  timerInterval = setInterval(() => {
    timerRemaining--;
    updateTimerDisplay();
    if (timerRemaining <= 0) {
      stopTimer();
      onTimerExpire();
    }
  }, 1000);
}

function stopTimer() {
  if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
}

function updateTimerDisplay() {
  const el = $('timer-count');
  const bar = $('timer-bar');
  if (!el) return;
  el.textContent = timerRemaining;
  el.classList.toggle('urgent', timerRemaining <= 5);
  if (bar) {
    bar.style.width = (timerRemaining / TIMER_SECONDS * 100) + '%';
    bar.classList.toggle('urgent', timerRemaining <= 5);
  }
}

function onTimerExpire() {
  if (!state || state.done || $('guess-input').disabled) return;
  state.answers[state.currentQ] = { tier: null, correct: false, guess: null };
  const q = getQuestionById(state.questionIds[state.currentQ]);
  advanceQuestion();
  $('feedback').textContent = `⏱️ Time's up!`;
  $('feedback').className = 'feedback miss';
  showReveal(q, true);
  if (state.done) setTimeout(renderEndScreen, 1500);
}

// ─── Persistence ──────────────────────────────────────────────────────────────
function loadStats() {
  try {
    return JSON.parse(localStorage.getItem('deenly_stats') || 'null') || {
      gamesPlayed: 0, perfectFills: 0, legendaryCount: 0, currentStreak: 0, lastPlayed: null
    };
  } catch { return { gamesPlayed: 0, perfectFills: 0, legendaryCount: 0, currentStreak: 0, lastPlayed: null }; }
}

function saveStats() {
  try { localStorage.setItem('deenly_stats', JSON.stringify(stats)); } catch {}
}

function loadGameState(key) {
  try { return JSON.parse(localStorage.getItem(key) || 'null'); } catch { return null; }
}

function saveGameState(key) {
  try { localStorage.setItem(key, JSON.stringify(state)); } catch {}
}

// ─── Data loading ─────────────────────────────────────────────────────────────
async function loadData() {
  const [qRes, sRes] = await Promise.all([
    fetch('./data/questions.json'),
    fetch('./data/schedule.json')
  ]);
  const qData = await qRes.json();
  questions = qData.questions;
  schedule = await sRes.json();
}

// ─── Question selection ───────────────────────────────────────────────────────
function todayString() {
  return new Date().toISOString().slice(0, 10);
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

function getDailyIds(dateStr) {
  if (schedule[dateStr]) return schedule[dateStr];
  // Fallback: seed-based selection (shouldn't happen if schedule is current)
  const rng = seededRng(dateSeed(dateStr));
  return shuffle(questions.map(q => q.id), rng).slice(0, 5);
}

function getPracticeIds() {
  const rng = seededRng(Date.now());
  return shuffle(questions.map(q => q.id), rng).slice(0, 5);
}

function getQuestionById(id) {
  return questions.find(q => q.id === id) || null;
}

// ─── Game init ────────────────────────────────────────────────────────────────
function initGame(mode) {
  const today = todayString();
  const stateKey = mode === 'daily' ? `deenly_daily_${today}` : null;

  if (mode === 'daily') {
    const saved = loadGameState(stateKey);
    if (saved) { state = saved; return; }
  }

  const ids = mode === 'daily' ? getDailyIds(today) : getPracticeIds();
  state = {
    mode,
    date: today,
    questionIds: ids,
    currentQ: 0,
    answers: Array(5).fill(null),
    usedGuesses: [],
    waterLevel: 0,
    done: false,
  };

  if (mode === 'daily') saveGameState(stateKey);
}

// ─── Answer processing ────────────────────────────────────────────────────────
function processGuess(raw) {
  if (state.done) return;
  const q = getQuestionById(state.questionIds[state.currentQ]);
  const result = matchAnswer(raw, q);

  if (result.type === 'hint') {
    return { type: 'hint', message: result.message };
  }

  if (result.type === 'correct') {
    const tier = result.answer.tier;
    state.answers[state.currentQ] = { tier, correct: true, guess: raw };
    state.waterLevel = Math.min(100, state.waterLevel + TIER_WATER[tier]);
    advanceQuestion();
    return { type: 'correct', tier, answer: result.answer };
  }

  // Miss — no try cost; timer is the only constraint
  state.usedGuesses.push(raw);
  return { type: 'miss' };
}

function advanceQuestion() {
  stopTimer();
  state.currentQ++;
  state.usedGuesses = [];
  if (state.currentQ >= 5) {
    state.done = true;
    updateStats();
  }
  if (state.mode === 'daily') saveGameState(`deenly_daily_${state.date}`);
}

function updateStats() {
  const today = todayString();
  stats.gamesPlayed++;
  for (const ans of state.answers) {
    if (ans && ans.tier === 'Legendary') stats.legendaryCount++;
  }
  if (state.waterLevel === 100) stats.perfectFills++;

  if (stats.lastPlayed) {
    const last = new Date(stats.lastPlayed);
    const now = new Date(today);
    const diff = (now - last) / 86400000;
    stats.currentStreak = diff <= 1 ? stats.currentStreak + 1 : 1;
  } else {
    stats.currentStreak = 1;
  }
  if (state.mode === 'daily') {
    stats.lastPlayed = today;
    saveStats();
  }
}

// ─── Game number (days since launch) ─────────────────────────────────────────
function getGameNumber() {
  const launch = new Date('2026-11-18');
  const today = new Date();
  today.setHours(0,0,0,0);
  return Math.floor((today - launch) / 86400000) + 1;
}

// ─── Share text ───────────────────────────────────────────────────────────────
function buildShareText() {
  const emojis = state.answers.map(a => a ? (a.correct ? TIER_EMOJI[a.tier] : TIER_EMOJI.miss) : '⬛');
  return `Zamzam #${getGameNumber()} — ${state.waterLevel}% 💧\n${emojis.join('')}`;
}

// ─── UI helpers ───────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const qs = sel => document.querySelector(sel);

function setWaterLevel(pct) {
  // The water-rect fills from bottom: translateY(0) = full, translateY(100%) = empty
  const rect = document.getElementById('water-rect');
  if (rect) {
    const bottleHeight = 320;
    const translateY = bottleHeight * (1 - pct / 100);
    rect.setAttribute('transform', `translate(0,${translateY})`);
  }
  // Position the wave at the water surface
  const waveG = document.getElementById('wave-g');
  if (waveG) {
    const bottleHeight = 320;
    const waterY = bottleHeight * (1 - pct / 100);
    waveG.setAttribute('transform', `translate(0,${waterY})`);
  }
  $('water-level-text').textContent = Math.round(pct) + '%';
  // Update legendary dots
  const dots = document.querySelectorAll('.leg-dot');
  const legend = state.answers.filter(a => a && a.tier === 'Legendary').length;
  dots.forEach((d, i) => d.classList.toggle('lit', i < legend));
}

function animateDrop() {
  const drop = document.createElement('div');
  drop.className = 'drop-anim';
  $('bottle-wrap').appendChild(drop);
  drop.addEventListener('animationend', () => drop.remove());
}

function renderAnswerList(qObj) {
  const tiers = ['Legendary', 'Rare', 'Uncommon', 'Common'];
  let html = '<div class="answer-list">';
  for (const tier of tiers) {
    const answers = qObj.answers.filter(a => a.tier === tier);
    if (!answers.length) continue;
    html += `<div class="answer-tier ${TIER_CLASS[tier]}">
      <span class="tier-label">${tier}</span>
      <div class="tier-answers">`;
    for (const a of answers) {
      const isPlayer = state.answers[state.currentQ - 1] &&
                       state.answers[state.currentQ - 1].correct &&
                       state.answers[state.currentQ - 1].tier === tier &&
                       a.name === (state.answers[state.currentQ - 1].answerName || a.name);
      html += `<span class="answer-pill ${isPlayer ? 'player-answer' : ''}">${a.name}</span>`;
    }
    html += `</div></div>`;
  }
  html += `<p class="did-you-know">💡 ${qObj.didYouKnow}</p>`;
  html += '</div>';
  return html;
}

function renderEndScreen() {
  const pct = state.waterLevel;
  let msg = '';
  if (pct === 100) msg = 'MashaAllah! A perfect fill! 🌟';
  else if (pct >= 80) msg = 'Excellent! You know your rare answers!';
  else if (pct >= 60) msg = 'Great effort — keep aiming for the rarer picks.';
  else if (pct >= 40) msg = 'Good start — try to go beyond the common answers.';
  else msg = 'Keep learning — the rarer answers are waiting!';

  const rows = state.answers.map((a, i) => {
    const q = getQuestionById(state.questionIds[i]);
    const emoji = a ? (a.correct ? TIER_EMOJI[a.tier] : TIER_EMOJI.miss) : TIER_EMOJI.miss;
    const label = a && a.correct ? `${a.tier} (${a.guess})` : 'Missed';
    return `<div class="result-row"><span>${emoji}</span><span class="result-q">${q.prompt}</span><span class="result-label ${a && a.correct ? TIER_CLASS[a.tier] : ''}">${label}</span></div>`;
  }).join('');

  $('end-screen').innerHTML = `
    <h2 class="end-title">${pct === 100 ? '✨ Bottle Filled!' : `${pct}% filled`}</h2>
    <p class="end-msg">${msg}</p>
    <div class="result-rows">${rows}</div>
    <div class="end-stats">
      <span>🕹️ ${stats.gamesPlayed} played</span>
      <span>🔥 ${stats.currentStreak} day streak</span>
      <span>🟡 ${stats.legendaryCount} Legendaries</span>
      <span>💎 ${stats.perfectFills} perfect fills</span>
    </div>
    <button id="copy-btn" class="btn-primary">Copy result</button>
    ${state.mode === 'practice' ? '<button id="play-daily-btn" class="btn-secondary">Play today\'s daily</button>' : '<button id="play-practice-btn" class="btn-secondary">Practice mode</button>'}
  `;
  $('end-screen').style.display = 'flex';
  $('game-area').style.display = 'none';

  $('copy-btn').addEventListener('click', async () => {
    const text = buildShareText();
    try {
      await navigator.clipboard.writeText(text);
      $('copy-btn').textContent = 'Copied!';
      setTimeout(() => { $('copy-btn').textContent = 'Copy result'; }, 2000);
    } catch {
      prompt('Copy this:', text);
    }
  });

  const switchBtn = $('play-daily-btn') || $('play-practice-btn');
  if (switchBtn) {
    switchBtn.addEventListener('click', () => {
      startGame(switchBtn.id === 'play-daily-btn' ? 'daily' : 'practice');
    });
  }

  if (pct === 100) triggerBottleFull();
}

function triggerBottleFull() {
  document.querySelector('.bottle-container').classList.add('bottle-full');
  setTimeout(() => document.querySelector('.bottle-container').classList.remove('bottle-full'), 4000);
}

// ─── Main render loop ──────────────────────────────────────────────────────────
function renderQuestion() {
  if (state.done) { renderEndScreen(); return; }

  const q = getQuestionById(state.questionIds[state.currentQ]);
  $('question-num').textContent = `Question ${state.currentQ + 1} of 5`;
  $('question-prompt').textContent = q.prompt;
  $('guess-input').value = '';
  $('guess-input').focus();
  $('feedback').textContent = '';
  $('answer-reveal').innerHTML = '';
  $('next-btn').style.display = 'none';
  $('game-area').style.display = 'flex';
  $('end-screen').style.display = 'none';

  setWaterLevel(state.waterLevel);
  startTimer();

  // Show previous guesses
  $('prev-guesses').innerHTML = state.usedGuesses.map(g =>
    `<span class="wrong-guess">${g}</span>`
  ).join(' ');
}

function showReveal(qObj) {
  stopTimer();
  $('answer-reveal').innerHTML = renderAnswerList(qObj);
  $('next-btn').style.display = 'block';
  $('next-btn').textContent = state.currentQ >= 5 ? 'See results' : 'Next question →';
  $('guess-input').disabled = true;
}

// ─── Event handlers ───────────────────────────────────────────────────────────
function handleGuess() {
  const raw = $('guess-input').value.trim();
  if (!raw || $('guess-input').disabled) return;

  const result = processGuess(raw);
  const q = getQuestionById(state.questionIds[state.currentQ - (result.type === 'correct' ? 1 : 0)]);

  if (result.type === 'hint') {
    $('feedback').textContent = result.message;
    $('feedback').className = 'feedback hint';
    return;
  }

  if (result.type === 'correct') {
    animateDrop();
    setWaterLevel(state.waterLevel);
    $('feedback').textContent = `✅ ${result.answer.name} — ${result.tier}!`;
    $('feedback').className = `feedback correct ${TIER_CLASS[result.tier]}`;
    if (state.answers[state.currentQ - 1]) {
      state.answers[state.currentQ - 1].answerName = result.answer.name;
    }
    showReveal(q);
    if (state.done) setTimeout(renderEndScreen, 1200);
    return;
  }

  if (result.type === 'miss') {
    $('feedback').textContent = `Not on the list — keep trying!`;
    $('feedback').className = 'feedback miss';
    $('prev-guesses').innerHTML = state.usedGuesses.map(g =>
      `<span class="wrong-guess">${g}</span>`
    ).join(' ');
  }

  $('guess-input').value = '';
}

// ─── Startup ──────────────────────────────────────────────────────────────────
function startGame(mode) {
  initGame(mode);
  $('mode-select').style.display = 'none';
  $('end-screen').style.display = 'none';
  renderQuestion();
}

async function boot() {
  try {
    await loadData();
  } catch (e) {
    console.error('Failed to load data:', e);
    document.body.innerHTML = '<p style="color:white;padding:2rem">Failed to load questions. Please refresh.</p>';
    return;
  }

  // Wire up events
  $('btn-daily').addEventListener('click', () => startGame('daily'));
  $('btn-practice').addEventListener('click', () => startGame('practice'));

  $('submit-btn').addEventListener('click', handleGuess);
  $('guess-input').addEventListener('keydown', e => { if (e.key === 'Enter') handleGuess(); });

  $('next-btn').addEventListener('click', () => {
    $('guess-input').disabled = false;
    renderQuestion();
  });

  // Check if daily is already complete
  const today = todayString();
  const saved = loadGameState(`deenly_daily_${today}`);
  if (saved && saved.done) {
    state = saved;
    $('mode-select').style.display = 'none';
    renderEndScreen();
  } else {
    $('mode-select').style.display = 'flex';
  }
}

document.addEventListener('DOMContentLoaded', boot);
