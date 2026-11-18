// Zamzam answer matcher
// Normalise → exact → fuzzy (Levenshtein) → longest-contained

export function normalizeArabic(s) {
  return s
    .replace(/[ً-ٰٟ]/g, '')  // strip harakat
    .replace(/[أإآ]/g, 'ا')                 // unify alef
    .replace(/ة/g, 'ه')                      // taa marbuta
    .replace(/ى/g, 'ي')                      // alef maqsura
    .trim();
}

const FILLER = /\b(al|ul|el|as|ash|ad|an|ar|at|az|the|surah|prophet|imam|battle|of|ibn|bint|abu|umm)\b/g;

export function normalize(s) {
  if (!s) return '';
  // Handle Arabic script separately
  if (/[؀-ۿ]/.test(s)) return normalizeArabic(s);
  return s
    .toLowerCase()
    .normalize('NFD').replace(/[̀-ͯ]/g, '')  // strip diacritics
    .replace(/['‘’ʾʿ`]/g, '')      // strip apostrophes
    .replace(/-/g, ' ')                                 // hyphens to spaces
    .replace(FILLER, ' ')
    .replace(/ee/g, 'i').replace(/oo/g, 'u').replace(/aa/g, 'a')
    .replace(/(.)\1+/g, '$1')                          // collapse doubles
    .replace(/\s+/g, ' ')
    .trim();
}

function levenshtein(a, b) {
  const m = a.length, n = b.length;
  if (!m) return n;
  if (!n) return m;
  const dp = Array.from({length: m + 1}, (_, i) => i);
  for (let j = 1; j <= n; j++) {
    let prev = dp[0];
    dp[0] = j;
    for (let i = 1; i <= m; i++) {
      const temp = dp[i];
      dp[i] = a[i-1] === b[j-1] ? prev : 1 + Math.min(prev, dp[i], dp[i-1]);
      prev = temp;
    }
  }
  return dp[m];
}

function fuzzyThreshold(len) {
  if (len < 4) return 0;
  if (len <= 6) return 1;
  return 2;
}

/**
 * Returns one of:
 *   { type: 'hint',    message: string }
 *   { type: 'correct', answer: AnswerObj, matchType: 'exact'|'fuzzy'|'contained' }
 *   { type: 'miss' }
 */
export function matchAnswer(raw, question) {
  const input = raw.trim();
  if (!input) return { type: 'miss' };

  const normInput = normalize(input);

  // Build normalised alias table once
  const aliasMap = [];
  for (const answer of question.answers) {
    for (const alias of answer.aliases) {
      aliasMap.push({ norm: normalize(alias), answer });
    }
  }

  // 1. Exact match first (must beat hints and fuzzy)
  for (const { norm, answer } of aliasMap) {
    if (norm === normInput) return { type: 'correct', answer, matchType: 'exact' };
  }

  // 2. Check hints only after no exact match (no try cost)
  for (const [pattern, msg] of Object.entries(question.hints || {})) {
    const normPat = normalize(pattern);
    if (normPat === normInput || normInput.includes(normPat)) {
      return { type: 'hint', message: msg };
    }
  }

  // 3. Fuzzy (Levenshtein)
  const threshold = fuzzyThreshold(normInput.length);
  if (threshold > 0) {
    let best = null, bestDist = threshold + 1;
    for (const { norm, answer } of aliasMap) {
      // Skip very different lengths to avoid false positives (e.g. "Nas" matching "Nasr")
      if (Math.abs(norm.length - normInput.length) > threshold) continue;
      const d = levenshtein(normInput, norm);
      if (d < bestDist) { bestDist = d; best = answer; }
    }
    if (best) return { type: 'correct', answer: best, matchType: 'fuzzy' };
  }

  // 4. Longest alias contained in input (for multi-word answers)
  let longestContained = null, longestLen = 0;
  for (const { norm, answer } of aliasMap) {
    if (norm.length > 3 && normInput.includes(norm) && norm.length > longestLen) {
      longestLen = norm.length;
      longestContained = answer;
    }
  }
  if (longestContained) return { type: 'correct', answer: longestContained, matchType: 'contained' };

  return { type: 'miss' };
}
