# Zamzam — Daily Islamic Trivia

A browser-based daily trivia game inspired by Wordle and Krillion. Players are rewarded for giving *rare* correct answers, not just correct ones. Every correct answer pours water into a stylised Zamzam bottle; only 5 Legendary answers fill it completely.

## Running locally

No build step needed — open `index.html` via a local HTTP server (required for ES modules and `fetch`):

```bash
npx serve zamzam
# or
python -m http.server 8080 --directory zamzam
```

Then open `http://localhost:8080` (or `http://localhost:3000` for serve).

## Files

```
zamzam/
  index.html          Main game UI + SVG bottle
  app.js              Game logic, state, localStorage
  matcher.js          Answer normalisation + matching
  admin.html          Schedule view + question browser (local only)
  tests.html          Unit tests (open in browser; no Node required)
  data/
    questions.json    Question bank
    schedule.json     Pre-generated daily schedule (365 days)
    questions.pending.json  Staged questions awaiting approval
  scripts/
    generate-schedule.js    Regenerate schedule.json from scratch
    approve-questions.js    Move pending → live and regenerate schedule
```

## How to add questions

### Quick path (use the slash command)

```
/add-questions core hadith 5
/add-questions fun football 8
```

This drafts candidates into `data/questions.pending.json` and a review CSV. Review the CSV, adjust any tiers with `overrideTier`, then approve:

```bash
node scripts/approve-questions.js q-pending-001 q-pending-002
```

### Manual path

1. Add entries to `data/questions.json` following this schema:

```json
{
  "id": "q34",
  "prompt": "Name a...",
  "category": "quran",
  "pillar": "core",
  "difficulty": 2,
  "didYouKnow": "One-line fun fact shown after the answer.",
  "answers": [
    {
      "name": "Canonical Name",
      "aliases": ["Canonical Name", "alternate spelling", "English equivalent"],
      "tier": "Common",
      "note": "optional note for review"
    }
  ],
  "hints": {
    "ambiguous keyword": "Message shown without costing a try."
  }
}
```

2. Regenerate the schedule:

```bash
node scripts/generate-schedule.js
```

### Question rules

- **Closed list**: every answer checkable against a named source.
- **Same kind**: all answers are the same type (all surahs, all people, etc.).
- **Spread**: ≥4 answers, ideally 8–60, with ≥1 Legendary.
- **No sect disputes**: avoid answers where madhhabs disagree.
- **Pillar**: `"core"` (Quran, Seerah, hadith, aqeedah, worship, Islamic history) or `"fun"` (Muslim culture, sport, food, etc.).

### Tier guidelines

| Tier | Water | Assignment |
|------|-------|------------|
| Common | 5% | First answer most people would give |
| Uncommon | 10% | Known, but not the top-of-mind pick |
| Rare | 15% | Requires real knowledge |
| Legendary | 20% | Only specialists know this |

The game rule: **only 5 Legendary answers fill the bottle to 100%**. The best non-perfect score is 95% (4 Legendary + 1 Rare). This is enforced by the water constants in `app.js` and tested in `tests.html`.

## Answer matching

The matcher (`matcher.js`) normalises input before comparing:

1. Lowercase, strip diacritics, remove apostrophes
2. Drop filler words: `al`, `ul`, `el`, `as`, `ash`, `ad`, `an`, `ar`, `at`, `az`, `the`, `surah`, `prophet`, `imam`, `battle`, `of`, `ibn`, `bint`, `abu`, `umm`
3. Vowel normalisation: `ee→i`, `oo→u`, `aa→a`
4. Collapse double letters

Then matches in order:
1. **Exact alias** — highest priority; prevents e.g. "Nas" matching "Nasr"
2. **Fuzzy** (Levenshtein ≤1 for 4–6 chars, ≤2 for 7+; disabled under 4 chars)
3. **Longest alias contained in input** — catches "I think it's Elisha"

**Hints** are free guesses (no try cost) for ambiguous inputs.

## Daily schedule

`data/schedule.json` maps each date to 5 question IDs. Generated deterministically from the question bank with these constraints:

- ≥1 `core` and ≥1 `fun` question per day
- No two questions from the same `category` on the same day
- Difficulty spread: ~1 easy, 3 medium, 1 hard
- No question repeats within 30 days
- Unused questions preferred over repeats

Regenerate with `node scripts/generate-schedule.js`. Adding new questions to `questions.json` before regenerating will fit them into upcoming days without changing past days (the schedule is written to a file, not computed at runtime).

## Running the tests

Open `tests.html` in a browser (must be served via HTTP, not `file://`). No framework required. Tests cover:

- Normalisation
- Exact vs fuzzy priority (Nas/Nasr, Tawaf/Tawaf al-Ifadah)
- 5 Legendaries = 100% rule
- Hint detection
- Contained-in-input fallback
