# Median

A daily browser game. 5 questions, same for everyone. Score by guessing close to the **median** of all players' answers.

No login. No download. Mobile-first.

---

## How it works

- Each day, 5 questions are selected deterministically from `questions.json`
- Each answer is a number; players drag a gauge needle to pick their value
- After locking in, you see every player's tick marks on the gauge, the bold median line, and your score
- **Scoring**: 100 points at the median, falling to 0 at 25% of the range away → max 500/day
- One play per day per device (device ID stored in `localStorage`)
- **Seed answers**: 40 fake answers generated from each question's crowd distribution keep the game fun before real players arrive. They phase out as real answers roll in (fully gone at 40 real players)

---

## Setup

### 1. Create a Supabase project

Go to [supabase.com](https://supabase.com) → New project (free tier is fine).

### 2. Run the schema

In your Supabase dashboard → **SQL Editor** → paste the contents of `schema.sql` and run it.

### 3. Add your keys to `index.html`

Open `index.html` and find these two lines near the top of the `<script>`:

```js
const SUPABASE_URL = 'YOUR_SUPABASE_URL';
const SUPABASE_ANON_KEY = 'YOUR_SUPABASE_ANON_KEY';
```

Replace the placeholders with values from your Supabase dashboard:
- **Project URL**: Settings → API → Project URL
- **anon/public key**: Settings → API → Project API keys → `anon public`

Also update the launch date:

```js
const LAUNCH_DATE = '2025-01-01'; // puzzle #1 starts on this date
```

### 4. Deploy to Vercel

The game is a single static folder — no build step needed.

```bash
npm i -g vercel
cd median
vercel
```

Vercel will auto-detect a static site and deploy it. You'll get a URL like `https://median-abc.vercel.app`.

Alternatively, drag the `median/` folder into [vercel.com/new](https://vercel.com/new) or use any static host (Netlify, GitHub Pages, Cloudflare Pages).

### 5. Set your domain in the share text (optional)

In `index.html`, find the `share()` function and update the link:

```js
const text = `Median #${state.puzzleNum} ${emojis} ${totalPts}/500\nhttps://your-domain.com`;
```

---

## Dev settings

Inside `index.html` near the top of the `<script>`:

```js
const SEED_ENABLED = true;   // set false to see only real answers
const SEED_COUNT = 40;       // fake answers per question at launch
```

---

## File structure

```
median/
  index.html      – single-file game (HTML + CSS + JS)
  questions.json  – 100 questions with seed crowd distributions
  schema.sql      – Supabase table + RLS policies
  README.md       – this file
```

---

## Adding questions

Add entries to `questions.json`. Each question needs:

| Field | Description |
|-------|-------------|
| `id` | Unique integer |
| `text` | Question string |
| `min` | Gauge minimum |
| `max` | Gauge maximum |
| `step` | Snap increment (e.g. `1`, `0.5`, `5`) |
| `unit` | Label shown after the value (e.g. `"£"`, `"%"`, `""`) |
| `seed.median` | Expected crowd median |
| `seed.spread` | Standard deviation of crowd answers |
| `seed.skew` | Skew factor: positive = right tail, negative = left tail |

---

## Share format

```
Median #3 🎯🟢🟡🎯🔴 412/500
https://your-domain.com
```

Emoji key: 🎯 within 2% of range · 🟢 within 7% · 🟡 within 15% · 🔴 otherwise
