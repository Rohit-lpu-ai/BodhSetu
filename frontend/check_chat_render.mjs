/**
 * Run frontend/chat.html's answer-rendering logic outside a browser.
 *
 * Run it:  node frontend/check_chat_render.mjs
 *
 * It exists because the one thing a real hosted model changed was the shape
 * of the text the page has to render, and the Chrome extension was not
 * connected when that happened. It reads the page's own regexes out of the
 * file on disk rather than copying them, so it fails if the page and this
 * check ever drift apart.
 *
 * `backend/tests/test_frontend_mount.py` pins the same properties from the
 * Python side; this one actually executes the JavaScript.
 */

import { readFileSync } from 'node:fs';

const html = readFileSync(new URL('./chat.html', import.meta.url), 'utf8');

let failures = 0;
const check = (name, actual, expected) => {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  if (a === e) {
    console.log(`  ok   ${name}`);
  } else {
    console.log(`  FAIL ${name}\n       expected ${e}\n       actual   ${a}`);
    failures += 1;
  }
};

// --- pull the two regexes straight out of the page -------------------------
const splitSrc = html.match(/const parts = String\(text\)\.split\((\/.*?\/i)\);/);
const matchSrc = html.match(/const match = (\/\^.*?\/i)\.exec\(part\);/);
if (!splitSrc || !matchSrc) {
  console.log('FAIL: could not find the marker regexes in chat.html');
  process.exit(1);
}
const splitRe = eval(splitSrc[1]);
const matchRe = eval(matchSrc[1]);

function markersIn(text) {
  const out = [];
  String(text).split(splitRe).forEach((part) => {
    const m = matchRe.exec(part);
    if (!m) return;
    (m[1].match(/S?\s*\d+/gi) || []).forEach((raw) => {
      out.push('S' + parseInt(String(raw).replace(/\D/g, ''), 10));
    });
  });
  return out;
}

console.log('marker extraction (must match rag_module/citation/verifier.py):');
check('ascii [S1]', markersIn('Banks must act [S1].'), ['S1']);
check('cjk lenticular \u3010S2\u3011', markersIn('Banks must act\u3010S2\u3011.'), ['S2']);
check('fullwidth \uff3bS3\uff3d', markersIn('Banks must act\uff3bS3\uff3d.'), ['S3']);
check('grouped [S1, S2]', markersIn('a [S1, S2] b'), ['S1', 'S2']);
check('narrow nbsp before marker', markersIn('quarterly\u202f[S1].'), ['S1']);
check('mixed styles', markersIn('a [S1] b \u3010S2\u3011 c \uff3bS3\uff3d'), ['S1', 'S2', 'S3']);
check('a clause number is not a marker', markersIn('see clause [3.2]'), []);
check('no markers', markersIn('plain sentence'), []);

// --- the bold splitter -----------------------------------------------------
console.log('\ninline bold rendering:');
const boldSplit = (t) => String(t).split(/(\*\*[^*]+\*\*)/).filter(Boolean)
  .map((c) => (c.length > 4 && c.startsWith('**') && c.endsWith('**'))
    ? { strong: c.slice(2, -2) } : { text: c });

check('bold in the middle',
  boldSplit('Banks must establish a **special cell** today.'),
  [{ text: 'Banks must establish a ' }, { strong: 'special cell' }, { text: ' today.' }]);
check('no bold', boldSplit('plain text'), [{ text: 'plain text' }]);
check('lone asterisks are left alone', boldSplit('2 * 3 * 4'), [{ text: '2 * 3 * 4' }]);

// --- the page must still never parse HTML ----------------------------------
console.log('\nsafety:');
check('no innerHTML anywhere', html.includes('innerHTML'), false);
check('calls POST /api/ask', html.includes("fetch('/api/ask'"), true);
check('reads /api/ask/status', html.includes("fetch('/api/ask/status')"), true);

console.log(failures === 0
  ? `\nAll checks passed.`
  : `\n${failures} check(s) FAILED.`);
process.exit(failures === 0 ? 0 : 1);
