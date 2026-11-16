/* BodhSetu — the one place the frontend talks to the NiyamSetu API.
 *
 * The pages are served by the FastAPI application itself, so `/api/...` is
 * same-origin: no CORS, no second port, no proxy. `API_BASE` is still a
 * variable rather than a literal, because opening a page straight off the
 * filesystem (file://) has no origin to be relative to, and falling back to
 * the loopback dev server keeps that case working.
 *
 * Three rules this module exists to enforce, because every page needs them
 * and none of them should be reimplemented per page:
 *
 *  1. The API's error envelope (CLAUDE.md §12.1) is surfaced, not swallowed.
 *     `ApiError` carries the real `code` and `message` so a page can show
 *     what actually went wrong instead of "something went wrong".
 *  2. A stale response can never win. `Sequencer` hands out monotonic ids;
 *     a render guarded by `seq.isCurrent(id)` cannot be overwritten by an
 *     earlier request that happened to land later.
 *  3. Document text is never handed to the browser as markup. `highlight()`
 *     builds DOM nodes from the API's `[[hl]]` sentinels; the text itself
 *     only ever reaches the page through `textContent`.
 */

const API_BASE = (function () {
  if (window.location.protocol === 'file:') return 'http://127.0.0.1:8000';
  return '';
})();

class ApiError extends Error {
  constructor(code, message, status, details) {
    super(message);
    this.name = 'ApiError';
    this.code = code || 'UNKNOWN';
    this.status = status || 0;
    this.details = details || {};
  }
}

/**
 * Call the API and return parsed JSON, or throw an ApiError.
 *
 * A non-2xx carrying the documented envelope keeps its code and message. A
 * non-2xx that does not (a proxy error page, say) still throws something
 * legible rather than a JSON parse failure three frames away.
 */
async function api(path, options) {
  let response;
  try {
    response = await fetch(API_BASE + '/api' + path, options);
  } catch (err) {
    throw new ApiError(
      'BACKEND_UNREACHABLE',
      'Could not reach the NiyamSetu backend. Is the application running?',
      0,
      { cause: String(err && err.message) }
    );
  }

  const type = response.headers.get('Content-Type') || '';
  let body = null;
  if (type.includes('json')) {
    try { body = await response.json(); } catch (err) { body = null; }
  }

  if (!response.ok) {
    const envelope = body && body.error;
    throw new ApiError(
      envelope ? envelope.code : 'HTTP_' + response.status,
      envelope ? envelope.message : `The API returned ${response.status}.`,
      response.status,
      envelope ? envelope.details : {}
    );
  }
  if (body === null) {
    throw new ApiError('BAD_RESPONSE', 'The API returned a non-JSON body.', response.status, {});
  }
  return body;
}

function apiGet(path, params) {
  const query = new URLSearchParams();
  Object.keys(params || {}).forEach((key) => {
    const value = params[key];
    if (value === null || value === undefined || value === '') return;
    if (Array.isArray(value)) value.forEach((v) => query.append(key, v));
    else query.append(key, value);
  });
  const qs = query.toString();
  return api(path + (qs ? '?' + qs : ''));
}

function apiPost(path, payload) {
  return api(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

/* ------------------------------------------------------------------ sequencing */

/** Monotonic request ids, so a slow earlier response cannot overwrite a newer one. */
class Sequencer {
  constructor() { this.current = 0; }
  next() { return ++this.current; }
  isCurrent(id) { return id === this.current; }
  /** Invalidate everything in flight — used when the selection itself changes. */
  invalidate() { this.current += 1; }
}

/* ----------------------------------------------------------------------- DOM */

const $ = (id) => document.getElementById(id);

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function clear(node) { while (node && node.firstChild) node.removeChild(node.firstChild); }

/**
 * Render the API's `[[hl]]`…`[[/hl]]` snippet sentinels as <mark> elements.
 *
 * The sentinels are plain text on purpose (see the API docstring for
 * /api/search/clauses): the surrounding content came out of a PDF, and
 * assembling HTML from document text is how a document becomes a script.
 * Everything between the sentinels goes in through textContent.
 */
function highlight(snippet) {
  const fragment = document.createDocumentFragment();
  const parts = String(snippet == null ? '' : snippet).split(/(\[\[hl\]\]|\[\[\/hl\]\])/);
  let on = false;
  parts.forEach((part) => {
    if (part === '[[hl]]') { on = true; return; }
    if (part === '[[/hl]]') { on = false; return; }
    if (!part) return;
    if (on) fragment.appendChild(el('mark', null, part));
    else fragment.appendChild(document.createTextNode(part));
  });
  return fragment;
}

/** A dashed note in the supplied design, used for every empty and error state. */
function noteNode(kind, heading, body) {
  const node = el('div', 'note' + (kind ? ' ' + kind : ''));
  node.appendChild(el('h2', null, heading));
  if (body) node.appendChild(el('p', null, body));
  return node;
}

/** Replace a container's contents with an error state built from an ApiError. */
function showError(container, err, what) {
  clear(container);
  const heading = err && err.code === 'BACKEND_UNREACHABLE'
    ? 'The backend is not responding'
    : `Could not load ${what}`;
  const detail = err && err.message ? err.message : 'Unknown error.';
  const node = noteNode('bad', heading, detail);
  if (err && err.code && err.code !== 'BACKEND_UNREACHABLE') {
    node.appendChild(el('p', 'meta', 'Error code: ' + err.code));
  }
  container.appendChild(node);
}

/** Loading placeholder rows, in the paper palette. */
function showSkeleton(container, rows) {
  clear(container);
  for (let i = 0; i < (rows || 3); i += 1) {
    const card = el('div', 'card');
    card.appendChild(el('div', 'skeleton wide'));
    card.appendChild(el('div', 'skeleton half'));
    card.appendChild(el('div', 'skeleton half'));
    container.appendChild(card);
  }
}

/* -------------------------------------------------------------------- format */

function fmtDate(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso;
  return d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
}

function plural(n, one, many) { return `${n} ${n === 1 ? one : (many || one + 's')}`; }

function orDash(value) {
  if (value === null || value === undefined || value === '') return 'Not stated';
  return String(value);
}

/* ------------------------------------------------------------------- session */

/**
 * The signed-in profile, if the teammate's auth flow stored one.
 *
 * Auth is the teammate's module (frontend_backend.py + Supabase); nothing in
 * Library, Compare, Ask or ingestion requires it, so this only decorates the
 * header avatar and never gates a page.
 */
function currentUser() {
  try {
    const raw = localStorage.getItem('bodhsetu_user') || sessionStorage.getItem('bodhsetu_user');
    if (!raw) return null;
    const user = JSON.parse(raw);
    return user && typeof user === 'object' ? user : null;
  } catch (err) {
    return null;
  }
}

function userInitial() {
  const user = currentUser();
  const name = (user && user.full_name) || localStorage.getItem('bodhsetu_registered_name') || '';
  return name.trim() ? name.trim().charAt(0).toUpperCase() : 'C';
}

/* ---------------------------------------------------------------------- saved */

/**
 * The saved list, kept in this browser.
 *
 * Only identifiers are stored locally; every saved item's title, pages and
 * text are re-read from the API when the Saved page renders, so nothing on
 * that page is a cached copy of document content. There is no `saved_items`
 * table and this does not invent one — adding one would mean a migration and
 * an account to hang it off, and the accounts live in the teammate's
 * Supabase project, not in PostgreSQL.
 */
const SAVED_KEY = 'bodhsetu_saved_v1';

function savedItems() {
  try {
    const raw = localStorage.getItem(SAVED_KEY);
    const list = raw ? JSON.parse(raw) : [];
    return Array.isArray(list) ? list : [];
  } catch (err) {
    return [];
  }
}

function savedKeyOf(item) {
  return item.kind === 'clause' ? 'clause:' + item.clause_id : 'document:' + item.document_id;
}

function isSaved(item) {
  const key = savedKeyOf(item);
  return savedItems().some((entry) => savedKeyOf(entry) === key);
}

function toggleSaved(item) {
  const key = savedKeyOf(item);
  const list = savedItems().filter((entry) => savedKeyOf(entry) !== key);
  const added = list.length === savedItems().length;
  if (added) list.unshift(Object.assign({ saved_at: new Date().toISOString() }, item));
  try { localStorage.setItem(SAVED_KEY, JSON.stringify(list.slice(0, 200))); } catch (err) { /* quota */ }
  return added;
}

function removeSaved(item) {
  const key = savedKeyOf(item);
  const list = savedItems().filter((entry) => savedKeyOf(entry) !== key);
  try { localStorage.setItem(SAVED_KEY, JSON.stringify(list)); } catch (err) { /* quota */ }
}

/* ---------------------------------------------------------------------- shell */

const NAV = [
  { href: 'user_home.html', label: 'Home', key: 'home' },
  { href: 'explore.html', label: 'Explore Policies', key: 'explore' },
  { href: 'updates.html', label: 'My Updates', key: 'updates' },
  { href: 'saved.html', label: 'Saved', key: 'saved' },
  { href: 'add-document.html', label: 'Add Document', key: 'add' },
];

/**
 * Build the header the supplied pages use, with one entry marked active.
 *
 * The markup mirrors user_home.html's header exactly — same nav, same
 * language select, same avatar circle — so a page built here is
 * indistinguishable from the ones that shipped with the design.
 */
function mountHeader(activeKey) {
  const header = el('header', 'site');

  const nav = el('nav');
  nav.setAttribute('aria-label', 'Main navigation');
  NAV.forEach((entry) => {
    const link = el('a', entry.key === activeKey ? 'active' : null, entry.label);
    link.href = entry.href;
    if (entry.key === activeKey) link.setAttribute('aria-current', 'page');
    nav.appendChild(link);
  });

  const right = el('div', 'right');
  const select = el('select');
  select.setAttribute('aria-label', 'Select language');
  select.appendChild(el('option', null, 'English'));
  select.appendChild(el('option', null, 'हिन्दी'));
  select.classList.add('compact');
  // Hindi output is Open Decision D-3 and is paused: the control is part of
  // the supplied design and stays, but it must not pretend to translate.
  // The notice is inline rather than a dialog — a modal would block the page
  // and says nothing an inline line cannot.
  select.addEventListener('change', (event) => {
    if (event.target.selectedIndex === 1) {
      event.target.selectedIndex = 0;
      const existing = document.getElementById('lang-notice');
      if (existing) existing.remove();
      const notice = el('p', 'meta', 'हिन्दी अनुवाद अभी उपलब्ध नहीं है — Hindi output is not implemented yet.');
      notice.id = 'lang-notice';
      notice.style.width = '100%';
      notice.style.textAlign = 'right';
      header.appendChild(notice);
      setTimeout(() => { const n = document.getElementById('lang-notice'); if (n) n.remove(); }, 6000);
    }
  });

  const avatar = el('a', 'avatar', userInitial());
  avatar.href = currentUser() ? 'user_home.html' : 'login.html';
  avatar.title = currentUser() ? 'Signed in' : 'Sign in';

  right.appendChild(select);
  right.appendChild(avatar);
  header.appendChild(nav);
  header.appendChild(right);
  document.body.insertBefore(header, document.body.firstChild);
  return header;
}

function mountFooter(text) {
  const footer = el('footer', 'site', text || 'Answers grounded in official sources.');
  document.body.appendChild(footer);
  return footer;
}

/** Read a query-string parameter. */
function param(name) {
  return new URLSearchParams(window.location.search).get(name);
}
