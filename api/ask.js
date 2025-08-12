import formidable from "formidable";
import { createReadStream } from "node:fs";
import OpenAI from "openai";
import { cosine } from "./_cosine.js";

export const config = { api: { bodyParser: false } };

const DEBUG = String(process.env.DEBUG || "").toLowerCase() === "true";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL || "text-embedding-3-small";
const STT_MODEL       = process.env.STT_MODEL || "whisper-1";
const THRESH          = parseFloat(process.env.SIMILARITY_THRESHOLD || "0.82");

const WEBFLOW_TOKEN         = process.env.WEBFLOW_TOKEN;
const WEBFLOW_SITE_ID       = process.env.WEBFLOW_SITE_ID;
const WEBFLOW_COLLECTION_ID = process.env.WEBFLOW_COLLECTION_ID;

const KB_CACHE_MS  = 5 * 60 * 1000;
const VEC_CACHE_MS = 30 * 60 * 1000;
globalThis.__kbCache  = globalThis.__kbCache  || { at: 0, kb: null };
globalThis.__vecCache = globalThis.__vecCache || { at: 0, model: "", count: 0, vectors: [] };

function cors(res){
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
}

function dbgPayload(where, detail, extra = {}) {
  if (!DEBUG) return { text: "Server error. Please try again in a moment." };
  return { error: true, where, detail: String(detail || ""), ...extra };
}

async function loadKB() {
  const now = Date.now();
  if (globalThis.__kbCache.kb && now - globalThis.__kbCache.at < KB_CACHE_MS) {
    return globalThis.__kbCache.kb;
  }
  if (!WEBFLOW_TOKEN || !WEBFLOW_SITE_ID || !WEBFLOW_COLLECTION_ID) {
    throw Object.assign(new Error("Missing Webflow env vars"), { code: "ENV" });
  }

  const url = `https://api.webflow.com/v2/collections/${WEBFLOW_COLLECTION_ID}/items?limit=100`;
  const r = await fetch(url, {
    headers: {
      Authorization: `Bearer ${WEBFLOW_TOKEN}`,
      "x-webflow-site-id": WEBFLOW_SITE_ID,
      accept: "application/json",
    },
  }).catch(e => { throw Object.assign(e, { code: "NET" }); });

  if (!r.ok) {
    const body = await r.text().catch(()=>"(no body)");
    const err = new Error(`Webflow ${r.status}`);
    err.code = "WEBFLOW";
    err.status = r.status;
    err.body = body;
    throw err;
  }

  const json = await r.json().catch(e => { throw Object.assign(e, { code: "PARSE" }); });
  const entries = (json?.items || []).map((item, i) => {
    const f = item.fieldData || item;
    const id  = f._id || item.id || String(i);
    const q   = (f.question || "").toString().trim();
    const aRaw= f.answer?.plainText ?? f.answer?.text ?? f.answer ?? "";
    const a   = (aRaw || "").toString().trim();
    const kw  = (f["keywords / variations"] ?? f["keywords-/-variations"] ?? f.keywords ?? "")
                  .toString().split(",").map(s=>s.trim()).filter(Boolean);
    if (!a) return null;
    return { id, question: q, question_patterns: [q, ...kw].filter(Boolean), answer: a };
  }).filter(Boolean);

  const kb = { entries };
  globalThis.__kbCache = { at: now, kb };
  return kb;
}

async function embed(text) {
  try {
    const r = await openai.embeddings.create({ model: EMBEDDING_MODEL, input: text });
    return r.data[0].embedding;
  } catch (e) {
    e.code = "OPENAI_EMB";
    throw e;
  }
}

async function ensureVectors(kb) {
  const now = Date.now();
  const stale = !globalThis.__vecCache.vectors.length ||
                globalThis.__vecCache.model !== EMBEDDING_MODEL ||
                globalThis.__vecCache.count !== kb.entries.length ||
                now - globalThis.__vecCache.at > VEC_CACHE_MS;
  if (!stale) return globalThis.__vecCache;

  const vectors = [];
  for (const e of kb.entries) {
    const text = `${(e.question_patterns || []).join(" ; ")}\n\n${e.answer}`;
    const v = await embed(text);
    vectors.push({ id: e.id, embedding: v });
  }
  globalThis.__vecCache = { at: now, model: EMBEDDING_MODEL, count: kb.entries.length, vectors };
  return globalThis.__vecCache;
}

function topK(qVec, vecs, k=3){
  return vecs.vectors.map(v => ({ id: v.id, score: cosine(qVec, v.embedding) }))
                     .sort((a,b)=>b.score-a.score).slice(0,k);
}

async function transcribeAudio(filepath){
  try{
    const file = createReadStream(filepath);
    const r = await openai.audio.transcriptions.create({ model: STT_MODEL, file });
    return (r.text || "").trim();
  } catch (e){
    e.code = "OPENAI_STT";
    throw e;
  }
}

export default async function handler(req, res) {
  cors(res);
  if (req.method === "OPTIONS") return res.status(204).end();

  try {
    // 1) get user text
    let queryText = "";
    if (req.method === "POST") {
      if (req.headers["content-type"]?.includes("multipart/form-data")) {
        const form = formidable({ multiples: false });
        const [_, files] = await form.parse(req);
        const audio = files?.audio?.[0];
        if (!audio) return res.status(400).json({ text: "Missing audio file" });
        queryText = await transcribeAudio(audio.filepath);
      } else {
        const chunks = []; for await (const c of req) chunks.push(c);
        const body = JSON.parse(Buffer.concat(chunks).toString() || "{}");
        queryText = (body.text || body.question || "").toString().trim();
      }
    } else if (req.method === "GET") {
      queryText = (req.query?.text || req.query?.q || "").toString().trim();
    } else {
      return res.status(405).json({ error: "Method not allowed" });
    }

    if (!queryText) {
      return res.status(200).json({ text: "I didn’t catch that. Could you rephrase?" });
    }

    // 2) KB + vectors
    let kb;
    try { kb = await loadKB(); }
    catch(e){ return res.status(500).json(dbgPayload("loadKB", e.code || e.message, { status: e.status, body: e.body })); }

    if (!kb.entries?.length) {
      return res.status(200).json({ text: "The hotel guide is empty. Please check back soon." });
    }

    let vecs;
    try { vecs = await ensureVectors(kb); }
    catch(e){ return res.status(500).json(dbgPayload("ensureVectors", e.code || e.message)); }

    // 3) retrieve
    let qVec;
    try { qVec = await embed(queryText); }
    catch(e){ return res.status(500).json(dbgPayload("embed_query", e.code || e.message)); }

    const top = topK(qVec, vecs, 3);
    if (!top.length || top[0].score < THRESH) {
      return res.status(200).json({ text: "I don’t have that in the hotel guide. Would you like the front desk contact details?" });
    }

    const byId = new Map(kb.entries.map(e => [e.id, e]));
    const answer = top.map(t => byId.get(t.id)).filter(Boolean).map(e => e.answer).join(" ");
    return res.status(200).json({ text: answer });

  } catch (err) {
    console.error("ask.js fatal:", err);
    return res.status(500).json(dbgPayload("fatal", err.message));
  }
}
