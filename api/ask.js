// api/ask.js
import { createReadStream } from "node:fs";
import formidable from "formidable";
import OpenAI from "openai";
import { cosine } from "./_cosine.js";

export const config = { api: { bodyParser: false } };

// ---- ENV ----
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL || "text-embedding-3-small";
const STT_MODEL       = process.env.STT_MODEL || "whisper-1";
const THRESH          = parseFloat(process.env.SIMILARITY_THRESHOLD || "0.82");
const WEBFLOW_TOKEN   = process.env.WEBFLOW_TOKEN;
const WEBFLOW_SITE_ID = process.env.WEBFLOW_SITE_ID;
const WEBFLOW_COLLECTION_ID = process.env.WEBFLOW_COLLECTION_ID;

// ---- small caches (persist across warm invocations) ----
const KB_CACHE_MS  = 5 * 60 * 1000;
const VEC_CACHE_MS = 30 * 60 * 1000;
globalThis.__kbCache  = globalThis.__kbCache  || { at: 0, kb: null };
globalThis.__vecCache = globalThis.__vecCache || { at: 0, model: "", count: 0, vectors: [] };

// ---- helpers ----
function setCORS(res){
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
}

async function embed(text){
  const r = await openai.embeddings.create({ model: EMBEDDING_MODEL, input: text });
  return r.data[0].embedding;
}

function topK(qVec, vecs, k=3){
  return vecs.vectors
    .map(v => ({ id: v.id, score: cosine(qVec, v.embedding) }))
    .sort((a,b) => b.score - a.score)
    .slice(0,k);
}

async function transcribeAudio(filepath){
  const file = createReadStream(filepath);
  const r = await openai.audio.transcriptions.create({ model: STT_MODEL, file });
  return (r.text || "").trim();
}

// ---- pull items from Webflow CMS v2 ----
async function loadKB(){
  const now = Date.now();
  if (globalThis.__kbCache.kb && now - globalThis.__kbCache.at < KB_CACHE_MS){
    return globalThis.__kbCache.kb;
  }
  if (!WEBFLOW_TOKEN || !WEBFLOW_SITE_ID || !WEBFLOW_COLLECTION_ID){
    throw new Error("Missing WEBFLOW_* env vars.");
  }

  const url = `https://api.webflow.com/v2/collections/${WEBFLOW_COLLECTION_ID}/items?limit=100`;
  const resp = await fetch(url, {
    headers: {
      Authorization: `Bearer ${WEBFLOW_TOKEN}`,
      "x-webflow-site-id": WEBFLOW_SITE_ID,
      accept: "application/json"
    }
  });
  if (!resp.ok){
    const t = await resp.text();
    throw new Error(`Webflow fetch failed: ${resp.status} ${t}`);
  }
  const data = await resp.json();

  // Your fields are "Question" (plain), "Answer" (rich), "Keywords / Variations" (plain)
  // API v2 returns them under fieldData with machine keys:
  // question -> question
  // answer   -> answer (may be {plainText,text} or string)
  // keywords / variations -> usually "keywords-/-variations"
  const entries = (data?.items || []).map((item,i) => {
    const f = item.fieldData || item;
    const id = f._id || item.id || String(i);
    const question = (f.question || "").toString().trim();

    // handle rich text structures
    const ansRaw = f.answer?.plainText ?? f.answer?.text ?? f.answer ?? "";
    const answer = (ansRaw || "").toString().trim();

    // try both key shapes for your field name
    const kw = (f["keywords / variations"] ?? f["keywords-/-variations"] ?? f.keywords ?? "")
      .toString().split(",").map(s=>s.trim()).filter(Boolean);

    if (!answer) return null;
    return { id, question, question_patterns: [question, ...kw].filter(Boolean), answer };
  }).filter(Boolean);

  const kb = { entries };
  globalThis.__kbCache = { at: now, kb };
  return kb;
}

async function ensureVectors(kb){
  const now = Date.now();
  const stale = !globalThis.__vecCache.vectors.length ||
                globalThis.__vecCache.model !== EMBEDDING_MODEL ||
                globalThis.__vecCache.count !== kb.entries.length ||
                now - globalThis.__vecCache.at > VEC_CACHE_MS;

  if (!stale) return globalThis.__vecCache;

  const vectors = [];
  for (const e of kb.entries){
    const text = `${(e.question_patterns || []).join(" ; ")}\n\n${e.answer}`;
    const v = await embed(text);
    vectors.push({ id: e.id, embedding: v });
  }
  globalThis.__vecCache = { at: now, model: EMBEDDING_MODEL, count: kb.entries.length, vectors };
  return globalThis.__vecCache;
}

// ---- main handler ----
export default async function handler(req, res){
  setCORS(res);
  if (req.method === "OPTIONS") return res.status(204).end();

  try{
    // 1) get query text (audio, JSON, or GET ?text=)
    let queryText = "";
    if (req.method === "POST"){
      if (req.headers["content-type"]?.includes("multipart/form-data")){
        const form = formidable({ multiples: false });
        const [fields, files] = await form.parse(req);
        const audio = files?.audio?.[0];
        if (!audio) return res.status(400).json({ text: "Missing audio file" });
        queryText = await transcribeAudio(audio.filepath);
      } else {
        const chunks = []; for await (const c of req) chunks.push(c);
        const body = JSON.parse(Buffer.concat(chunks).toString() || "{}");
        queryText = (body.text || body.question || "").toString().trim();
      }
    } else if (req.method === "GET"){
      queryText = (req.query?.text || req.query?.q || "").toString().trim();
    } else {
      return res.status(405).json({ error: "Method not allowed" });
    }

    if (!queryText){
      return res.status(200).json({ text: "I didn’t catch that. Could you repeat your question?" });
    }

    // 2) load CMS + vectors
    const kb = await loadKB();
    if (!kb.entries?.length){
      return res.status(200).json({ text: "The hotel guide is empty. Please check back soon." });
    }
    const vecs = await ensureVectors(kb);

    // 3) retrieve
    const qVec = await embed(queryText);
    const top = topK(qVec, vecs, 3);
    if (!top.length || top[0].score < THRESH){
      return res.status(200).json({ text: "I don’t have that in the hotel guide. Would you like the front desk contact details?" });
    }
    const byId = new Map(kb.entries.map(e => [e.id, e]));
    const answer = top.map(t => byId.get(t.id)).filter(Boolean).map(e => e.answer).join(" ");
    return res.status(200).json({ text: answer });
  }catch(err){
    console.error("[ask] Error:", err);
    return res.status(500).json({ text: "Server error. Please try again in a moment." });
  }
}
