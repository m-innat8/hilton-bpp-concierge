import { createReadStream } from "node:fs";
import fs from "node:fs/promises";
import path from "node:path";
import formidable from "formidable";
import OpenAI from "openai";
import { cosine } from "./_cosine.js";

export const config = { api: { bodyParser: false } };

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL || "text-embedding-3-small";
const STT_MODEL = process.env.STT_MODEL || "whisper-1";
const THRESH = parseFloat(process.env.SIMILARITY_THRESHOLD || "0.82");

// CORS
function setCORS(res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
}

async function loadKB() {
  const kbPath = path.join(process.cwd(), "kb.json");           // root-level kb.json
  const embPath = path.join(process.cwd(), "data", "kb_embeddings.json");
  const kb = JSON.parse(await fs.readFile(kbPath, "utf-8"));
  let embeddings = { model: "", vectors: [] };
  try { embeddings = JSON.parse(await fs.readFile(embPath, "utf-8")); } catch {}
  return { kb, embPath, embeddings };
}

async function ensureEmbeddings(kb, embPath, embeddings) {
  if (embeddings.vectors?.length === kb.entries.length && embeddings.model) return embeddings;
  const vectors = [];
  for (const entry of kb.entries) {
    const text = `${(entry.question_patterns||[]).join(" ; ")}\n\n${entry.answer}`;
    const resp = await openai.embeddings.create({ model: EMBEDDING_MODEL, input: text });
    vectors.push({ id: entry.id, embedding: resp.data[0].embedding });
  }
  const out = { model: EMBEDDING_MODEL, vectors };
  await fs.mkdir(path.dirname(embPath), { recursive: true });
  await fs.writeFile(embPath, JSON.stringify(out));
  return out;
}

async function transcribeAudio(filepath) {
  const file = createReadStream(filepath);
  const resp = await openai.audio.transcriptions.create({ model: STT_MODEL, file });
  return (resp.text || "").trim();
}

async function embed(text) {
  const resp = await openai.embeddings.create({ model: EMBEDDING_MODEL, input: text });
  return resp.data[0].embedding;
}

function topK(queryVec, embeddings, k = 3) {
  return embeddings.vectors
    .map(v => ({ id: v.id, score: cosine(queryVec, v.embedding) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
}

export default async function handler(req, res) {
  setCORS(res);
  if (req.method === "OPTIONS") return res.status(204).end();
  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });

  try {
    const form = formidable({ multiples: false });
    const [fields, files] = await form.parse(req);
    const audio = files?.audio?.[0];
    if (!audio) return res.status(400).json({ text: "Missing audio file" });

    const queryText = await transcribeAudio(audio.filepath);
    if (!queryText) return res.status(200).json({ text: "I didn't catch that. Could you repeat your question?" });

    const { kb, embPath, embeddings } = await loadKB();
    const ensured = await ensureEmbeddings(kb, embPath, embeddings);

    const qVec = await embed(queryText);
    const top = topK(qVec, ensured, 3);

    if (!top.length || top[0].score < THRESH) {
      return res.status(200).json({
        text: "I don’t have that information in the hotel guide. Would you like the front desk contact details?"
      });
    }

    const table = new Map(kb.entries.map(e => [e.id, e]));
    const selected = top.map(t => table.get(t.id)).filter(Boolean);
    const answer = selected.map(e => e.answer).join(" ");

    res.status(200).json({ text: answer });
  } catch (err) {
    console.error(err);
    res.status(500).json({ text: "Server error. Please try again in a moment." });
  }
}
