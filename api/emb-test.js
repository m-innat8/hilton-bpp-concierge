import OpenAI from "openai";

export default async function handler(req, res) {
  try {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) return res.status(500).json({ ok:false, where:"env", detail:"Missing OPENAI_API_KEY" });

    const client = new OpenAI({ apiKey });
    const model = process.env.EMBEDDING_MODEL || "text-embedding-3-small";
    const r = await client.embeddings.create({ model, input: "hello from hilton bpp" });

    return res.status(200).json({ ok:true, model, dim: r.data[0].embedding.length });
  } catch (e) {
    return res.status(500).json({ ok:false, where:"openai", detail: String(e?.error?.message || e.message || e) });
  }
}
