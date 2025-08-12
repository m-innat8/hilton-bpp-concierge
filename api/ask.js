// api/ask.js
export const config = { api: { bodyParser: false } };

export default async function handler(req, res) {
  try {
    const q = (req.query?.text || req.query?.q || "").toString().trim();
    if (!q) return res.status(400).json({ answer: "Please provide a question via ?text=..." });

    const WEBFLOW_TOKEN = process.env.WEBFLOW_TOKEN;
    const WEBFLOW_SITE_ID = process.env.WEBFLOW_SITE_ID;
    const WEBFLOW_COLLECTION_ID = process.env.WEBFLOW_COLLECTION_ID;

    if (!WEBFLOW_TOKEN || !WEBFLOW_SITE_ID || !WEBFLOW_COLLECTION_ID) {
      return res.status(500).json({ answer: "Missing Webflow env vars." });
    }

    // Pull items from Webflow CMS (v2)
    const url = `https://api.webflow.com/v2/collections/${WEBFLOW_COLLECTION_ID}/items?limit=100`;
    const cmsRes = await fetch(url, {
      headers: {
        Authorization: `Bearer ${WEBFLOW_TOKEN}`,
        "x-webflow-site-id": WEBFLOW_SITE_ID,
        accept: "application/json"
      }
    });

    if (!cmsRes.ok) {
      const txt = await cmsRes.text();
      throw new Error(`Webflow fetch failed: ${cmsRes.status} ${txt}`);
    }

    const data = await cmsRes.json();
    const items = data?.items || [];

    // Your field keys (Webflow lowercases; "Keywords / Variations" can appear as "keywords-/-variations")
    const query = q.toLowerCase();
    let best = null;

    for (const it of items) {
      const f = it.fieldData || it;
      const question = (f.question || "").toString();
      const answerRaw = f.answer?.plainText ?? f.answer?.text ?? f.answer ?? "";
      const answer = (answerRaw || "").toString().trim();
      const keywords = (
        f["keywords / variations"] ??
        f["keywords-/-variations"] ??
        f.keywords ??
        ""
      ).toString();

      if (!answer) continue;

      const hay = `${question}\n${keywords}`.toLowerCase();
      if (hay.includes(query)) { best = { answer }; break; }
    }

    if (best) return res.status(200).json({ answer: best.answer });
    return res.status(200).json({ answer: "I don’t have that in the hotel guide yet." });
  } catch (err) {
    console.error("ask.js error:", err);
    return res.status(500).json({ answer: "Server error. Please try again later." });
  }
}
