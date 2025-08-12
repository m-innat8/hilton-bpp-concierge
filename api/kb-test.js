export default async function handler(req, res) {
  try {
    const token = process.env.WEBFLOW_TOKEN;
    const site  = process.env.WEBFLOW_SITE_ID;
    const col   = process.env.WEBFLOW_COLLECTION_ID;
    if (!token || !site || !col) {
      return res.status(500).json({ ok:false, where:"env", detail:"Missing WEBFLOW_* env vars" });
    }

    const url = `https://api.webflow.com/v2/collections/${col}/items?limit=100`;
    const r = await fetch(url, {
      headers: {
        Authorization: `Bearer ${token}`,
        "x-webflow-site-id": site,
        accept: "application/json"
      }
    });

    const text = await r.text(); // read body once
    if (!r.ok) {
      return res.status(500).json({ ok:false, where:"webflow", status:r.status, body:text });
    }

    const json = JSON.parse(text);
    const items = json?.items || [];
    const first = items[0]?.fieldData || items[0];

    return res.status(200).json({
      ok: true,
      count: items.length,
      sample: first ? {
        id: first._id || items[0]?.id,
        keys: Object.keys(first).slice(0,10),
        question: first.question ?? null,
        answer: first.answer ?? null,
        "keywords/variations": first["keywords / variations"] ?? first["keywords-/-variations"] ?? null
      } : null
    });
  } catch (e) {
    return res.status(500).json({ ok:false, where:"kb-test-fatal", detail:String(e) });
  }
}
