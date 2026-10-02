import "dotenv/config";
import { GoogleGenAI } from "@google/genai";

const key = process.env.GEMINI_API_KEYS.split(",")[0].trim();
const ai = new GoogleGenAI({ apiKey: key });
const pager = await ai.models.list();
for await (const m of pager) {
  if (!m.supportedActions || m.supportedActions.includes("generateContent")) {
    console.log(m.name.replace("models/", ""));
  }
}