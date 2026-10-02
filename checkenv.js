import "dotenv/config";

const keys = (process.env.GEMINI_API_KEYS || "")
  .split(",")
  .map((k) => k.trim())
  .filter(Boolean);

console.log("So key doc duoc:", keys.length);
keys.forEach((k, i) => {
  console.log(`Key ${i} | do dai: ${k.length} | 6 ky tu dau: ${k.slice(0, 6)}`);
});