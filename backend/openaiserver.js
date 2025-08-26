import axios from "axios";
import express from "express";
import cors from "cors";
import dotenv from "dotenv";
dotenv.config();

const app = express();
app.use(cors());
app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ extended: true, limit: "50mb" }));

const API_KEY = process.env.OPENAI_API_KEY;

const sanitizePrompt = (prompt) => {
  const noImagePrompt = prompt.replace(/<img[^>]+src="data:image\/[^">]+"[^>]*>/g, '[image]');
  const words = noImagePrompt.split(/\s+/);
  return words.length > 1000 ? words.slice(0, 1000).join(" ") : noImagePrompt;
};

app.get("/", (req, res) => {
  res.send("OpenAI proxy server is running.");
});

app.post("/openai-api/openai", async (req, res) => {
  const { prompt } = req.body;

  try {
    const trimmedPrompt = sanitizePrompt(prompt);

    const response = await axios.post(
      "https://api.openai.com/v1/chat/completions",
      {
        model: "gpt-3.5-turbo",
        messages: [{ role: "user", content: trimmedPrompt }],
        max_tokens: 300,
      },
      {
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${API_KEY}`,
        },
      }
    );

    res.json(response.data);
  } catch (error) {
    console.error("OpenAI Error:", error.response?.data || error.message);
    res.status(500).json({
      error: error.response?.data?.error?.message || "Something went wrong",
    });
  }
});

app.listen(3001, () => console.log('Server running on http://localhost:3001'));
