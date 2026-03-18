import { AIProcessor } from "./aiProcessor.js";

const aiServiceInstance = new AIProcessor();

const errors = await aiServiceInstance.checkRequirements();
if (errors) {
  view.showError(errors);
  // continue: allow form to work with fallback parameters
}

/**
 * Summarize text with options: language (pt/en), format (summary/points/actions)
 * Returns an async generator for streaming chunks
 */
export async function* summarize(text, options) {
  const { language = "pt", format = "summary" } = options || {};

  // Compose prompt based on format and language
  let promptText = "";
  if (language === "pt") {
    if (format === "summary") {
      promptText = `Resuma o texto abaixo em 5 linhas:`;
    } else if (format === "points") {
      promptText = `Liste os pontos principais do texto abaixo:`;
    } else if (format === "actions") {
      promptText = `Liste as ações sugeridas baseadas no texto abaixo:`;
    }
  } else if (language === "en") {
    if (format === "summary") {
      promptText = `Summarize the text below in 5 lines:`;
    } else if (format === "points") {
      promptText = `List the key points of the text below:`;
    } else if (format === "actions") {
      promptText = `List the action items based on the text below:`;
    }
  }

  const fullPrompt = `${promptText}\n\n${text}`;

  // Use AIService to create session and stream response
  for await (const chunk of aiServiceInstance.createSession(
    fullPrompt,
    0.5,
    5,
    null,
    options,
  )) {
    yield chunk;
  }
}
