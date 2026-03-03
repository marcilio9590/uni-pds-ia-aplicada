export class AIProcessor {
  constructor() {
    this.session = null;
    this.abortController = null;
    this.translator = null;
  }

  async checkRequirements() {
    const errors = [];

    // @ts-ignore
    const isChrome = !!window.chrome;
    if (!isChrome) {
      errors.push(
        "⚠️ Este recurso só funciona no Google Chrome ou Chrome Canary (versão recente).",
      );
    }

    if (!("LanguageModel" in self)) {
      errors.push("⚠️ As APIs nativas de IA não estão ativas.");
      errors.push("Ative a seguinte flag em chrome://flags/:");
      errors.push(
        "- Prompt API for Gemini Nano (chrome://flags/#prompt-api-for-gemini-nano)",
      );
      errors.push("Depois reinicie o Chrome e tente novamente.");
      return errors;
    }

    // Check Translator availability
    if ("Translator" in self) {
      const translatorAvailability = await Translator.availability({
        sourceLanguage: "en",
        targetLanguage: "pt",
      });
      console.log("Translator Availability:", translatorAvailability);

      if (translatorAvailability === "no") {
        errors.push(
          "⚠️ Tradução de inglês para português não está disponível.",
        );
      }
    } else {
      errors.push("⚠️ A API de Tradução não está ativa.");
      errors.push("Ative a seguinte flag em chrome://flags/:");
      errors.push("- Translation API (chrome://flags/#translation-api)");
    }

    // Check Language Detection API
    if (!("LanguageDetector" in self)) {
      errors.push("⚠️ A API de Detecção de Idioma não está ativa.");
      errors.push("Ative a seguinte flag em chrome://flags/:");
      errors.push(
        "- Language Detection API (chrome://flags/#language-detector-api)",
      );
    }
    this.languageDetector = await LanguageDetector.create();

    if (errors.length > 0) {
      return errors;
    }

    const availability = await LanguageModel.availability({
      languages: ["en"],
    });
    console.log("Language Model Availability:", availability);

    if (availability === "available") {
      console.log("Language Model:", availability);
    }

    if (availability === "unavailable") {
      errors.push(
        `⚠️ O seu dispositivo não suporta modelos de linguagem nativos de IA.`,
      );
    }

    if (availability === "downloading") {
      errors.push(
        `⚠️ O modelo de linguagem de IA está sendo baixado. Por favor, aguarde alguns minutos e tente novamente.`,
      );
    }

    if (availability === "downloadable") {
      errors.push(
        `⚠️ O modelo de linguagem de IA precisa ser baixado, baixando agora... (acompanhe o progresso no terminal do chrome)`,
      );
      console.log("Language Model:", availability);
    }

    return errors.length > 0 ? errors : null;
  }

  async translate(text, lang) {
    try {
      let detectedLanguage = null;

      // Detect language first
      if (this.languageDetector) {
        const detectionResults = await this.languageDetector.detect(text);
        console.log("Detected languages:", detectionResults);

        detectedLanguage = detectionResults[0]?.detectedLanguage;

        // If already in Portuguese, no need to translate
        if (detectedLanguage === "pt") {
          console.log("Text is already in Portuguese");
          return text;
        }
      }

      // 🔎 Verifica se precisa recriar o translator
      if (!this.translator || this.currentTargetLanguage !== lang) {
        console.log("Creating new translator instance...");

        this.translator = await Translator.create({
          sourceLanguage: detectedLanguage,
          targetLanguage: lang,
        });

        // Salva qual idioma está sendo usado
        this.currentTargetLanguage = lang;
      }

      const translated = await this.translator.translate(text);

      console.log("Translated text:", translated);
      return translated;
    } catch (error) {
      console.error("Translation error:", error);
      return text;
    }
  }

  async getParams() {
    const params = await LanguageModel.params();
    console.log("Language Model Params:", params);
    return params;
  }

  async *createSession(question, temperature, topK, file = null, options) {
    this.abortController?.abort();
    this.abortController = new AbortController();
    const lang = options?.language || "en";

    // Destroy previous session and create new one with updated parameters
    if (this.session) {
      this.session.destroy();
    }

    this.session = await LanguageModel.create({
      expectedInputs: [
        { type: "text", languages: ["en"] },
        { type: "audio" },
        { type: "image" },
      ],
      expectedOutputs: [{ type: "text", languages: ["en"] }],
      temperature: temperature,
      topK: topK,
      initialPrompts: [
        {
          role: "system",
          content: [
            {
              type: "text",
              value: `You are an AI assistant that responds clearly and objectively.
                        Always respond in plain text format instead of markdown.`,
            },
          ],
        },
      ],
    });

    // Build content array with text and optional file
    const contentArray = [{ type: "text", value: question }];

    if (file) {
      const fileType = file.type.split("/")[0];
      if (fileType === "image" || fileType === "audio") {
        // Convert file to blob for proper handling
        const blob = new Blob([await file.arrayBuffer()], { type: file.type });
        contentArray.push({ type: fileType, value: blob });
        console.log(`Adding ${fileType} to prompt:`, file.name);
      }
    }

    const responseStream = await this.session.promptStreaming(
      [
        {
          role: "user",
          content: contentArray,
        },
      ],
      {
        signal: this.abortController.signal,
      },
    );

    // Accumulate all chunks and translate only the final complete text (avoid translating chunk-by-chunk)
    let accumulated = "";
    for await (const chunk of responseStream) {
      if (this.abortController.signal.aborted) {
        break;
      }

      accumulated += chunk;
    }

    // If aborted, don't proceed with translation/yielding
    if (this.abortController.signal.aborted) {
      return;
    }

    // Preserve newlines by using a token so translation doesn't collapse them.
    const NEWLINE_TOKEN = '<<NL>>';
    const textWithTokens = accumulated.replace(/\r\n|\r|\n/g, NEWLINE_TOKEN);

    // Translate the text containing tokens
    const translatedWithTokens = await this.translate(textWithTokens, lang);

    // Restore newline characters
    const finalTranslated = translatedWithTokens.replace(new RegExp(NEWLINE_TOKEN, 'gi'), '\n');
    yield finalTranslated;
  }

  abort() {
    this.abortController?.abort();
  }

  isAborted() {
    return this.abortController?.signal.aborted;
  }
}
