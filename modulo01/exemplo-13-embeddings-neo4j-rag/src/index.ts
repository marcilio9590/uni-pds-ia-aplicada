import { type PretrainedOptions } from "@huggingface/transformers";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import { Neo4jVectorStore } from "@langchain/community/vectorstores/neo4j_vector";
import { ChatOpenAI } from "@langchain/openai";
import { AI } from "./ai.ts";
import { CONFIG } from "./config.ts";
import { DocumentProcessor } from "./documentProcessor.ts";
import { writeFile, mkdir } from "node:fs/promises";

let _neo4jVectorStore = null;
async function clearAll(vectorStore: Neo4jVectorStore, nodeLabel: string) {
  console.log("Removendo todos os documentos existentes");
  await vectorStore.query(`MATCH (n: \`${nodeLabel}\`) DETACH DELETE n`);
  console.log("Documentos removidos com sucesso");
}

try {
  console.log("Inicializando sistema de embeddings com neo4j");

  const documentProcessor = new DocumentProcessor(
    CONFIG.pdf.path,
    CONFIG.textSplitter,
  );
  const documents = await documentProcessor.loadAndSplit();
  const embeddings = new HuggingFaceTransformersEmbeddings({
    model: CONFIG.embedding.modelName,
    pretrainedOptions: CONFIG.embedding.pretrainedOptions as PretrainedOptions,
  });

  const nlpModel = new ChatOpenAI({
    temperature: CONFIG.openRouter.temperature,
    maxRetries: CONFIG.openRouter.maxRetries,
    modelName: CONFIG.openRouter.nlpModel,
    openAIApiKey: CONFIG.openRouter.apiKey,
    configuration: {
      baseURL: CONFIG.openRouter.url,
      defaultHeaders: CONFIG.openRouter.defaultHeaders,
    },
  });

  _neo4jVectorStore = await Neo4jVectorStore.fromExistingGraph(
    embeddings,
    CONFIG.neo4j,
  );
  clearAll(_neo4jVectorStore, CONFIG.neo4j.nodeLabel);

  for (const [index, doc] of documents.entries()) {
    console.log(`Adicionando documento ${index + 1}/${documents.length}`);
    await _neo4jVectorStore.addDocuments([doc]);
  }
  console.log("Base de dados populada com sucesso");

  console.log("Start - Buscando a similaridade");
  const questions = [
    "Oque são tensores e como são representados em javascript?",
    "Oque é normalização de dados e por que é necessária?",
    "Como funciona uma rede neural no tensorFlow.js?",
    "Oque significa treinar uma rede neural?",
  ];

  const ai = new AI({
    nlpModel,
    debugLog: console.log,
    vectorStore: _neo4jVectorStore,
    promptConfig: CONFIG.promptConfig,
    templateText: CONFIG.templateText,
    topK: CONFIG.similarity.topK,
  });

  for (const index in questions) {
    const question = questions[index];
    // console.log(`Pergunta: ${question}`);
    const result = await ai.answerQuestion(question!);
    if (result.error) {
      console.log(`Erro: ${result.error}`);
      continue;
    }
    console.log(result.answer);
    await mkdir(CONFIG.output.answerFolder, { recursive: true });
    const fileName = `${CONFIG.output.answerFolder}/${CONFIG.output.fileName}-${index}-${Date.now()}.md`;
    await writeFile(fileName, result.answer!);
  }
  console.log("End - Buscando a similaridade");
} catch (error) {
  console.log("ERROR", error);
} finally {
  await _neo4jVectorStore?.close();
}
