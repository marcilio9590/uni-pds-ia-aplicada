import { CONFIG } from "./config.ts";
import { DocumentProcessor } from "./documentProcessor.ts";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import { type PretrainedOptions } from "@huggingface/transformers";
import { Neo4jVectorStore } from "@langchain/community/vectorstores/neo4j_vector";
import { displayResults } from "./util.ts";

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
  const quetions = [
    "Oque são tensores e como são representados em javascript?",
    "Oque é normalização de dados e por que é necessária?",
    "Como funciona uma rede neural no tensorFlow.js?",
    "Oque significa treinar uma rede neural?",
  ];

  for (const question of quetions) {
    console.log(`Pergunta: ${question}`);
    const results = await _neo4jVectorStore.similaritySearch(
      question,
      CONFIG.similarity.topK,
    );
    console.log(displayResults(results));
  }
  console.log("End - Buscando a similaridade");
} catch (error) {
  console.log("ERROR", error);
} finally {
  await _neo4jVectorStore?.close();
}
