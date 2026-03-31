import { RemoveMessage } from "@langchain/core/messages";
import type { Runtime } from "@langchain/langgraph";
import { HumanMessage } from "langchain";
import {
  type ConversationSummary,
  getSummarizationSystemPrompt,
  getSummarizationUserPrompt,
  SummarySchema,
} from "../../prompts/v1/summarization.ts";
import { OpenRouterService } from "../../services/openrouterService.ts";
import { PreferencesService } from "../../services/preferencesService.ts";
import type { GraphState } from "../graph.ts";

export function createSummarizationNode(
  llmClient: OpenRouterService,
  preferencesService: PreferencesService,
) {
  return async (
    state: GraphState,
    runtime?: Runtime,
  ): Promise<Partial<GraphState>> => {
    const conversationHistory = state.messages.map((msg) => ({
      role: HumanMessage.isInstance(msg) ? "User" : "AI",
      content: msg.text,
    }));

    const previousSummary = state.conversationSummary as
      | ConversationSummary
      | undefined;

    const systemPromp = getSummarizationSystemPrompt();
    const userPrompt = getSummarizationUserPrompt(
      conversationHistory,
      previousSummary,
    );

    const result = await llmClient.generateStructured(
      systemPromp,
      userPrompt,
      SummarySchema,
    );
    if (result.error || !result.data) {
      console.log("Erro ao fazer resumo", result.error);
      return {
        needsSummarization: false,
      };
    }

    const userId = String(
      runtime?.context?.userId || state.userId || "unknown",
    );

    preferencesService.storeSummary(userId, result.data);

    const deleteMessage = state.messages
      .slice(0, -2)
      .map((m) => new RemoveMessage({ id: module.id as string }));
    
    return {
      messages: deleteMessage,
      conversationSummary: result.data,
      needsSummarization: false,
    };
  };
}
