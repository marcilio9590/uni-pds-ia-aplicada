import { END, MessagesZodMeta, START, StateGraph } from "@langchain/langgraph";
import { withLangGraph } from "@langchain/langgraph/zod";
import { BaseMessage } from "langchain";
import { z } from "zod/v3";
import { chatResponse } from "./nodes/chatResponseNode.ts";
import { identifyIntent } from "./nodes/identifyIntentNode.ts";

const GraphState = z.object({
  messages: withLangGraph(z.custom<BaseMessage[]>(), MessagesZodMeta),
  output: z.string(),
  command: z.enum(["uppercase", "lowercase", "unknown"]),
});

export type GraphState = z.infer<typeof GraphState>;

export function buildGraph() {
  const workflow = new StateGraph({
    stateSchema: GraphState,
  })
    .addNode("identifyIntenet", identifyIntent)
    .addNode("chatResponse", chatResponse)
    // .addNode("identifyIntenet", (state, GraphState) => {
    //   return {
    //     ...state,
    //     output: "Teste",
    //   };
    // })
    .addEdge(START, "identifyIntenet")
    .addEdge("identifyIntenet", "chatResponse")
    .addEdge("chatResponse", END);
  return workflow.compile();
}
