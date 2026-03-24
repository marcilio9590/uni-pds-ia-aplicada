import { END, MessagesZodMeta, START, StateGraph } from "@langchain/langgraph";
import { withLangGraph } from "@langchain/langgraph/zod";
import { BaseMessage } from "langchain";
import { z } from "zod/v3";
import { chatResponse } from "./nodes/chatResponseNode.ts";
import { fallbackNode } from "./nodes/fallbackNode.ts";
import { identifyIntent } from "./nodes/identifyIntentNode.ts";
import { lowerCaseNode } from "./nodes/lowerCaseNode.ts";
import { upperCaseNode } from "./nodes/upperCaseNode.ts";

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
    .addNode("identifyIntent", identifyIntent)
    .addNode("upperCaseNode", upperCaseNode)
    .addNode("lowerCaseNode", lowerCaseNode)
    .addNode("chatResponse", chatResponse)
    .addNode("fallbackNode", fallbackNode)
    // .addNode("identifyIntenet", (state, GraphState) => {
    //   return {
    //     ...state,
    //     output: "Teste",
    //   };
    // })
    .addEdge(START, "identifyIntent")
    .addConditionalEdges(
      "identifyIntent",
      (state: GraphState) => {
        switch (state.command) {
          case "uppercase":
            return "uppercase";
          case "lowercase":
            return "lowercase";
          default:
            return "fallback";
        }
      },
      {
        uppercase: "upperCaseNode",
        lowercase: "lowerCaseNode",
        fallback: "fallbackNode",
      },
    )
    .addEdge("upperCaseNode", "chatResponse")
    .addEdge("lowerCaseNode", "chatResponse")
    .addEdge("fallbackNode", "chatResponse")
    .addEdge("chatResponse", END);
  return workflow.compile();
}
