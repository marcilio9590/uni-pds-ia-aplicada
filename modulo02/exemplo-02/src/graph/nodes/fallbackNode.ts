import { AIMessage } from "langchain";
import { type GraphState } from "../graph.ts";

export function fallbackNode(state: GraphState): GraphState {
  const message =
    "Unknow command. Try 'make this uppercase' or 'convert to lowercase'";
  const aiMessage = new AIMessage(message).content.toString();
  return {
    ...state,
    output: message,
    messages: [...state.messages],
  };
}
