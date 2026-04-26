import { MultiServerMCPClient } from "@langchain/mcp-adapters";
import { getCSVTOJSONTool } from "../tools/csvToJsonTool.ts";
import { getFsTool } from "../tools/fsTool.ts";
import { getMongoDbTool } from "../tools/mongodbTool.ts";

export const getMCPTools = async () => {
  const client = new MultiServerMCPClient({
    mcpServers: {
      ...getMongoDbTool(),
      ...getFsTool(),
    },
    onMessage: (log, source) => {
      console.log(`[${source.server}] [${log.data}]`);
    },
  });
  const mCPTools = await client.getTools();

  return [...mCPTools, getCSVTOJSONTool()];
};
