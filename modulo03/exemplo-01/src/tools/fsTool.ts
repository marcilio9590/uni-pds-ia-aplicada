export const getFsTool = () => {
  return {
    filesystem: {
      command: "npx",
      transport: "stdio" as const,
      args: [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        `${process.cwd()}/reports`,
      ],
    },
  };
};
