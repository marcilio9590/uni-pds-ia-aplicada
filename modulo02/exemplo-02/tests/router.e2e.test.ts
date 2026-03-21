import assert from "node:assert/strict";
import { test } from "node:test";
import { createServer } from "../src/server.ts";

test("command upper transforms message into UPPERCASE", async () => {
  const app = createServer();
  const message = "make this message UPPER please!";
  const expect = message.toUpperCase();
  const response = await app.inject({
    method: "POST",
    url: "/chat",
    body: {
      question: message,
    },
  });

  assert.equal(response.statusCode, 200);
  assert.equal(response.body, expect);
});
