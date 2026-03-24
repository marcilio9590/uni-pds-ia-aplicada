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

test("command upper transforms message into lowerCase", async () => {
  const app = createServer();
  const message = "MAKE THIS MESSAGE LOWER CASE!";
  const expect = message.toLowerCase();
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

test("command upper transforms message into UNKNOW", async () => {
  const app = createServer();
  const message = "Hey There!";
  const expect =
    "Unknow command. Try 'make this uppercase' or 'convert to lowercase'";
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
