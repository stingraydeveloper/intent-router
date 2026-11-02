// The web demo must agree with the Python model it was exported from.
// node --test tests/browser_parity.test.mjs
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { classify } from "../web/router.js";

const read = (path) => JSON.parse(readFileSync(new URL(path, import.meta.url)));
const model = read("../results/browser_model.json");
const fixture = read("../results/browser_parity.json");

test("browser predictions match Python on 400 test messages", () => {
  let same = 0;
  let drift = 0;
  for (const row of fixture) {
    const [best] = classify(model, row.text);
    if (best.label === row.label) same++;
    drift = Math.max(drift, Math.abs(best.p - row.conf));
  }
  assert.equal(same, fixture.length);
  assert.ok(drift < 0.01);
});
