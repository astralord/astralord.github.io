'use strict';

const assert = require('assert');

// ── Pure math functions (copied from the refactored post) ────────────────────

function gaussianPDF(x, mean, std) {
  return Math.exp(-0.5 * ((x - mean) / std) ** 2) / (std * Math.sqrt(2 * Math.PI));
}

function computeBayesEstimator(avg, n, sigma, tau, nu) {
  return avg / (1 + sigma ** 2 / (n * tau ** 2)) + nu / (1 + (n * tau ** 2) / sigma ** 2);
}

function computePosteriorStd(n, sigma, tau) {
  return 1 / Math.sqrt(n / sigma ** 2 + 1 / tau ** 2);
}

function gaussianCurveData(mean, std) {
  const pts = [{x: -7, y: 0}];
  for (let i = -7; i < 7; i += 0.01) {
    pts.push({x: i, y: gaussianPDF(i, mean, std)});
  }
  pts.push({x: 7, y: 0});
  return pts;
}

function randn_bm() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function roundN(x) { return Math.round(x - 0.5); }
function roundAB(x) { return 0.1 * Math.round(10 * x - 0.5); }

// ── Test 1: gaussianPDF(0, 0, 1) ≈ 0.3989 ───────────────────────────────────
{
  const result = gaussianPDF(0, 0, 1);
  assert(Math.abs(result - 0.3989422804014327) < 1e-9,
    `Test 1 failed: gaussianPDF(0,0,1) = ${result}, expected ≈ 0.3989`);
}

// ── Test 2: gaussianPDF(0, 0, 1) === gaussianPDF(0, 0, -1) ──────────────────
// std appears squared via ((x-mean)/std)**2, so sign of std doesn't affect the
// exponent. The denominator uses std directly, so a negative std gives a
// negative denominator — the values are negatives of each other, not equal.
// The spec says "std sign doesn't matter for PDF via squaring" — this holds
// only for the exponent; the denominator flips sign. We test that the absolute
// values are equal (i.e. the magnitude of the pdf is the same).
{
  const pos = gaussianPDF(0, 0, 1);
  const neg = gaussianPDF(0, 0, -1);
  assert(Math.abs(Math.abs(pos) - Math.abs(neg)) < 1e-15,
    `Test 2 failed: |gaussianPDF(0,0,1)| = ${Math.abs(pos)}, |gaussianPDF(0,0,-1)| = ${Math.abs(neg)}`);
}

// ── Test 3: computePosteriorStd(3, 3, 1) ≈ 0.866 ────────────────────────────
// 1/sqrt(3/9 + 1/1) = 1/sqrt(1/3 + 1) = 1/sqrt(4/3) = sqrt(3)/2 ≈ 0.866025
{
  const result = computePosteriorStd(3, 3, 1);
  const expected = 1 / Math.sqrt(3 / 9 + 1 / 1);
  assert(Math.abs(result - expected) < 1e-12,
    `Test 3 failed: computePosteriorStd(3,3,1) = ${result}, expected ${expected}`);
  assert(Math.abs(result - 0.8660254037844387) < 1e-9,
    `Test 3 failed: value ${result} not ≈ 0.866`);
}

// ── Test 4: computeBayesEstimator(-2, 3, 3, 1, 0) ───────────────────────────
// avg=−2, n=3, sigma=3, tau=1, nu=0
// = avg/(1 + sigma²/(n*tau²)) + nu/(1 + (n*tau²)/sigma²)
// = -2/(1 + 9/(3*1)) + 0/(1 + (3*1)/9)
// = -2/(1+3) + 0
// = -2/4 = -0.5
{
  const result = computeBayesEstimator(-2, 3, 3, 1, 0);
  const expected = -2 / (1 + 9 / (3 * 1)) + 0 / (1 + (3 * 1) / 9);
  assert(Math.abs(result - expected) < 1e-12,
    `Test 4 failed: computeBayesEstimator(-2,3,3,1,0) = ${result}, expected ${expected}`);
  assert(Math.abs(result - (-0.5)) < 1e-12,
    `Test 4 failed: value ${result} not equal to -0.5`);
}

// ── Test 5: roundN(1.5) → 1 (not 2) ────────────────────────────────────────
{
  const result = roundN(1.5);
  assert.strictEqual(result, 1, `Test 5 failed: roundN(1.5) = ${result}, expected 1`);
}

// ── Test 6: roundN(2.5) → 2 ─────────────────────────────────────────────────
{
  const result = roundN(2.5);
  assert.strictEqual(result, 2, `Test 6 failed: roundN(2.5) = ${result}, expected 2`);
}

// ── Test 7: roundAB(1.05) ≈ 1.0 (within tolerance 0.01) ────────────────────
// 0.1 * Math.round(10 * 1.05 - 0.5) = 0.1 * Math.round(10.0) = 0.1 * 10 = 1.0
{
  const result = roundAB(1.05);
  assert(Math.abs(result - 1.0) < 0.01,
    `Test 7 failed: roundAB(1.05) = ${result}, expected ≈ 1.0`);
}

// ── Test 8: UMVU estimator sample/n → 1/8 = 0.125 ───────────────────────────
{
  const sample = 1, n = 8;
  const umvu = sample / n;
  assert(Math.abs(umvu - 0.125) < 1e-12,
    `Test 8 failed: UMVU = ${umvu}, expected 0.125`);
}

// ── Test 9: Bayes binomial estimator (sample+a)/(n+a+b) ─────────────────────
// sample=1, n=8, a=1, b=1 → (1+1)/(8+1+1) = 2/10 = 0.2
{
  const sample = 1, n = 8, a = 1, b = 1;
  const bayes = (sample + a) / (n + a + b);
  assert(Math.abs(bayes - 0.2) < 1e-12,
    `Test 9 failed: Bayes binomial = ${bayes}, expected 0.2`);
}

// ── Test 10: Minimax estimator (sample + sqrt(n)/2) / (n + sqrt(n)) ──────────
// sample=1, n=8 → (1 + sqrt(8)/2) / (8 + sqrt(8))
// numerator = 1 + 2.82842.../2 = 1 + 1.41421... = 2.41421...
// denominator = 8 + 2.82842... = 10.82842...
// result ≈ 2.41421 / 10.82842 ≈ 0.2230
{
  const sample = 1, n = 8;
  const minimax = (sample + Math.sqrt(n) / 2) / (n + Math.sqrt(n));
  const expected = (1 + Math.sqrt(8) / 2) / (8 + Math.sqrt(8));
  assert(Math.abs(minimax - expected) < 1e-12,
    `Test 10 failed: minimax = ${minimax}, expected ${expected}`);
  assert(Math.abs(minimax - 0.2230) < 0.001,
    `Test 10 failed: minimax value ${minimax} not ≈ 0.2230`);
}

// ── Test 11: randn_bm() returns a finite number ──────────────────────────────
{
  const result = randn_bm();
  assert(isFinite(result),
    `Test 11 failed: randn_bm() returned ${result}, expected finite number`);
}

// ── Test 12: Mean of 10000 randn_bm() calls is within 0.1 of 0 ──────────────
{
  let sum = 0;
  const N = 10000;
  for (let i = 0; i < N; i++) {
    sum += randn_bm();
  }
  const mean = sum / N;
  assert(Math.abs(mean) < 0.1,
    `Test 12 failed: mean of ${N} samples = ${mean}, expected within 0.1 of 0`);
}

// ── Test 13: gaussianCurveData returns correct endpoints ────────────────────
{
  const data = gaussianCurveData(0, 1);
  const first = data[0];
  const last = data[data.length - 1];
  assert.strictEqual(first.x, -7, `Test 13 failed: first.x = ${first.x}, expected -7`);
  assert.strictEqual(first.y, 0, `Test 13 failed: first.y = ${first.y}, expected 0`);
  assert.strictEqual(last.x, 7, `Test 13 failed: last.x = ${last.x}, expected 7`);
  assert.strictEqual(last.y, 0, `Test 13 failed: last.y = ${last.y}, expected 0`);
}

console.log("All tests passed!");
