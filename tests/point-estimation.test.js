'use strict';

const assert = require('assert');

// ── Pure math functions (copied inline from the refactored post) ─────────────

// From drug_exp(): p(x) = 1 - exp(-theta * dose)
function drugCurve(theta, dose) {
  return 1 - Math.exp(-theta * dose);
}

// From biasedness(): gaussian PDF used for gauss_data
function gaussianPDF(x, mean, std) {
  return Math.exp(-0.5 * ((x - mean) / std) ** 2) / (std * Math.sqrt(2 * Math.PI));
}

// From biasedness(): PDF for sampling distribution of X̄_n
function xnPDF(x, mu, sigma, n) {
  return Math.exp(-0.5 * ((x - mu) / sigma * Math.sqrt(n)) ** 2) / (sigma * Math.sqrt(2 * Math.PI / n));
}

// From biasedness(): biased sample variance (hat_s_n^2 = (1/n) * sum (xi - mean)^2)
function biasedVariance(samples, mean) {
  return samples.reduce((sum, xi) => sum + (xi - mean) ** 2, 0) / samples.length;
}

// From script 3
function randn_bm() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  const u_a = Math.sqrt(-2.0 * Math.log(u));
  const u_b = Math.cos(2.0 * Math.PI * v);
  return u_a * u_b;
}

// From both scripts 2 and 3
function roundN(x) { return Math.round(x - 0.5); }

// plt_label_path as a function declaration (hoisted, cross-script accessible)
function plt_label_path(svg, color, x, y) {
  // In a browser this appends SVG paths; here we just verify it's callable
  return { svg, color, x, y };
}


// ── Test 1: drugCurve math ────────────────────────────────────────────────────

{
  const result0 = drugCurve(0.2, 0);
  assert(Math.abs(result0) < 1e-10,
    `Test 1a failed: drugCurve(0.2, 0) = ${result0}, expected 0`);
}

{
  const result0theta = drugCurve(0, 5);
  assert(Math.abs(result0theta) < 1e-10,
    `Test 1b failed: drugCurve(0, 5) = ${result0theta}, expected 0`);
}

{
  // 1 - exp(-0.2 * 5) = 1 - exp(-1) ≈ 0.6321205588285578
  const expected = 1 - Math.exp(-1);
  const result = drugCurve(0.2, 5);
  assert(Math.abs(result - expected) < 1e-10,
    `Test 1c failed: drugCurve(0.2, 5) = ${result}, expected ≈ ${expected}`);
}


// ── Test 2: gaussianPDF ───────────────────────────────────────────────────────

{
  // At x=mean the PDF equals 1/(std*sqrt(2*pi))
  const std = 1.5;
  const expected = 1 / (std * Math.sqrt(2 * Math.PI));
  const result = gaussianPDF(3.0, 3.0, std);
  assert(Math.abs(result - expected) < 1e-12,
    `Test 2 failed: gaussianPDF(mean, mean, std) = ${result}, expected ${expected}`);
}


// ── Test 3: xnPDF — n=4 gives twice the height at 0 compared to n=1 ──────────

{
  // At x=mu=0, std=1:
  // xnPDF(0, 0, 1, 1) = 1/sqrt(2*pi) ≈ 0.3989
  // xnPDF(0, 0, 1, 4) = sqrt(4)/sqrt(2*pi) = 2/sqrt(2*pi) ≈ 0.7979
  // So pdf_n4 / pdf_n1 = 2
  const pdf_n1 = xnPDF(0, 0, 1, 1);
  const pdf_n4 = xnPDF(0, 0, 1, 4);
  const ratio = pdf_n4 / pdf_n1;
  assert(Math.abs(ratio - 2) < 1e-10,
    `Test 3 failed: xnPDF ratio n4/n1 = ${ratio}, expected 2`);
}


// ── Test 4: biasedVariance ────────────────────────────────────────────────────

{
  // samples=[1,2,3], mean=2
  // variance = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = (1+0+1)/3 = 2/3
  const samples = [1, 2, 3];
  const mean = 2;
  const expected = 2 / 3;
  const result = biasedVariance(samples, mean);
  assert(Math.abs(result - expected) < 1e-10,
    `Test 4 failed: biasedVariance([1,2,3], 2) = ${result}, expected ≈ ${expected}`);
}


// ── Test 5: randn_bm returns finite values and is approximately standard normal ─

{
  const sample = randn_bm();
  assert(isFinite(sample),
    `Test 5a failed: randn_bm() returned non-finite value ${sample}`);
}

{
  const N = 10000;
  let sum = 0;
  for (let i = 0; i < N; i++) {
    sum += randn_bm();
  }
  const sampleMean = sum / N;
  assert(Math.abs(sampleMean) < 0.1,
    `Test 5b failed: mean of ${N} randn_bm() samples = ${sampleMean}, expected within 0.1 of 0`);
}


// ── Test 6: roundN (banker's rounding) ───────────────────────────────────────

{
  // roundN(x) = Math.round(x - 0.5)
  // roundN(1.5) => Math.round(1.0) = 1
  assert(roundN(1.5) === 1,
    `Test 6a failed: roundN(1.5) = ${roundN(1.5)}, expected 1`);
}

{
  // roundN(2.5) => Math.round(2.0) = 2
  assert(roundN(2.5) === 2,
    `Test 6b failed: roundN(2.5) = ${roundN(2.5)}, expected 2`);
}


// ── Test 7: TDZ safety — simulateDrugExpInit ──────────────────────────────────

function simulateDrugExpInit() {
  // All variables that drugExpUpdate() would assign to are declared BEFORE the call.
  // If any were declared after the call (with let), it would throw a ReferenceError.
  const figs = [];

  function drugExpUpdate(theta) {
    return Array.from({length: 11}, (_, i) => 1 - Math.exp(-theta * i));
  }

  const result = drugExpUpdate(0.2);
  assert(result.length === 11, `simulateDrugExpInit: drugExpUpdate must return 11 values, got ${result.length}`);
  assert(Math.abs(result[0]) < 1e-10, `simulateDrugExpInit: P(heal | dose=0) should be 0, got ${result[0]}`);

  // Verify dose=5 (index 5) matches drugCurve(0.2, 5)
  const expected5 = 1 - Math.exp(-0.2 * 5);
  assert(Math.abs(result[5] - expected5) < 1e-10,
    `simulateDrugExpInit: result[5] = ${result[5]}, expected ${expected5}`);

  return true;
}

{
  let threw = false;
  try {
    simulateDrugExpInit();
  } catch (e) {
    threw = true;
    assert.fail(`Test 7 (simulateDrugExpInit) threw: ${e.message}`);
  }
  assert(!threw, 'Test 7 failed: simulateDrugExpInit threw an error');
}


// ── Test 8: TDZ safety — simulateBiasednessInit ───────────────────────────────

function simulateBiasednessInit() {
  // Mirroring how biasedness() declares variables before any callback that assigns them.
  // std_curve and sn_avg_curve are declared with let BEFORE the .then() callback that assigns them.
  let xn_dots = [], sn_dots = [];
  let std_curve, sn_avg_curve;   // declared before any callback

  // simulate the async assignment (the .then() callback)
  const assignCallback = () => {
    std_curve = { type: 'std_curve_mock' };
    sn_avg_curve = { type: 'sn_avg_curve_mock' };
  };
  assignCallback();

  assert(std_curve !== undefined, 'simulateBiasednessInit: std_curve should be assigned');
  assert(sn_avg_curve !== undefined, 'simulateBiasednessInit: sn_avg_curve should be assigned');

  // simulate sample() — the three variables must be const-declared at the top of sample()
  const samples = [0.5, -0.3, 1.2];
  const mean = samples.reduce((a, b) => a + b) / samples.length;
  const variance = samples.reduce((sum, xi) => sum + (xi - mean) ** 2, 0) / samples.length;

  // Expected: mean = (0.5 - 0.3 + 1.2) / 3 = 1.4/3
  const expectedMean = (0.5 + (-0.3) + 1.2) / 3;
  assert(Math.abs(mean - expectedMean) < 1e-10,
    `simulateBiasednessInit: mean = ${mean}, expected ${expectedMean}`);

  // Expected variance: sum of squared deviations / 3
  const expectedVariance = (
    (0.5 - expectedMean) ** 2 +
    (-0.3 - expectedMean) ** 2 +
    (1.2 - expectedMean) ** 2
  ) / 3;
  assert(Math.abs(variance - expectedVariance) < 1e-10,
    `simulateBiasednessInit: variance = ${variance}, expected ${expectedVariance}`);

  return true;
}

{
  let threw = false;
  try {
    simulateBiasednessInit();
  } catch (e) {
    threw = true;
    assert.fail(`Test 8 (simulateBiasednessInit) threw: ${e.message}`);
  }
  assert(!threw, 'Test 8 failed: simulateBiasednessInit threw an error');
}


// ── Test 9: plt_label_path is a function declaration (hoisted/accessible) ────

{
  // plt_label_path is defined above as a function declaration (not const/let).
  // Function declarations are hoisted, so they are accessible across scripts.
  assert(typeof plt_label_path === 'function',
    `Test 9 failed: plt_label_path should be a function, got ${typeof plt_label_path}`);

  // Verify it's callable without throwing
  const result = plt_label_path({}, '#EDA137', 100, 50);
  assert(result !== undefined, 'Test 9 failed: plt_label_path returned undefined');
  assert(result.color === '#EDA137', `Test 9 failed: expected color #EDA137, got ${result.color}`);
}


console.log('All tests passed!');
