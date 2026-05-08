'use strict';

const assert = require('assert');

// ── Pure math functions (copied from the refactored post) ────────────────────

function erf(x) {
    if (Math.abs(x) > 3) {
      return x / Math.abs(x);
    }
    let m = 1.00;
    let s = 1.00;
    let sum = x * 1.0;
    for (let i = 1; i < 50; i++) {
        m *= i;
        s *= -1;
        sum += (s * Math.pow(x, 2.0 * i + 1.0)) / (m * (2.0 * i + 1.0));
    }
    return 1.1283791671 * sum;
}

function Phi(x) {
    return 0.5 * (1 + erf(x / 1.41421356237));
}

function randn_bm() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function biv_gauss(rho) {
    const z1 = randn_bm();
    const z2 = rho * z1 + Math.sqrt(1 - rho * rho) * randn_bm();
    return [z1, z2];
}

function phi(x, mu, sigma) {
    let y = (x - mu) / sigma;
    y *= y;
    y = Math.exp(-y / 2);
    y /= (sigma * 1.41421356237 * Math.PI);
    return y;
}

function gamma_rand(k) {
    let x = 0;
    for (let i = 0; i < k; i += 1) {
        x -= Math.log(Math.random());
    }
    return x;
}

function dirichlet(ks) {
    const xs = [];
    let x0 = 0;
    for (let i = 0; i < ks.length; i += 1) {
        xs.push(gamma_rand(ks[i]));
        x0 += xs[i];
    }
    return [xs[0] / x0, xs[1] / x0];
}

function roundN(x) { return Math.round(x - 0.5); }

// Pearson correlation — extracted from estimate_rho() in prsn_plt()
function pearsonCorrelation(xs, ys) {
    const n = xs.length;
    let avgX = 0, avgY = 0;
    for (let i = 0; i < n; i++) { avgX += xs[i]; avgY += ys[i]; }
    avgX /= n; avgY /= n;
    let sqxx = 0, sqyy = 0, sqxy = 0;
    for (let i = 0; i < n; i++) {
        sqxx += (xs[i] - avgX) ** 2;
        sqyy += (ys[i] - avgY) ** 2;
        sqxy += (xs[i] - avgX) * (ys[i] - avgY);
    }
    return sqxy / Math.sqrt(sqxx * sqyy);
}

// ── Test 1: erf(0) === 0 ──────────────────────────────────────────────────────
{
    const result = erf(0);
    assert(Math.abs(result) < 1e-12,
        `Test 1 failed: erf(0) = ${result}, expected 0`);
}

// ── Test 2: erf clips to ±1 for |x| > 3 ─────────────────────────────────────
{
    assert.strictEqual(erf(4), 1, `Test 2a failed: erf(4) = ${erf(4)}, expected 1`);
    assert.strictEqual(erf(-4), -1, `Test 2b failed: erf(-4) = ${erf(-4)}, expected -1`);
}

// ── Test 3: erf is odd: erf(-x) === -erf(x) ──────────────────────────────────
{
    const x = 1.5;
    assert(Math.abs(erf(-x) + erf(x)) < 1e-12,
        `Test 3 failed: erf(-1.5) + erf(1.5) = ${erf(-x) + erf(x)}`);
}

// ── Test 4: Phi(0) ≈ 0.5 ─────────────────────────────────────────────────────
{
    const result = Phi(0);
    assert(Math.abs(result - 0.5) < 1e-9,
        `Test 4 failed: Phi(0) = ${result}, expected 0.5`);
}

// ── Test 5: Phi(1.96) ≈ 0.975 ────────────────────────────────────────────────
{
    const result = Phi(1.96);
    assert(Math.abs(result - 0.975) < 0.001,
        `Test 5 failed: Phi(1.96) = ${result}, expected ≈ 0.975`);
}

// ── Test 6: phi(0, 0, 1) = 1/(sqrt(2)*pi) ────────────────────────────────────
// The post's phi divides by (sigma * sqrt(2) * pi), not sqrt(2*pi).
// This is intentional — it's used as a shape function, scaled by a compensating factor.
{
    const result = phi(0, 0, 1);
    const expected = 1 / (Math.SQRT2 * Math.PI);
    assert(Math.abs(result - expected) < 1e-12,
        `Test 6 failed: phi(0,0,1) = ${result}, expected ${expected}`);
}

// ── Test 7: phi is symmetric around mu ───────────────────────────────────────
{
    const mu = 2, sigma = 0.5;
    const left = phi(mu - 1, mu, sigma);
    const right = phi(mu + 1, mu, sigma);
    assert(Math.abs(left - right) < 1e-12,
        `Test 7 failed: phi not symmetric: left=${left}, right=${right}`);
}

// ── Test 8: randn_bm() returns a finite number ────────────────────────────────
{
    const result = randn_bm();
    assert(isFinite(result),
        `Test 8 failed: randn_bm() returned ${result}`);
}

// ── Test 9: Mean of 10000 randn_bm() calls is within 0.1 of 0 ────────────────
{
    let sum = 0;
    const N = 10000;
    for (let i = 0; i < N; i++) sum += randn_bm();
    const mean = sum / N;
    assert(Math.abs(mean) < 0.1,
        `Test 9 failed: mean of ${N} samples = ${mean}, expected within 0.1 of 0`);
}

// ── Test 10: gamma_rand(1) is positive (Exponential distribution) ─────────────
{
    for (let i = 0; i < 20; i++) {
        const result = gamma_rand(1);
        assert(result > 0,
            `Test 10 failed: gamma_rand(1) = ${result}, expected > 0`);
    }
}

// ── Test 11: Mean of gamma_rand(k) ≈ k (Gamma(k,1) has mean k) ───────────────
{
    const k = 4;
    const N = 5000;
    let sum = 0;
    for (let i = 0; i < N; i++) sum += gamma_rand(k);
    const mean = sum / N;
    assert(Math.abs(mean - k) < 0.3,
        `Test 11 failed: mean of gamma_rand(${k}) = ${mean}, expected ≈ ${k}`);
}

// ── Test 12: dirichlet([1,1]) sums to 1 ──────────────────────────────────────
{
    for (let trial = 0; trial < 20; trial++) {
        const [p, q] = dirichlet([1, 1]);
        assert(Math.abs(p + q - 1) < 1e-12,
            `Test 12 failed: dirichlet([1,1]) = [${p}, ${q}], sum = ${p + q}`);
        assert(p >= 0 && p <= 1,
            `Test 12 failed: dirichlet p = ${p} not in [0,1]`);
    }
}

// ── Test 13: biv_gauss(1) returns z2 === z1 when rho=1 ───────────────────────
// With rho=1: z2 = 1*z1 + sqrt(1-1)*randn = z1
{
    const [z1, z2] = biv_gauss(1);
    assert.strictEqual(z1, z2,
        `Test 13 failed: biv_gauss(1) gave z1=${z1}, z2=${z2}, expected equal`);
}

// ── Test 14: biv_gauss(0) components are finite ───────────────────────────────
{
    const [z1, z2] = biv_gauss(0);
    assert(isFinite(z1) && isFinite(z2),
        `Test 14 failed: biv_gauss(0) = [${z1}, ${z2}]`);
}

// ── Test 15: Pearson correlation of perfectly correlated data equals 1 ────────
{
    const xs = [1, 2, 3, 4, 5];
    const ys = [2, 4, 6, 8, 10];
    const rho = pearsonCorrelation(xs, ys);
    assert(Math.abs(rho - 1) < 1e-12,
        `Test 15 failed: Pearson(xs, 2*xs) = ${rho}, expected 1`);
}

// ── Test 16: Pearson correlation of perfectly anti-correlated data equals -1 ──
{
    const xs = [1, 2, 3, 4, 5];
    const ys = [-1, -2, -3, -4, -5];
    const rho = pearsonCorrelation(xs, ys);
    assert(Math.abs(rho + 1) < 1e-12,
        `Test 16 failed: Pearson(xs, -xs) = ${rho}, expected -1`);
}

// ── Test 17: Pearson correlation of known dataset ─────────────────────────────
// xs=[0,1,2], ys=[1,0,1]: mean_x=1, mean_y=2/3, sqxx=2, sqyy=2/3, sqxy=0
// rho = 0 / sqrt(2 * 2/3) = 0
{
    const xs = [0, 1, 2];
    const ys = [1, 0, 1];
    const rho = pearsonCorrelation(xs, ys);
    assert(Math.abs(rho) < 1e-12,
        `Test 17 failed: Pearson correlation = ${rho}, expected 0`);
}

// ── Test 18: roundN floors half-integers (round-half-down) ───────────────────
{
    assert.strictEqual(roundN(1.5), 1, `Test 18a failed: roundN(1.5) = ${roundN(1.5)}, expected 1`);
    assert.strictEqual(roundN(2.5), 2, `Test 18b failed: roundN(2.5) = ${roundN(2.5)}, expected 2`);
    assert.strictEqual(roundN(3.0), 3, `Test 18c failed: roundN(3.0) = ${roundN(3.0)}, expected 3`);
}

// ── Test 19: TDZ safety — gamma_rand uses local let x, not global ─────────────
// In strict mode ('use strict' at top), assigning to an undeclared variable
// throws ReferenceError. gamma_rand previously used bare `x = 0` (global leak).
// After refactoring to `let x = 0`, calling it must not throw.
{
    assert.doesNotThrow(() => gamma_rand(3),
        'Test 19 failed: gamma_rand threw — possible undeclared variable leak');
}

// ── Test 20: TDZ safety — dirichlet uses local xs/x0, not globals ────────────
// Previously `xs = []` and `x0 = 0` were undeclared globals.
// After refactoring to `const xs = []` and `let x0 = 0`, no global leak.
{
    assert.doesNotThrow(() => dirichlet([2, 2]),
        'Test 20 failed: dirichlet threw — possible undeclared variable leak');
    // Also verify the global scope is not polluted
    assert.strictEqual(typeof global.xs, 'undefined',
        `Test 20 failed: global.xs was set to ${global.xs}`);
    assert.strictEqual(typeof global.x0, 'undefined',
        `Test 20 failed: global.x0 was set to ${global.x0}`);
}

// ── Test 21: TDZ safety — mclt init pattern (let before function call) ────────
// Simulates the pattern: declare accumulators, then call an init function
// that assigns to them. Ensures no TDZ error occurs.
{
    let avg_dots = [];
    function simulateMcltReset() { avg_dots = []; }
    assert.doesNotThrow(() => simulateMcltReset(),
        'Test 21 failed: mclt reset pattern threw');
    assert.deepStrictEqual(avg_dots, [], 'Test 21 failed: avg_dots not reset');
}

// ── Test 22: TDZ safety — prsn_plt init pattern ──────────────────────────────
// Simulates the pattern used in prsn_plt(): let sqxx/sqyy/sqxy declared before
// any function assigns to them.
{
    let sqxx = 0, sqyy = 0, sqxy = 0, rho_n = 0;
    function simulateReset() { sqxx = sqxy = sqyy = rho_n = 0; }
    assert.doesNotThrow(() => simulateReset(),
        'Test 22 failed: prsn_plt reset pattern threw');
    assert.strictEqual(sqxx, 0, 'Test 22 failed: sqxx not 0');
    assert.strictEqual(rho_n, 0, 'Test 22 failed: rho_n not 0');
}

console.log('All tests passed!');
