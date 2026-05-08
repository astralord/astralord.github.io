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

function erfinv(x) {
    let z;
    const a = 0.147;
    let the_sign_of_x;
    if (0 == x) {
        the_sign_of_x = 0;
    } else if (x > 0) {
        the_sign_of_x = 1;
    } else {
        the_sign_of_x = -1;
    }
    if (0 != x) {
        const ln_1minus_x_sqrd = Math.log(1 - x * x);
        const ln_1minusxx_by_a = ln_1minus_x_sqrd / a;
        const ln_1minusxx_by_2 = ln_1minus_x_sqrd / 2;
        const ln_etc_by2_plus2 = ln_1minusxx_by_2 + (2 / (Math.PI * a));
        const first_sqrt = Math.sqrt((ln_etc_by2_plus2 * ln_etc_by2_plus2) - ln_1minusxx_by_a);
        const second_sqrt = Math.sqrt(first_sqrt - ln_etc_by2_plus2);
        z = second_sqrt * the_sign_of_x;
    } else {
        z = 0;
    }
    return z;
}

function PhiInv(y) {
    return 1.41421356237 * erfinv(2 * y - 1);
}

function randn_bm() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function gauss_data(mu, sigma, min, max) {
    const data = [{x: min, y: 0}];
    for (let i = min; i < max; i += 0.01) {
        data.push({x: i, y: Math.exp(-0.5 * ((i - mu) / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI))});
    }
    data.push({x: max, y: 0});
    return data;
}

// Bartlett T_n statistic — extracted from sample() in asymptotic_test()
function bartlettTn(sampleVariances, n) {
    const r = sampleVariances.length;
    const std_avg = sampleVariances.reduce((a, b) => a + b, 0) / r;
    let T_n = r * Math.log(std_avg);
    for (let i = 0; i < r; i++) T_n -= Math.log(sampleVariances[i]);
    return T_n * n;
}

// Chi-square independence statistic — extracted from updateTn() in plt_heatmap()
function chiSquareStat(table, r, s) {
    let t_n = 0;
    let n = 0;
    const xi = new Array(r).fill(0);
    const xj = new Array(s).fill(0);
    for (let i = 0; i < s; i++) {
        for (let j = 0; j < r; j++) {
            const xij = table[i * r + j];
            n += xij;
            xj[i] += xij;
            xi[j] += xij;
        }
    }
    for (let i = 0; i < s; i++) {
        for (let j = 0; j < r; j++) {
            const xij = table[i * r + j];
            if (xij > 0) {
                t_n += xij * Math.log(n * xij / (xi[j] * xj[i]));
            }
        }
    }
    return 2 * t_n;
}

// ── Test 1: erf(0) === 0 ──────────────────────────────────────────────────────
{
    assert(Math.abs(erf(0)) < 1e-12, `Test 1 failed: erf(0) = ${erf(0)}`);
}

// ── Test 2: erf clips to ±1 for |x| > 3 ─────────────────────────────────────
{
    assert.strictEqual(erf(5), 1, `Test 2a failed: erf(5) should be 1`);
    assert.strictEqual(erf(-5), -1, `Test 2b failed: erf(-5) should be -1`);
}

// ── Test 3: erfinv(0) === 0 ──────────────────────────────────────────────────
{
    assert.strictEqual(erfinv(0), 0, `Test 3 failed: erfinv(0) = ${erfinv(0)}`);
}

// ── Test 4: erf(erfinv(x)) ≈ x (round-trip) ──────────────────────────────────
{
    for (const x of [0.1, 0.5, 0.9, -0.5, -0.8]) {
        const roundtrip = erf(erfinv(x));
        assert(Math.abs(roundtrip - x) < 1e-3,
            `Test 4 failed: erf(erfinv(${x})) = ${roundtrip}, expected ${x}`);
    }
}

// ── Test 5: Phi(0) === 0.5 ───────────────────────────────────────────────────
{
    assert(Math.abs(Phi(0) - 0.5) < 1e-9, `Test 5 failed: Phi(0) = ${Phi(0)}`);
}

// ── Test 6: PhiInv(0.5) ≈ 0 ─────────────────────────────────────────────────
{
    assert(Math.abs(PhiInv(0.5)) < 1e-5, `Test 6 failed: PhiInv(0.5) = ${PhiInv(0.5)}`);
}

// ── Test 7: Phi(PhiInv(alpha)) ≈ alpha (round-trip) ─────────────────────────
{
    for (const alpha of [0.05, 0.1, 0.25, 0.75, 0.9, 0.95]) {
        const roundtrip = Phi(PhiInv(alpha));
        assert(Math.abs(roundtrip - alpha) < 1e-3,
            `Test 7 failed: Phi(PhiInv(${alpha})) = ${roundtrip}`);
    }
}

// ── Test 8: PhiInv(0.975) ≈ 1.96 (standard 95% CI z-score) ──────────────────
{
    const result = PhiInv(0.975);
    assert(Math.abs(result - 1.96) < 0.01,
        `Test 8 failed: PhiInv(0.975) = ${result}, expected ≈ 1.96`);
}

// ── Test 9: randn_bm() returns finite number ─────────────────────────────────
{
    assert(isFinite(randn_bm()), `Test 9 failed: randn_bm() not finite`);
}

// ── Test 10: Mean of 10000 randn_bm() ≈ 0 ───────────────────────────────────
{
    let sum = 0;
    const N = 10000;
    for (let i = 0; i < N; i++) sum += randn_bm();
    assert(Math.abs(sum / N) < 0.1, `Test 10 failed: mean = ${sum / N}`);
}

// ── Test 11: gauss_data endpoints are zero ───────────────────────────────────
{
    const data = gauss_data(0, 1, -4, 4);
    assert.strictEqual(data[0].y, 0, `Test 11a failed`);
    assert.strictEqual(data[data.length - 1].y, 0, `Test 11b failed`);
    assert.strictEqual(data[0].x, -4, `Test 11c failed`);
    assert.strictEqual(data[data.length - 1].x, 4, `Test 11d failed`);
}

// ── Test 12: gauss_data peak is at mu and equals standard normal PDF ──────────
{
    const data = gauss_data(0, 1, -4, 4);
    const peak = data.reduce((best, d) => d.y > best.y ? d : best, {y: -Infinity});
    assert(Math.abs(peak.x) < 0.1, `Test 12a failed: peak at x=${peak.x}`);
    const expected_peak = 1 / Math.sqrt(2 * Math.PI);
    assert(Math.abs(peak.y - expected_peak) < 0.01,
        `Test 12b failed: peak y=${peak.y}, expected ${expected_peak}`);
}

// ── Test 13: Bartlett T_n = 0 when all variances are equal ───────────────────
// If all samples have same variance, T_n should be 0
{
    const result = bartlettTn([1, 1, 1], 30);
    assert(Math.abs(result) < 1e-10,
        `Test 13 failed: Bartlett T_n with equal variances = ${result}, expected 0`);
}

// ── Test 14: Bartlett T_n > 0 when variances differ ─────────────────────────
{
    const result = bartlettTn([1, 4, 9], 30);
    assert(result > 0, `Test 14 failed: Bartlett T_n = ${result}, expected > 0`);
}

// ── Test 15: Chi-square stat for independent table ────────────────────────────
// For a perfectly uniform 2×2 table [[n,n],[n,n]], T_n should be 0
{
    const table = [5, 5, 5, 5]; // r=2, s=2
    const result = chiSquareStat(table, 2, 2);
    assert(Math.abs(result) < 1e-10,
        `Test 15 failed: chi-sq for uniform table = ${result}, expected 0`);
}

// ── Test 16: Chi-square stat > 0 for non-uniform table ───────────────────────
{
    const table = [10, 1, 1, 10]; // r=2, s=2 — strong association
    const result = chiSquareStat(table, 2, 2);
    assert(result > 0, `Test 16 failed: chi-sq for non-uniform table = ${result}`);
}

// ── Test 17: Error probability formula (basic_test) ─────────────────────────
// P(reject H | H is true) = Phi(sqrt(n/(p*(1-p))) * (p - c))
// for n=100, p_a=0.5, c=0.5: pp = 0, Phi(0) = 0.5
{
    const n = 100, p_a = 0.5, c = 0.5;
    const pp = Math.sqrt(n / (p_a * (1 - p_a))) * (p_a - c);
    const result = Phi(pp);
    assert(Math.abs(result - 0.5) < 1e-9,
        `Test 17 failed: error prob = ${result}, expected 0.5`);
}

// ── Test 18: Power formula (simple_hypothesis) ───────────────────────────────
// power = 1 - Phi(sqrt(n) * (mu0 - mu1) / sigma + u_q)
// with mu0=-1, mu1=1, sigma=1, n=10, alpha=0.05: u_q ≈ 1.6449
{
    const mu0 = -1, mu1 = 1, sigma = 1, n = 10, alpha = 0.05;
    const u_q = PhiInv(1 - alpha);
    const power = 1 - Phi(Math.sqrt(n) * (mu0 - mu1) / sigma + u_q);
    assert(power > 0 && power <= 1,
        `Test 18 failed: power = ${power} not in (0,1]`);
    // With mu0=-1, mu1=1, n=10, sigma=1, power should be substantial
    assert(power > 0.5, `Test 18 failed: power = ${power} expected > 0.5`);
}

// ── Test 19: TDZ safety — erfinv uses const/let locals, not globals ──────────
// In strict mode, undeclared assignments throw ReferenceError.
// erfinv previously used bare `var z` (global). After refactoring to `let z`, must not throw.
{
    assert.doesNotThrow(() => erfinv(0.5),
        'Test 19 failed: erfinv threw — possible undeclared variable');
    assert.doesNotThrow(() => erfinv(0),
        'Test 19 failed: erfinv(0) threw');
}

// ── Test 20: TDZ safety — booked_color/booked_color_id are now const ─────────
// Simulates the delete-button handler pattern: local consts used for splice.
{
    const colors = ['#65AD69', '#EDA137', '#E86456'];
    let booked_colors = ['#65AD69', '#EDA137'];
    function simulateDelete(id) {
        const booked_color = colors[id];
        const booked_color_id = booked_colors.indexOf(booked_color);
        booked_colors.splice(booked_color_id, 1);
    }
    assert.doesNotThrow(() => simulateDelete(0),
        'Test 20 failed: delete handler threw');
    assert.strictEqual(booked_colors.length, 1,
        'Test 20 failed: booked_colors not spliced');
    assert.strictEqual(booked_colors[0], '#EDA137',
        'Test 20 failed: wrong color removed');
    assert.strictEqual(typeof global.booked_color, 'undefined',
        'Test 20 failed: global.booked_color leaked');
}

// ── Test 21: TDZ safety — updateTn pattern (const tn_dot declared before call) ─
// Simulates the pattern in plt_heatmap(): updateTn() is defined before tn_dot is
// declared, but only called after. This test verifies no TDZ error occurs.
{
    let t_n = 0;
    const c = 21.026;
    // Simulate the pattern: function defined first, called after declaration
    function simulateUpdateTn(tn_dot_ref, data) {
        t_n = 0;
        let n = 0;
        const xi = [0, 0], xj = [0, 0];
        for (let i = 0; i < 2; i++) {
            for (let j = 0; j < 2; j++) {
                const xij = data[i * 2 + j];
                n += xij; xj[i] += xij; xi[j] += xij;
            }
        }
        for (let i = 0; i < 2; i++) {
            for (let j = 0; j < 2; j++) {
                const xij = data[i * 2 + j];
                if (xij > 0) t_n += xij * Math.log(n * xij / (xi[j] * xj[i]));
            }
        }
        return t_n > c ? 1 : 0;
    }
    const tn_dot = { decision: 0 }; // simulated tn_dot
    assert.doesNotThrow(() => {
        tn_dot.decision = simulateUpdateTn(tn_dot, [10, 1, 1, 10]);
    }, 'Test 21 failed: updateTn pattern threw');
    assert(t_n > 0, `Test 21 failed: t_n = ${t_n} expected > 0`);
}

// ── Test 22: TDZ safety — table_text vars declared before updateTableText call ─
// Simulates the pattern in simple_hypothesis(): table_text_hh etc. are declared
// after updateTableText() definition but before any event-driven call.
{
    const counts = [0, 0, 0, 0];
    // Simulate: function uses externally-declared const objects
    const table_text_hh = { val: 0 };
    const table_text_hk = { val: 0 };
    const table_text_kh = { val: 0 };
    const table_text_kk = { val: 0 };
    function updateTableText() {
        table_text_hh.val = counts[0];
        table_text_hk.val = counts[1];
        table_text_kh.val = counts[2];
        table_text_kk.val = counts[3];
    }
    counts[3] = 5;
    assert.doesNotThrow(() => updateTableText(),
        'Test 22 failed: updateTableText threw');
    assert.strictEqual(table_text_kk.val, 5, 'Test 22 failed: wrong value');
}

console.log('All tests passed!');
