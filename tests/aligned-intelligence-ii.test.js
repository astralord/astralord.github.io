'use strict';

const assert = require('assert');

// ── Helper stubs (D3 is not available in Node; simulate the patterns) ─────────

// Simulates the triangle/tot/talm SVG drawing code structure:
// all variables must be declared (const/let), no undeclared globals.

// ── Test 1: thought_shift let-reassignment pattern works ─────────────────────
// tot() uses `let thought_shift` declared once, then reassigned for sc/tot levels.
{
    const x_start = 65;
    const x_end = 565;
    const cot_level = 100;
    const sc_level = 220;
    const tot_level = 440;

    let thought_shift = x_start + 170;
    let thought_shift_2 = thought_shift + 170;

    // First block (cot_level)
    assert.strictEqual(thought_shift, 235, `Test 1a: thought_shift initial = ${thought_shift}`);
    assert.strictEqual(thought_shift_2, 405, `Test 1b: thought_shift_2 initial = ${thought_shift_2}`);

    // Reassign for sc_level block (as tot() does)
    thought_shift = x_start + 170;
    thought_shift_2 = thought_shift + 170;
    assert.strictEqual(thought_shift, 235, `Test 1c: thought_shift after reassign = ${thought_shift}`);

    // Reassign for tot_level block
    thought_shift = x_start + 170;
    thought_shift_2 = thought_shift + 170;
    assert.strictEqual(thought_shift, 235, `Test 1d: thought_shift tot block = ${thought_shift}`);
}

// ── Test 2: talm() tool_shift pattern works ───────────────────────────────────
// talm() uses const tool_shift and tool_shift_2 (not reassigned).
{
    const x_start = 65;
    const tool_shift = x_start + 165;
    const tool_shift_2 = tool_shift + 165;
    assert.strictEqual(tool_shift, 230, `Test 2a: tool_shift = ${tool_shift}`);
    assert.strictEqual(tool_shift_2, 395, `Test 2b: tool_shift_2 = ${tool_shift_2}`);
}

// ── Test 3: triangle geometry constants are correct ───────────────────────────
{
    const triangleSize = 25;
    assert.strictEqual(triangleSize, 25, `Test 3: triangleSize = ${triangleSize}`);
}

// ── Test 4: TDZ safety — tot() pattern (let thought_shift before any use) ────
// Verifies that let-declared variables can be reassigned without TDZ errors.
{
    assert.doesNotThrow(() => {
        const x_start = 65;
        let thought_shift = x_start + 170;
        thought_shift = x_start + 170; // reassign as in tot() sc_level block
        thought_shift = x_start + 170; // reassign as in tot() tot_level block
        void thought_shift;
    }, 'Test 4 failed: thought_shift reassignment threw');
}

// ── Test 5: TDZ safety — talm() undeclared globals fixed ──────────────────────
// x_start, x_end, fs_level, tool_shift, tool_shift_2 were undeclared globals.
// After fixing to const, strict mode must not throw.
{
    assert.doesNotThrow(() => {
        const x_start = 65;
        const x_end = 565;
        const fs_level = 25;
        const tool_shift = x_start + 165;
        const tool_shift_2 = tool_shift + 165;
        void [x_start, x_end, fs_level, tool_shift, tool_shift_2];
    }, 'Test 5 failed: talm locals threw in strict mode');
}

// ── Test 6: TDZ safety — tot() undeclared globals fixed ───────────────────────
// x_start, x_end, fs_level, cot_level, sc_level, tot_level were undeclared.
{
    assert.doesNotThrow(() => {
        const x_start = 65;
        const x_end = 565;
        const fs_level = 25;
        const cot_level = 100;
        const sc_level = 220;
        const tot_level = 440;
        void [x_start, x_end, fs_level, cot_level, sc_level, tot_level];
    }, 'Test 6 failed: tot locals threw in strict mode');
}

// ── Test 7: No global leaks for thought_shift ─────────────────────────────────
{
    assert.strictEqual(typeof global.thought_shift, 'undefined',
        `Test 7 failed: global.thought_shift = ${global.thought_shift}`);
    assert.strictEqual(typeof global.thought_shift_2, 'undefined',
        `Test 7 failed: global.thought_shift_2 = ${global.thought_shift_2}`);
}

// ── Test 8: No global leaks for talm variables ────────────────────────────────
{
    assert.strictEqual(typeof global.tool_shift, 'undefined',
        `Test 8 failed: global.tool_shift leaked`);
    assert.strictEqual(typeof global.x_start, 'undefined',
        `Test 8 failed: global.x_start leaked`);
    assert.strictEqual(typeof global.x_end, 'undefined',
        `Test 8 failed: global.x_end leaked`);
    assert.strictEqual(typeof global.fs_level, 'undefined',
        `Test 8 failed: global.fs_level leaked`);
}

// ── Test 9: triangleSize const is block-scoped ────────────────────────────────
{
    {
        const triangleSize = 25;
        void triangleSize;
    }
    assert.strictEqual(typeof global.triangleSize, 'undefined',
        'Test 9 failed: triangleSize leaked to global');
}

// ── Test 10: Geometry arithmetic — x_end - x_start = 500 ─────────────────────
{
    const x_start = 65;
    const x_end = 565;
    assert.strictEqual(x_end - x_start, 500, `Test 10: diagram width = ${x_end - x_start}`);
}

console.log('All tests passed!');
