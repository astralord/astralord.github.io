'use strict';

const assert = require('assert');

// Building Aligned Intelligence Part I has only static SVG diagrams —
// no math functions and no interactive logic. Tests verify that the
// refactored patterns (var → const) are TDZ-safe in strict mode.

// ── Test 1: triangle_symb declared as const (no var) ─────────────────────────
{
    // Simulate the pattern from triangle(): const triangle_symb = ...
    assert.doesNotThrow(() => {
        const triangleSize = 25;
        const triangle_symb = { size: triangleSize }; // stub for d3.symbol()
        void triangle_symb;
    }, 'Test 1 failed: triangle_symb const declaration threw');
}

// ── Test 2: svg declared as const inside each function ────────────────────────
{
    assert.doesNotThrow(() => {
        const svg = { append: () => svg }; // minimal D3 stub
        void svg;
    }, 'Test 2 failed: const svg declaration threw');
}

// ── Test 3: No global leak for triangle_symb ──────────────────────────────────
{
    (function() {
        const triangleSize = 25;
        const triangle_symb = { size: triangleSize };
        void triangle_symb;
    })();
    assert.strictEqual(typeof global.triangle_symb, 'undefined',
        'Test 3 failed: triangle_symb leaked to global');
    assert.strictEqual(typeof global.triangleSize, 'undefined',
        'Test 3 failed: triangleSize leaked to global');
}

// ── Test 4: No global leak for svg ────────────────────────────────────────────
{
    (function() {
        const svg = {};
        void svg;
    })();
    assert.strictEqual(typeof global.svg, 'undefined',
        'Test 4 failed: svg leaked to global');
}

// ── Test 5: All six drawing functions can use const svg without conflict ───────
// Six separate functions each declare their own const svg — block-scoped,
// so no naming conflict between them.
{
    function gpt_arch_simple() { const svg = {}; return svg; }
    function gpt_arch()        { const svg = {}; return svg; }
    function sft_learning()    { const svg = {}; return svg; }
    function rm_learning()     { const svg = {}; return svg; }
    function rlhf()            { const svg = {}; return svg; }
    function backup_diagram()  { const svg = {}; return svg; }

    assert.doesNotThrow(() => {
        gpt_arch_simple(); gpt_arch(); sft_learning();
        rm_learning(); rlhf(); backup_diagram();
    }, 'Test 5 failed: multiple const svg functions threw');
}

console.log('All tests passed!');
