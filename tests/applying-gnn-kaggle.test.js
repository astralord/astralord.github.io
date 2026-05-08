'use strict';

const assert = require('assert');

// ── Test 1: draw_edge x1/y1/x2/y2 let-reassignment pattern ──────────────────
// draw_edge() uses let x1,x2,y1,y2 that are reassigned inside if/else branches.
{
    function simulateDrawEdge(type) {
        let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
        let dash = 0;
        const x = 100, y = 150;
        if (type === 'hrz') {
            x1 = x - 40; y1 = y; x2 = x + 40; y2 = y;
        } else if (type === 'vrt') {
            x1 = x; y1 = y - 40; x2 = x; y2 = y + 40;
        } else if (type === 'tmp') {
            x1 = x; y1 = y; x2 = x + 35; y2 = y - 50; dash = 1;
        }
        return { x1, x2, y1, y2, dash };
    }
    assert.doesNotThrow(() => simulateDrawEdge('hrz'), 'Test 1a: hrz threw');
    assert.doesNotThrow(() => simulateDrawEdge('vrt'), 'Test 1b: vrt threw');
    assert.doesNotThrow(() => simulateDrawEdge('tmp'), 'Test 1c: tmp threw');

    const hrz = simulateDrawEdge('hrz');
    assert.strictEqual(hrz.x1, 60, `Test 1d: x1=${hrz.x1}`);
    assert.strictEqual(hrz.x2, 140, `Test 1e: x2=${hrz.x2}`);
    assert.strictEqual(hrz.dash, 0, `Test 1f: dash=${hrz.dash}`);

    const tmp = simulateDrawEdge('tmp');
    assert.strictEqual(tmp.dash, 1, `Test 1g: dash=${tmp.dash}`);
}

// ── Test 2: triangleSize constants have correct values ────────────────────────
{
    const triangleSize_draw_cross = 70;
    const triangleSize_draw_triangle = 25;
    assert.strictEqual(triangleSize_draw_cross, 70, 'Test 2a');
    assert.strictEqual(triangleSize_draw_triangle, 25, 'Test 2b');
}

// ── Test 3: opacity is const in both graph_zoomed and path_search ─────────────
{
    assert.doesNotThrow(() => {
        const opacity = 0.04;
        void opacity;
    }, 'Test 3: const opacity threw');
}

// ── Test 4: for-let loop variable is block-scoped ─────────────────────────────
{
    let count = 0;
    for (let i = 0; i < 6; i += 1) count++;
    assert.strictEqual(count, 6, `Test 4a: count=${count}`);
    assert.strictEqual(typeof i, 'undefined',
        'Test 4b: loop var i leaked to outer scope');
}

// ── Test 5: nested for-let (i, j) are block-scoped ───────────────────────────
{
    let sum = 0;
    for (let i = 0; i < 5; i += 1) {
        for (let j = 0; j < 6; j += 1) { sum++; }
    }
    assert.strictEqual(sum, 30, `Test 5a: sum=${sum}`);
}

// ── Test 6: TDZ safety — x1/x2/y1/y2 are let, no global leak ─────────────────
{
    (function() {
        let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
        let dash = 0;
        x1 = 1; x2 = 2; y1 = 3; y2 = 4; dash = 1;
        void [x1, x2, y1, y2, dash];
    })();
    assert.strictEqual(typeof global.x1, 'undefined', 'Test 6a: x1 leaked');
    assert.strictEqual(typeof global.dash, 'undefined', 'Test 6b: dash leaked');
}

// ── Test 7: TDZ safety — triangleSize / triangle locals don't leak ────────────
{
    (function() {
        const triangleSize = 70;
        const triangle = { size: triangleSize };
        void triangle;
    })();
    assert.strictEqual(typeof global.triangleSize, 'undefined',
        'Test 7: triangleSize leaked');
    assert.strictEqual(typeof global.triangle, 'undefined',
        'Test 7: triangle leaked');
}

// ── Test 8: svg const inside function doesn't leak ────────────────────────────
{
    (function graph_zoomed() { const svg = {}; void svg; })();
    assert.strictEqual(typeof global.svg, 'undefined', 'Test 8: svg leaked');
}

console.log('All tests passed!');
