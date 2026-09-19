# Voxel Temple — Qwen3.8-Flash-Next V3 Fix

> 177B parameter MoE model on M1 Max 64GB fixes a broken Three.js voxel temple

## Quick Start

```bash
# View the temple (any browser, no install needed)
open index.html

# Or run the Python HTTP server for ES module support
python3 -m http.server 8080
```

## Files

| File | Description |
|------|-------------|
| `index.html` | Fixed Three.js voxel temple (671 cubes, 4 spirals, portal+orb) |
| `algorithm.md` | Full algorithm documentation (scene graph, animation loop, inputs) |
| `qwen-3.8-flash-m1-max-64gb.gif` | 20-second animated GIF of the temple rendering |
| `launch-qwen-flash-8086.sh` | Script to launch Qwen3.8-Flash-Next on OpenAI-compatible port 8086 |
| `MEDIUM-ARTICLE.md` | Medium-style article on downloading + building the model |

## Demo

**GitHub Pages:** https://frankhouston.github.io/voxel-temple-qwen/

**Key features:** EffectComposer (UnrealBloom), OrbitControls, Stats.js, lil-gui,
InstancedMesh (671 + 160 + 320), FogExp2, sky gradient, emissive orb pulse.

## Model Details

- **Qwen3.8-Flash-Next V3**: 177B params, 512 experts, Q4_0 (95.5 GiB)
- **Hardware**: M1 Max 64GB, Metal GPU (AGXMetalG13X)
- **Fork**: [npanj/llama.cpp](https://github.com/npanj/llama.cpp) with `--moe-stream`
- **API**: OpenAI-compatible on port 8086

## Fixes Applied by Qwen

| Bug | Fix |
|-----|-----|
| Black screen (no animate loop) | Added complete EffectComposer pipeline |
| Stats.js CDN 404 | Changed to `three/addons/libs/stats.module.js` |
| Portal on wrong side | Moved to TOP_Y + 1.3 |
| Missing orb | Added IcosahedronGeometry with emissive #00f0ff |
| Missing staircases | 4 spiral staircases, 40 steps each |
| No GUI/telemetry | Added lil-gui + HUD with FPS, pulse, embers |
| Corrupted hex tokens | Fixed `0xffa martial` → `0xffaa00`, `0x built` → `0xff7a2a` |

## License

MIT
