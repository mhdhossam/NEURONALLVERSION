# NEURON Vision — Phase 0

A deliberately small, matrix-free proof of concept for graph-native visual synthesis.

## Run

From the repository root (Python 3.10+, standard library only):

```bash
python vision_research/neuron_vision.py "a red sports car beside a tree with a sun and mountains" --output artifacts/scene.svg
```

The program writes:
- `artifacts/scene.svg`: rendered image
- `artifacts/scene.json`: graph, edges, activation trace, and explicit limitations

Open the SVG in a browser or image viewer.

## What this tests

1. Compile a constrained prompt into typed visual nodes.
2. Represent containment and simple relationships as graph edges.
3. Propagate bounded activation through the graph.
4. Decode selected graph concepts into deterministic SVG primitives.
5. Preserve an inspectable generation trace.

## What this does NOT do

- It does not train from image datasets.
- It does not learn visual concepts, textures, or object geometry.
- It does not implement diffusion, flow matching, a VAE, or a text encoder.
- It does not generate photorealistic images.
- It is not a Qwen replacement.

This experiment deliberately contains no neural model, learned matrix, tensor library, pretrained embedding, or hosted inference call. The renderer uses ordinary scalar geometry and SVG primitives. The pixels of a rasterized image will still form a 2D grid; that is image representation, not learned neural matrix inference.

## Automated validation and microbenchmark

GitHub Actions run: **42548218618** — all 4 unit tests passed and the benchmark completed successfully on Ubuntu 24.04 / Python 3.11.

Each case ran 500 iterations. Timings cover prompt-to-graph compilation plus SVG string rendering, with Python `tracemalloc` active. They are CI-runner measurements, not desktop measurements.

| Case | Nodes | Edges | Mean ms | P95 ms | Peak traced KiB |
|---|---:|---:|---:|---:|---:|
| Simple car | 2 | 1 | 0.1247 | 0.1297 | 46.62 |
| Car beside tree | 3 | 3 | 0.1497 | 0.1554 | 23.14 |
| Car, tree, sun, mountains | 5 | 5 | 0.1857 | 0.1931 | 24.75 |
| Chair, house, cloud | 4 | 4 | 0.2055 | 0.2126 | 22.01 |
| Unsupported prompt fallback | 4 | 3 | 0.1267 | 0.1342 | 21.31 |

The first test run caught an object-ordering bug (objects were ordered by vocabulary phrase length rather than prompt position). That parser issue was fixed, and the subsequent CI run passed.

**Interpretation:** this demonstrates very fast execution for a tiny deterministic scene compiler/renderer. It says nothing yet about image quality, learned visual capability, or Qwen-equivalent performance. The fallback case also exposes a limitation: unsupported descriptions currently become a generic default landscape instead of being rejected or faithfully represented.

Run locally:

```bash
python -m unittest discover -s vision_research -p "test_*.py" -v
cd vision_research && python benchmark.py
```

## Research questions

- Can NEURON's existing graph semantics express a compositional scene?
- Can relationship constraints influence actual layout rather than merely appear in metadata?
- Can visual primitives and their relationships be learned from examples through local graph updates?
- Can graph-native refinement improve fidelity and diversity as complexity grows?

## Next experiments

1. Add tests for graph invariants, relation handling, and deterministic output.
2. Make relations affect layout and occlusion explicitly.
3. Add a tiny synthetic training corpus of shapes and scenes.
4. Learn reusable primitives and composition rules from that corpus without neural embeddings.
5. Compare graph-only generation with a conventional baseline on the same constrained tasks.

Do not describe this Phase-0 renderer as a general text-to-image model. Its value is that it makes the matrix-free hypothesis testable.
