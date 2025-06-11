# LLM GPU VRAM Calculator

An interactive web tool to calculate VRAM requirements and performance characteristics for running Large Language Models (LLMs) on different GPU configurations.

## Features

- Calculate VRAM requirements for different LLM models
- Support for various quantization types (FP16, FP8, INT8, INT4)
- GPU scaling analysis with parallel processing
- Estimates for:
  - Model VRAM usage
  - KV cache requirements
  - Generation and prompt processing speeds
  - Multi-user concurrent processing capabilities
- Real-time updates based on:
  - GPU selection
  - Model selection
  - VRAM utilization settings
  - Token length configuration
  - Concurrent user count

## Technology Stack

- React + TypeScript
- Vite
- Cloudflare Pages for deployment

## Development

1. Install dependencies:
```bash
npm install
```

2. Start development server:
```bash
npm run dev
```

3. Build for production:
```bash
npm run build
```

## Configuration

The calculator uses pre-defined GPU cards and model definitions located in:
- `src/data/gpuCards.ts` - GPU specifications
- `src/data/modelDefs.ts` - LLM model definitions

## License

MIT License