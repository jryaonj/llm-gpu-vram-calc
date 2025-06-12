import { useState, useEffect } from 'react';
import { Info, Github, Settings2 } from 'lucide-react';

// --- CONFIG: please populate these arrays from your spreadsheet export ---

import type { GPUCard, ModelDef, CalcResults } from '../types/index.ts'; // Assuming GPUCard is defined in another file

import { gpuCards } from '../data/gpuCards.ts'; // Import GPU cards data
import { modelDefs } from '../data/modelDefs.ts'; // Import model definitions

function computeModelVramGB(model: ModelDef, quant: 'fp16' | 'fp8' | 'int8' | 'int4'): number {
  // If using the model's native quantization type
  console.log(`Computing VRAM for model: ${model.name}, ${model.quantType}, quant: ${quant}`);
  if (quant === model.quantType) {
    return model.modelSizeGB;
  }
  else {
    // For other quantization types
    const bytesPerParam = quant === 'fp16' ? 2
      : quant === 'fp8' ? 1
      : quant === 'int8' ? 1
      : 0.5; // int4
    const quantFactor = quant === 'fp8' ? (0.5 + 3/32)/0.5 // worst case quantization factor for int4, fp8, fp16
      : quant === 'int4' ? (0.5 + 3/32)/0.5
      : quant === 'fp16' ? 1
      : 1; // fp32
    return model.totalParamsB * bytesPerParam * quantFactor
  }
}

function computeKvCacheVramGB(_hiddenSize: number, maxLength: number, quant: 'fp16' | 'fp8' | 'int8' | 'int4', model: ModelDef, _card: GPUCard): number {
  const bytesPerValue = quant === 'fp16' ? 2
    : quant === 'fp8' ? 1
    : quant === 'int8' ? 1
    : quant === 'int4' ? 0.5
    : 4; // fp32

  // Check if GPU supports FP8 for KV cache (Ampere and newer architectures)
  // const supportsFp8KV = card.kvQuantType === 'fp8';
  const baseKVSize = model.perKVsizeFp8; // This is the FP8 size per token

  // If GPU doesn't support FP8 KV cache, use precalculated FP8 value * 2 for FP16
  // if (!supportsFp8KV && model.perKVsizeFp8) {
  //   const sizeMultiplier = quant === 'fp16' ? 2 : 
  //                         quant === 'fp8' ? 1 :
  //                         quant === 'int8' ? 1 : 
  //                         quant === 'int4' ? 0.5 : 1;
  //   // Calculate total KV cache size in GB
  //   // Notes: both k and v are stored, so we multiply by 2
  //   // Also, maxLength is the number of tokens
  //   // and baseKVSize is in bytes per token
  //   // Finally, we convert bytes to gigabytes
  //   return (baseKVSize * sizeMultiplier * maxLength * 2) / (1024 * 1024 * 1024);
  // }

  // For GPUs that support FP8 or when using other quantization
  return (baseKVSize * bytesPerValue * maxLength * 2) / (1024 * 1024 * 1024);
}

export default function LLMVRAMCalculator() {
  // selections
  const [selectedCard, setSelectedCard] = useState<GPUCard | null>(null);
  const [selectedModel, setSelectedModel] = useState<ModelDef | null>(null);
  const [quantType, setQuantType] = useState<'fp16' | 'fp8' | 'int8' | 'int4'>('int4');
  // kv cache quantization type, default to fp8
  // as it is the most common for KV cache on production environments
  const [kvQuantType, _setKvQuantType] = useState<'fp16' | 'fp8' | 'int8' | 'int4'>('fp8');
  const [maxLength, setMaxLength] = useState<number>(8192);
  const [userCount, setUserCount] = useState<number>(10);

  // outputs
  const [results, setResults] = useState<CalcResults | null>(null);

  // Add these to the state declarations in LLMVRAMCalculator:
  const [vramUtilProportion, setVramUtilProportion] = useState<number>(0.9); // 90% default
  const [minReserveVramGB, setMinReserveVramGB] = useState<number>(2); // 2GB default
  const [parallelGPUs, setParallelGPUs] = useState<number>(1);
  const [isAdvanced, setIsAdvanced] = useState(false);

  // Add touch event handler for range inputs
  const handleTouchMove = (e: React.TouchEvent) => {
    e.stopPropagation();
  };

  useEffect(() => {
    if (selectedCard && selectedModel) {
      // compute base values
      const modelVram = computeModelVramGB(selectedModel, quantType);
      const kvVram = computeKvCacheVramGB(selectedModel.hiddenSize, maxLength, kvQuantType, selectedModel, selectedCard);
      // const total = modelVram + kvVram;

      // Parallel scaling factors
      const ppScaling = Math.pow(parallelGPUs, 0.6);
      const membwScaling = Math.pow(parallelGPUs, 0.8);

      const totalVram = selectedCard.vramGb * parallelGPUs

      // compute usable VRAM per GPU
      const proportionalReserve = totalVram * (1 - vramUtilProportion);
      const effectiveReserve = Math.max(proportionalReserve, minReserveVramGB);
      const usableVram = Math.max(0, totalVram - effectiveReserve);
      const usableKvCacheVram = Math.max(0, usableVram - modelVram);

      // If total VRAM exceeds usable VRAM, set results to null
      if (usableVram==0 || usableKvCacheVram==0 || modelVram === 0 || kvVram === 0) {
        // If model or kv cache VRAM is zero, set results with error message
        setResults({
          modelVram: modelVram,
          kvCacheVram: kvVram,
          totalVram: totalVram,
          usableVram: usableVram,
          usableKvCacheVram: usableKvCacheVram,
          reservedVram: effectiveReserve,
          genSpeed: 0,
          promptSpeed: 0,
          membwScaling,
          ppScaling,
          sharedGen: 0,
          sharedPrompt: 0,
          fullLengthGenCount: 0,
          maxTokenCountSimultaneous: 0,
          error: modelVram === 0 ? "Invalid model VRAM calculation" : 
           kvVram === 0 ? "Invalid KV cache calculation" :
           "No usable VRAM available"
        });
        return;
      }

      // Calculate how much full-length generation can be done
      // Note: this is a simplified model, real-world performance may vary
      // For generation speed, we consider memory bandwidth and process power
      // quantization scaling factor
      const fullLengthGenCount = usableKvCacheVram / kvVram;
      // Calculate total simultaneous token generate/prompt possible
      const maxTokenCountSimultaneous = maxLength * fullLengthGenCount;
      
      // speeds with parallel scaling
      const genQuant = 
        (selectedModel.quantBits === 4 && selectedCard.processPower.fp16) ? 2 :
        (selectedModel.quantBits === 8 && !selectedCard.processPower.fp8) ? 2 :
        1;

      const baseGenSpeed = (selectedCard.memoryBandwidthGBs / selectedModel.activeParamsB) * genQuant;
      const genSpeed = baseGenSpeed * membwScaling;

      // noe vLLM backend only use fp16 for process power
      const ppow = selectedCard.processPower.fp16 ?? selectedCard.processPower.fp32 ?? 0;
      const basePromptSpeed = (ppow * 1000 / selectedModel.totalParamsB) / Math.sqrt(2);
      const promptSpeed = basePromptSpeed * ppScaling;

      // calculate if usableVram is sufficient for model and kv cache
      // If we can't do full-length generation, set results to null
      const total = modelVram + kvVram;
      if (total > usableVram) {
        // make the null results more informative
        // Set results with zero speeds to indicate insufficient VRAM
        setResults({
          modelVram: modelVram,
          kvCacheVram: kvVram,
          totalVram: totalVram,
          usableVram: usableVram,
          usableKvCacheVram: usableKvCacheVram,
          reservedVram: effectiveReserve,
          genSpeed,
          promptSpeed,
          membwScaling,    // Add scaling factors
          ppScaling,       // Add scaling factors
          sharedGen: genSpeed / userCount,
          sharedPrompt: promptSpeed / userCount,
          fullLengthGenCount: fullLengthGenCount,
          maxTokenCountSimultaneous: maxTokenCountSimultaneous,
          error: `Insufficient VRAM: ${usableVram.toFixed(2)} GB available, ${total.toFixed(2)} GB required.`
        });
        return;
      }

      setResults({
        modelVram: modelVram,
        kvCacheVram: kvVram,
        totalVram: totalVram,
        usableVram: usableVram,
        usableKvCacheVram: usableKvCacheVram,
        reservedVram: effectiveReserve,
        genSpeed,
        promptSpeed,
        membwScaling,    // Add scaling factors
        ppScaling,       // Add scaling factors
        sharedGen: genSpeed / userCount,
        sharedPrompt: promptSpeed / userCount,
        fullLengthGenCount: fullLengthGenCount,
        maxTokenCountSimultaneous: maxTokenCountSimultaneous,
        error: null // No error
      });
    }
  }, [selectedCard, selectedModel, quantType, maxLength, userCount, vramUtilProportion, minReserveVramGB, parallelGPUs]);

  return (
    <div className="max-w-4xl mx-auto p-4 sm:p-6 space-y-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <h1 className="text-2xl sm:text-3xl font-extrabold bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
          LLM GPU VRAM & Speed Calculator
        </h1>
        <div className="flex items-center gap-4">
          <a
            href="https://github.com/jryaonj/llm-gpu-vram-calc"
            target="_blank"
            rel="noopener noreferrer"
            className="btn btn-ghost btn-sm gap-2 hover:scale-105 transition-transform"
          >
            <Github className="w-5 h-5" />
            <span className="hidden sm:inline">Star ★</span>
          </a>
          <button
            onClick={() => setIsAdvanced(!isAdvanced)}
            className={`btn btn-sm gap-2 transition-all duration-300 ${
              isAdvanced ? 'btn-primary' : 'btn-ghost'
            }`}
          >
            <Settings2 className="w-4 h-4" />
            <span>Advanced</span>
          </button>
        </div>
      </div>

      <div className="card bg-base-100 shadow-lg p-4 sm:p-6 transition-all duration-300">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Model Selection */}
          <div className="form-control">
            <label className="label">
              <span className="label-text font-medium">Model</span>
            </label>
            <select
              value={selectedModel?.name || ""}
              onChange={e => setSelectedModel(modelDefs.find(m => m.name === e.target.value) || null)}
              className="select select-bordered w-full transition-all duration-200 hover:border-primary focus:border-primary"
            >
              <option value="">Select a model</option>
              {modelDefs.map(m => <option key={m.name}>{m.name}</option>)}
            </select>
          </div>

          {/* Model Quantization */}
          <div className="form-control">
            <label className="label">
              <span className="label-text font-medium">Model Quantization</span>
            </label>
            <select
              value={quantType}
              onChange={e => setQuantType(e.target.value as any)}
              className="select select-bordered w-full transition-all duration-200 hover:border-primary focus:border-primary"
            >
              <option value="int4">AWQ/GPTQ/INT4</option>
              <option value="fp8">FP8/INT8</option>
              <option value="fp16">FP16</option>
            </select>
          </div>

          {/* GPU Card Selection */}
          <div className="form-control">
            <label className="label">
              <span className="label-text font-medium">GPU Card</span>
              <span
                className="tooltip tooltip-bottom"
                data-tip="Card marked with * are theoretical estimates and may not even properly execute."
              >
                <Info className="w-4 h-4 text-gray-400 hover:text-gray-600 cursor-pointer transition-colors" />
              </span>
            </label>
            <select
              onChange={e => {
                const card = gpuCards.find(c => c.name === e.target.value) || null;
                setSelectedCard(card);
                if (card) {
                  _setKvQuantType(card.kvQuantType as 'fp16' | 'fp8' | 'int8' | 'int4' || 'fp16');
                }
              }}
              className="select select-bordered w-full transition-all duration-200 hover:border-primary focus:border-primary"
            >
              <option value="">Select a card</option>
              {gpuCards.map(c => <option key={c.name}>{c.name}</option>)}
            </select>
          </div>

          {/* Parallel GPUs */}
          <div className="form-control">
            <label className="label">
              <span className="label-text font-medium">Parallel GPUs</span>
              <span
                className="tooltip tooltip-bottom"
                data-tip="Number of GPUs to run in parallel. We apply sub-linear scaling efficiency: throughput (gpu_pp) scales as n⁰·⁶ and memory bandwidth (gpu_membw) as n⁰·⁸."
              >
                <Info className="w-4 h-4 text-gray-400 hover:text-gray-600 cursor-pointer transition-colors" />
              </span>
            </label>
            <div className="flex items-center gap-4">
              <input
                type="range"
                min={1}
                max={8}
                value={parallelGPUs}
                onChange={e => setParallelGPUs(+e.target.value)}
                onTouchMove={handleTouchMove}
                className="range range-primary flex-grow"
              />
              {isAdvanced ? (
                <input
                  type="number"
                  value={parallelGPUs}
                  onChange={e => setParallelGPUs(Math.min(8, Math.max(1, +e.target.value)))}
                  className="input input-bordered w-20 text-center"
                  min={1}
                  max={8}
                />
              ) : (
                <span className="text-sm font-medium min-w-[3rem] text-center">
                  {parallelGPUs}
                </span>
              )}
            </div>
          </div>

          {/* VRAM Utilization */}
          <div className="form-control">
            <label className="label">
              <span className="label-text font-medium">VRAM Utilization</span>
              <span 
                className="tooltip tooltip-bottom" 
                data-tip="What percentage of your card's VRAM should be reserved for model + cache."
              >
                <Info className="w-4 h-4 text-gray-400 hover:text-gray-600 cursor-pointer transition-colors" />
              </span>
            </label>
            <div className="flex items-center gap-4">
              <input
                type="range"
                min={10}
                max={100}
                value={vramUtilProportion * 100}
                onChange={e => setVramUtilProportion(+e.target.value / 100)}
                onTouchMove={handleTouchMove}
                className="range range-secondary flex-grow"
              />
              {isAdvanced ? (
                <input
                  type="number"
                  value={Math.round(vramUtilProportion * 100)}
                  onChange={e => setVramUtilProportion(Math.min(100, Math.max(10, +e.target.value)) / 100)}
                  className="input input-bordered w-20 text-center"
                  min={10}
                  max={100}
                />
              ) : (
                <span className="text-sm font-medium min-w-[3rem] text-center">
                  {Math.round(vramUtilProportion * 100)}%
                </span>
              )}
            </div>
          </div>

          {/* Reserve VRAM */}
          <div className="form-control">
            <label className="label">
              <span className="label-text font-medium">Reserve VRAM</span>
              <span 
                className="tooltip tooltip-bottom" 
                data-tip="Minimum VRAM to reserve for system and other GPU tasks (embedding, rerank, etc). This ensures stable operation when running multiple models."
              >
                <Info className="w-4 h-4 text-gray-400 hover:text-gray-600 cursor-pointer transition-colors" />
              </span>
            </label>
            <div className="flex items-center gap-4">
              <input
                type="range"
                min={0}
                max={selectedCard ? selectedCard.vramGb : 16}
                value={minReserveVramGB}
                onChange={e => setMinReserveVramGB(+e.target.value)}
                onTouchMove={handleTouchMove}
                className="range range-secondary flex-grow"
                step={0.5}
              />
              {isAdvanced ? (
                <input
                  type="number"
                  value={minReserveVramGB}
                  onChange={e => setMinReserveVramGB(Math.min(selectedCard ? selectedCard.vramGb : 16, Math.max(0, +e.target.value)))}
                  className="input input-bordered w-20 text-center"
                  min={0}
                  max={selectedCard ? selectedCard.vramGb : 16}
                  step={0.5}
                />
              ) : (
                <span className="text-sm font-medium min-w-[3rem] text-center">
                  {minReserveVramGB}GB
                </span>
              )}
            </div>
          </div>

          {/* Max Length */}
          <div className="form-control">
            <label className="label">
              <span className="label-text font-medium">Max Length</span>
              <span 
                className="tooltip tooltip-bottom" 
                data-tip="Maximum sequence length (in tokens) that the model will process. Longer sequences require more VRAM for KV cache."
              >
                <Info className="w-4 h-4 text-gray-400 hover:text-gray-600 cursor-pointer transition-colors" />
              </span>
            </label>
            {isAdvanced ? (
              <div className="flex items-center gap-4">
                <input
                  type="range"
                  min={2048}
                  max={131072}
                  value={maxLength}
                  onChange={e => setMaxLength(+e.target.value)}
                  onTouchMove={handleTouchMove}
                  className="range range-success flex-grow"
                  step={1}
                />
                <input
                  type="number"
                  value={maxLength}
                  onChange={e => setMaxLength(Math.min(131072, Math.max(2048, +e.target.value)))}
                  className="input input-bordered w-24 text-center"
                  min={2048}
                  max={131072}
                />
              </div>
            ) : (
              <div className="flex items-center gap-4">
                <input
                  type="range"
                  min={0}
                  max={12}
                  value={[2048, 4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768, 40960, 65536, 98304, 131072].indexOf(maxLength)}
                  onChange={e => setMaxLength([2048, 4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768, 40960, 65536, 98304, 131072][Number(e.target.value)])}
                  onTouchMove={handleTouchMove}
                  className="range range-success range-with-marks flex-grow"
                />
                <span className="text-sm font-medium min-w-[4rem] text-center">
                  {maxLength.toLocaleString()}
                </span>
              </div>
            )}
          </div>

          {/* Concurrent Users */}
          <div className="form-control">
            <label className="label">
              <span className="label-text font-medium">Concurrent Users</span>
              <span 
                className="tooltip tooltip-bottom" 
                data-tip="Number of users concurrently using the model. The throughput speed will be divided among these users."
              >
                <Info className="w-4 h-4 text-gray-400 hover:text-gray-600 cursor-pointer transition-colors" />
              </span>
            </label>
            <div className="flex items-center gap-4">
              <input
                type="range"
                min={1}
                max={isAdvanced ? 1000 : 100}
                value={userCount}
                onChange={e => setUserCount(+e.target.value)}
                onTouchMove={handleTouchMove}
                className="range range-accent flex-grow"
                step={1}
              />
              {isAdvanced ? (
                <input
                  type="number"
                  value={userCount}
                  onChange={e => setUserCount(Math.min(1000, Math.max(1, +e.target.value)))}
                  className="input input-bordered w-20 text-center"
                  min={1}
                  max={1000}
                />
              ) : (
                <span className="text-sm font-medium min-w-[3rem] text-center">
                  {userCount}
                </span>
              )}
            </div>
          </div>
        </div>
      </div>

      {results?.error && (
        <div className="alert alert-error shadow-lg transition-all duration-300">
          <svg xmlns="http://www.w3.org/2000/svg" className="stroke-current shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span>{results.error}</span>
        </div>
      )}

      {results ? (
        <div className="card bg-base-100 shadow-lg p-4 sm:p-6 transition-all duration-300">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            <div className="stat">
              <div className="stat-title">Total VRAM Required</div>
              <div className="stat-value">{results.totalVram.toFixed(2)} GB</div>
              <progress
                className="progress progress-info w-full"
                value={(results.usableVram / results.totalVram) * 100}
                max={100}
              />
              <div className="stat-desc">Usable: {results.usableVram.toFixed(2)} GB</div>
            </div>
            <div className="stat">
              <div className="stat-title">Model Weights</div>
              <div className="stat-value">{results.modelVram.toFixed(2)} GB</div>
              <progress
                className="progress progress-success w-full"
                value={(results.modelVram / results.totalVram) * 100}
                max={100}
              />
              <div className="stat-desc">Quant: {
                quantType === 'int4' ? '4-bit' :
                quantType === 'fp8' ? '8-bit' :
                '16-bit'
              }</div>
            </div>
            
            {!results?.error && (
              <>
                <div className="stat">
                  <div className="stat-title">Full-length Context KV Cache</div>
                  <div className="stat-value">{results.kvCacheVram.toFixed(2)} GB</div>
                  <progress
                    className="progress progress-warning w-full"
                    value={(results.kvCacheVram / results.totalVram) * 100}
                    max={100}
                  />
                  <div className="stat-desc">
                    Quant: {
                      selectedCard?.kvQuantType === 'fp8' ? '8-bit' :
                      selectedCard?.kvQuantType === 'fp16' ? '16-bit' :
                      selectedCard?.kvQuantType === 'int8' ? '8-bit' :
                      selectedCard?.kvQuantType === 'int4' ? '4-bit' : '32-bit'
                    }
                  </div>
                </div>

                <div className="stat">
                  <div className="stat-title">Throughput Gen Speed</div>
                  <div className="stat-value">{results.genSpeed.toFixed(0)} tok/s</div>
                  <div className="stat-desc">parallel eff. ×{results.membwScaling.toFixed(2)}</div>
                </div>

                <div className="stat">
                  <div className="stat-title">Throughput Prompt Speed</div>
                  <div className="stat-value">{results.promptSpeed.toFixed(0)} tok/s</div>
                  <div className="stat-desc">parallel eff. ×{results.ppScaling.toFixed(2)}</div>
                </div>

                <div className="stat">
                  <div className="stat-title">Max Concurrent Token Capacity</div>
                  <div className="stat-value">{results.maxTokenCountSimultaneous.toFixed(0)} toks</div>
                  <div className="stat-desc">Full-length Context ×{results.fullLengthGenCount.toFixed(2)}</div>
                </div>

                <div className="stat">
                  <div className="stat-title">Shared Gen Speed</div>
                  <div className="stat-value">{results.sharedGen.toFixed(1)} tok/s</div>
                  <div className="stat-desc">concurrent user ×{userCount}</div>
                </div>

                <div className="stat">
                  <div className="stat-title">Shared Prompt Speed</div>
                  <div className="stat-value">{results.sharedPrompt.toFixed(1)} tok/s</div>
                  <div className="stat-desc">concurrent user ×{userCount}</div>
                </div>
              </>
            )}

            {selectedCard && selectedModel && (
                <div className="stat">
                  <div className="stat-title">{selectedCard.name}</div>
                  <div className="stat-value">{selectedCard.memoryBandwidthGBs} GB/s</div>
                  <div className="stat-desc">
                    {selectedCard?.processPower.fp16 ? 'FP16' : 'FP32'}: {selectedCard.processPower.fp16 || selectedCard.processPower.fp32} TFLOPS
                  </div>
                </div>
            )}
          </div>
        </div>
      ) : (
        <div className="text-center text-gray-400 py-8 transition-all duration-300">
          Select a card & model to see results
        </div>
      )}
    </div>
  );
}
