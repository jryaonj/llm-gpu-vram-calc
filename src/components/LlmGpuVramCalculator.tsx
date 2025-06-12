import { useState, useEffect } from 'react';
import { Info, Github, Settings2, ChevronDown, ChevronUp } from 'lucide-react';

// --- CONFIG: please populate these arrays from your spreadsheet export ---

import type { GPUCard, ModelDef, CalcResults } from '../types/index.ts'; // Assuming GPUCard is defined in another file

import { gpuCards } from '../data/gpuCards.ts'; // Import GPU cards data
import { modelDefs } from '../data/modelDefs.ts'; // Import model definitions

// // Estimate KV cache size per token in FP8 based on model parameters
// function estimateKvCacheSizePerToken(_totalParamsB: number, quantType: 'fp16' | 'fp8' | 'int8' | 'int4'): number {
//   // Simplified heuristic: use a constant FP8 baseline irrespective of total parameters.
//   // Empirically, ~8–12 bytes per token (FP8) covers most architectures; choose 8 for simplicity.
//   const baseBytesFp8 = 8; // bytes/token when KV stored in 8-bit precision

//   const quantMultiplier = quantType === 'fp16' ? 2
//     : (quantType === 'fp8' || quantType === 'int8') ? 1
//     : 0.5; // int4

//   return Math.round(baseBytesFp8 * quantMultiplier);
// }

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

// Estimate model size (GB) from total parameters and quantization type
function estimateModelSizeGB(totalParamsB: number, quantType: 'fp16' | 'fp8' | 'int8' | 'int4'): number {
  const bytesPerParam = quantType === 'fp16' ? 2
    : quantType === 'fp8' ? 1
    : quantType === 'int8' ? 1
    : 0.5; // int4

  // Additional safety multiplier similar to computeModelVramGB for fp8/int4 worst-case
  const quantFactor = (quantType === 'fp8' || quantType === 'int4') ? ((0.5 + 3 / 32) / 0.5) : 1;

  const sizeGB = (totalParamsB * bytesPerParam * quantFactor);
  return Math.round(sizeGB * 100) / 100; // round to 2 decimals for UI neatness
}

// Compute per-token KV cache size in bytes for FP8 given transformer parameters.
// Formula: layers * numKVHeads * headDim (bytes per value) * 1 (key OR value).
// The computeKvCacheVramGB function later multiplies by 2 for both K & V.
function computeKvSizeFp8FromParams(layers: number, numKVHeads: number, headDim: number): number {
  return layers * numKVHeads * headDim;
}

export default function LLMVRAMCalculator() {
  /* ─── default selections ────────────────────────────────────────────── */
  const defaultCard  = gpuCards.find(c => c.name === 'RTX3090 24G') ?? null;
  const defaultModel = modelDefs.find(m => m.name === 'Qwen3-32B')  ?? null;

  // selections
  const [selectedCard,  setSelectedCard]  = useState<GPUCard | null>(defaultCard);
  const [selectedModel, setSelectedModel] = useState<ModelDef | null>(defaultModel);
  const [quantType, setQuantType] = useState<'fp16' | 'fp8' | 'int8' | 'int4'>('int4');
  // kv cache quantization type, default to fp8
  const [kvQuantType, _setKvQuantType] = useState<'fp16' | 'fp8' | 'int8' | 'int4'>(defaultCard?.kvQuantType && ['fp16', 'fp8', 'int8', 'int4'].includes(defaultCard.kvQuantType) ? defaultCard.kvQuantType as 'fp16' | 'fp8' | 'int8' | 'int4' : 'fp8');
  const [maxLength, setMaxLength] = useState<number>(8192);
  const [userCount, setUserCount] = useState<number>(10);

  // outputs
  const [results, setResults] = useState<CalcResults | null>(null);

  // Add these to the state declarations in LLMVRAMCalculator:
  const [vramUtilProportion, setVramUtilProportion] = useState<number>(0.9); // 90% default
  const [minReserveVramGB, setMinReserveVramGB] = useState<number>(2); // 2GB default
  const [parallelGPUs, setParallelGPUs] = useState<number>(1);
  const [isAdvanced, setIsAdvanced] = useState(false);

  // Custom model and GPU inputs
  const [useCustomModel, setUseCustomModel] = useState(false);
  const [useCustomGPU, setUseCustomGPU] = useState(false);
  
  // Custom model parameters
  const [customModelSizeGB, setCustomModelSizeGB] = useState<number>(32.8); // Qwen3-32B model default
  const [customTotalParamsB, setCustomTotalParamsB] = useState<number>(32.8); // Qwen3-32B params
  const [customLayers, setCustomLayers] = useState<number>(64);
  const [customNumKVHeads, setCustomNumKVHeads] = useState<number>(8);
  const [customHeadDim, setCustomHeadDim] = useState<number>(128);
  const [customActiveParamsB, setCustomActiveParamsB] = useState<number>(32.8); // 32.8B active params
  const [customPerKVsizeFp8, setCustomPerKVsizeFp8] = useState<number>(65536); // bytes per token for KV cache in FP8 (64KB default)
  const [kvSizeUserModified, setKvSizeUserModified] = useState<boolean>(false); // Track if user manually set KV size
  const [modelSizeUserModified, setModelSizeUserModified] = useState<boolean>(false); // Track if user manually set model size
  
  // Custom GPU parameters  
  const [customVramGB, setCustomVramGB] = useState<number>(24); // 24GB default
  const [customMemoryBandwidthGBs, setCustomMemoryBandwidthGBs] = useState<number>(900); // 900 GB/s default
  const [customProcessPowerFP16, setCustomProcessPowerFP16] = useState<number>(100); // 100 TFLOPS default
  const [customKvQuantType, setCustomKvQuantType] = useState<'fp16' | 'fp8' | 'int8' | 'int4'>('fp8');

  // Expand / collapse states for advanced parameter cards
  const [showCustomModelParams, setShowCustomModelParams] = useState<boolean>(true);
  const [showCustomGPUParams, setShowCustomGPUParams] = useState<boolean>(true);

  // Auto-update KV cache size estimate when model parameters or quantization changes
  useEffect(() => {
    if (useCustomModel && !kvSizeUserModified) {
      const estimatedKvSize = computeKvSizeFp8FromParams(customLayers, customNumKVHeads, customHeadDim);
      setCustomPerKVsizeFp8(estimatedKvSize);
    }
  }, [customLayers, customNumKVHeads, customHeadDim, useCustomModel, kvSizeUserModified]);

  // Auto-update Model size estimate when parameters or quantization changes
  useEffect(() => {
    if (useCustomModel && !modelSizeUserModified) {
      const estimatedModelSize = estimateModelSizeGB(customTotalParamsB, quantType);
      setCustomModelSizeGB(estimatedModelSize);
    }
  }, [customTotalParamsB, quantType, useCustomModel, modelSizeUserModified]);

  // Reset Model size user modification flag when switching to custom model
  useEffect(() => {
    if (useCustomModel) {
      setModelSizeUserModified(false);
    }
  }, [useCustomModel]);

  // Reset KV size user modification flag when switching to custom model
  useEffect(() => {
    if (useCustomModel) {
      setKvSizeUserModified(false);
    }
  }, [useCustomModel]);

  // Add touch event handler for range inputs
  const handleTouchMove = (e: React.TouchEvent) => {
    e.stopPropagation();
  };

  useEffect(() => {
    // Determine which model and card data to use
    const effectiveModel = useCustomModel ? {
      name: 'Custom Model',
      modelSizeGB: customModelSizeGB,
      totalParamsB: customTotalParamsB,
      activeParamsB: customActiveParamsB,
      perKVsizeFp8: customPerKVsizeFp8,
      quantType: quantType,
      quantBits: quantType === 'int4' ? 4 : quantType === 'fp8' ? 8 : quantType === 'int8' ? 8 : 16,
      hiddenSize: 4096, // reasonable default for calculations
      layers: customLayers,
      numKVHeads: customNumKVHeads,
      headDim: customHeadDim
    } as ModelDef : selectedModel;

    const effectiveCard = useCustomGPU ? {
      name: 'Custom GPU',
      vramGb: customVramGB,
      memoryBandwidthGBs: customMemoryBandwidthGBs,
      processPower: {
        fp16: customProcessPowerFP16,
        fp32: customProcessPowerFP16 / 2 // rough estimate
      },
      kvQuantType: customKvQuantType
    } as GPUCard : selectedCard;

    // Only calculate if we have both model and card (either selected or custom)
    if ((selectedCard || useCustomGPU) && (selectedModel || useCustomModel) && effectiveModel && effectiveCard) {
      // compute base values
      const modelVram = computeModelVramGB(effectiveModel, quantType);
      const kvVram = computeKvCacheVramGB(effectiveModel.hiddenSize, maxLength, kvQuantType, effectiveModel, effectiveCard);
      // const total = modelVram + kvVram;

      // Parallel scaling factors
      const ppScaling = Math.pow(parallelGPUs, 0.6);
      const membwScaling = Math.pow(parallelGPUs, 0.8);

      const totalVram = effectiveCard.vramGb * parallelGPUs

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
        (effectiveModel.quantBits === 4 && effectiveCard.processPower.fp16) ? 2 :
        (effectiveModel.quantBits === 8 && !effectiveCard.processPower.fp8) ? 2 :
        1;

      const baseGenSpeed = (effectiveCard.memoryBandwidthGBs / effectiveModel.activeParamsB) * genQuant;
      const genSpeed = baseGenSpeed * membwScaling;

      // noe vLLM backend only use fp16 for process power
      const ppow = effectiveCard.processPower.fp16 ?? effectiveCard.processPower.fp32 ?? 0;
      const basePromptSpeed = (ppow * 1000 / effectiveModel.totalParamsB) / Math.sqrt(2);
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
    } else {
      setResults(null);
    }
  }, [selectedCard, selectedModel, quantType, maxLength, userCount, vramUtilProportion, minReserveVramGB, parallelGPUs, 
      useCustomModel, useCustomGPU, customModelSizeGB, customTotalParamsB, customLayers, customNumKVHeads, customHeadDim,
      customActiveParamsB, customPerKVsizeFp8,
      customVramGB, customMemoryBandwidthGBs, customProcessPowerFP16, customKvQuantType, kvQuantType]);

  return (
    <div className="max-w-4xl mx-auto p-4 sm:p-6 space-y-6">
      {/* Header with centered title and GitHub link at top-right */}
      <div className="relative mb-4">
        <h1 className="text-center text-2xl sm:text-3xl font-extrabold bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
          LLM GPU VRAM &amp; Speed Calculator
        </h1>
        <a
          href="https://github.com/jryaonj/llm-gpu-vram-calc"
          target="_blank"
          rel="noopener noreferrer"
          className="btn btn-ghost btn-sm gap-2 hover:scale-105 transition-transform absolute right-0 top-0"
        >
          <Github className="w-5 h-5" />
          <span className="hidden sm:inline">fork me</span>
        </a>
      </div>

      <div className="card bg-base-100 shadow-lg p-4 sm:p-6 transition-all duration-300 relative">
        {/* Advanced toggle icon inside card */}
        <button
          onClick={() => setIsAdvanced(!isAdvanced)}
          data-tip={isAdvanced ? 'Hide advanced settings' : 'Show advanced settings'}
          className={`btn btn-xs gap-1 absolute top-2 right-2 tooltip tooltip-left ${
            isAdvanced ? 'btn-primary' : 'btn-ghost'
          }`}
        >
          <Settings2
            className={`w-4 h-4 transform transition-transform ${
              isAdvanced ? 'rotate-90' : ''
            }`}
          />
        </button>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Model Selection */}
          <div className="form-control">
            <label className="label">
              <span className="label-text font-medium">Model</span>
              {isAdvanced && (
                <label className="label cursor-pointer gap-2">
                  <span className="label-text text-xs">Custom</span>
                  <input
                    type="checkbox"
                    className="toggle toggle-xs toggle-primary"
                    checked={useCustomModel}
                    onChange={e => {
                      setUseCustomModel(e.target.checked);
                      if (e.target.checked) setSelectedModel(null);
                    }}
                  />
                </label>
              )}
            </label>
            {useCustomModel && isAdvanced ? (
              <input
                type="text"
                value="Custom Model"
                readOnly
                className="input input-bordered w-full bg-base-200"
              />
            ) : (
              <select
                value={selectedModel?.name || ""}
                onChange={e => setSelectedModel(modelDefs.find(m => m.name === e.target.value) || null)}
                className="select select-bordered w-full transition-all duration-200 hover:border-primary focus:border-primary"
                disabled={useCustomModel}
              >
                <option value="">Select a model</option>
                {modelDefs.map(m => <option key={m.name}>{m.name}</option>)}
              </select>
            )}
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
              <div className="flex items-center gap-2">
                <span
                  className="tooltip tooltip-bottom"
                  data-tip="Card marked with * are theoretical estimates and may not even properly execute."
                >
                  <Info className="w-4 h-4 text-gray-400 hover:text-gray-600 cursor-pointer transition-colors" />
                </span>
                {isAdvanced && (
                  <label className="label cursor-pointer gap-2">
                    <span className="label-text text-xs">Custom</span>
                    <input
                      type="checkbox"
                      className="toggle toggle-xs toggle-primary"
                      checked={useCustomGPU}
                      onChange={e => {
                        setUseCustomGPU(e.target.checked);
                        if (e.target.checked) setSelectedCard(null);
                      }}
                    />
                  </label>
                )}
              </div>
            </label>
            {useCustomGPU && isAdvanced ? (
              <input
                type="text"
                value="Custom GPU"
                readOnly
                className="input input-bordered w-full bg-base-200"
              />
            ) : (
              <select
                onChange={e => {
                  const card = gpuCards.find(c => c.name === e.target.value) || null;
                  setSelectedCard(card);
                  if (card) {
                    _setKvQuantType(card.kvQuantType as 'fp16' | 'fp8' | 'int8' | 'int4' || 'fp16');
                  }
                }}
                className="select select-bordered w-full transition-all duration-200 hover:border-primary focus:border-primary"
                disabled={useCustomGPU}
              >
                <option value="">Select a card</option>
                {gpuCards.map(c => <option key={c.name}>{c.name}</option>)}
              </select>
            )}
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

        {/* Custom Model Parameters */}
        {isAdvanced && useCustomModel && (
          <div className="mt-6 p-4 bg-base-200 rounded-lg">
            <div className="relative mb-4 flex items-center">
              {/* centred title */}
              <h3 className="text-lg font-semibold text-primary text-center flex-grow">
                Custom Model Parameters
              </h3>
              {/* toggle button right-aligned */}
              <button
                type="button"
                onClick={() => setShowCustomModelParams(s => !s)}
                className="btn btn-ghost btn-xs absolute right-0"
              >
                {showCustomModelParams ? (
                  <ChevronUp className="w-4 h-4" />
                ) : (
                  <ChevronDown className="w-4 h-4" />
                )}
              </button>
            </div>
            {showCustomModelParams && (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="form-control order-8">
                  <label className="label">
                    <span className="label-text font-medium">Model Size (GB)</span>
                    <span 
                      className="tooltip tooltip-bottom" 
                      data-tip={`Estimated from total parameters and quantization. Click restore to recalc.`}
                    >
                      <Info className="w-4 h-4 text-gray-400 hover:text-gray-600 cursor-pointer transition-colors" />
                    </span>
                  </label>
                  <div className="flex gap-2">
                    <input
                      type="number"
                      value={customModelSizeGB}
                      onChange={e => {
                        setCustomModelSizeGB(Math.max(0.1, +e.target.value));
                        setModelSizeUserModified(true);
                      }}
                      className="input input-bordered flex-1"
                      min={0.1}
                      step={0.1}
                    />
                    <button
                      type="button"
                      onClick={() => {
                        const est = estimateModelSizeGB(customTotalParamsB, quantType);
                        setCustomModelSizeGB(est);
                        setModelSizeUserModified(false);
                      }}
                      className="btn btn-ghost btn-sm"
                      title="Restore estimated value"
                    >
                      ↻
                    </button>
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    Est: {estimateModelSizeGB(customTotalParamsB, quantType)} GB ({quantType.toUpperCase()})
                  </div>
                </div>
                
                <div className="form-control order-1">
                  <label className="label">
                    <span className="label-text font-medium">Total Params (B)</span>
                    <span 
                      className="tooltip tooltip-bottom" 
                      data-tip="Total number of parameters in billions (e.g., 7 for 7B model)."
                    >
                      <Info className="w-4 h-4 text-gray-400 hover:text-gray-600 cursor-pointer transition-colors" />
                    </span>
                  </label>
                  <input
                    type="number"
                    value={customTotalParamsB}
                    onChange={e => setCustomTotalParamsB(Math.max(0.1, +e.target.value))}
                    className="input input-bordered w-full"
                    min={0.1}
                    step={0.1}
                  />
                </div>
                
                <div className="form-control order-2">
                  <label className="label">
                    <span className="label-text font-medium">Layers</span>
                    <span
                      className="tooltip tooltip-bottom"
                      data-tip="Number of transformer layers."
                    >
                      <Info className="w-4 h-4 text-gray-400 hover:text-gray-600 cursor-pointer transition-colors" />
                    </span>
                  </label>
                  <input
                    type="number"
                    value={customLayers}
                    onChange={e => setCustomLayers(Math.max(1, +e.target.value))}
                    className="input input-bordered w-full"
                    min={1}
                    step={1}
                  />
                </div>
                
                <div className="form-control order-3">
                  <label className="label">
                    <span className="label-text font-medium">KV Heads</span>
                    <span
                      className="tooltip tooltip-bottom"
                      data-tip="Number of KV heads used in attention (may differ from total heads)."
                    >
                      <Info className="w-4 h-4 text-gray-400 hover:text-gray-600 cursor-pointer transition-colors" />
                    </span>
                  </label>
                  <input
                    type="number"
                    value={customNumKVHeads}
                    onChange={e => setCustomNumKVHeads(Math.max(1, +e.target.value))}
                    className="input input-bordered w-full"
                    min={1}
                    step={1}
                  />
                </div>

                <div className="form-control order-4">
                  <label className="label">
                    <span className="label-text font-medium">Head Dim</span>
                    <span
                      className="tooltip tooltip-bottom"
                      data-tip="Dimension per attention head (e.g., 128)."
                    >
                      <Info className="w-4 h-4 text-gray-400 hover:text-gray-600 cursor-pointer transition-colors" />
                    </span>
                  </label>
                  <input
                    type="number"
                    value={customHeadDim}
                    onChange={e => setCustomHeadDim(Math.max(1, +e.target.value))}
                    className="input input-bordered w-full"
                    min={1}
                    step={1}
                  />
                </div>

                <div className="form-control order-6">
                  <label className="label">
                    <span className="label-text font-medium">Active Params (B)</span>
                    <span 
                      className="tooltip tooltip-bottom" 
                      data-tip="Active parameters during inference (usually same as total for non-MoE models)."
                    >
                      <Info className="w-4 h-4 text-gray-400 hover:text-gray-600 cursor-pointer transition-colors" />
                    </span>
                  </label>
                  <input
                    type="number"
                    value={customActiveParamsB}
                    onChange={e => setCustomActiveParamsB(Math.max(0.1, +e.target.value))}
                    className="input input-bordered w-full"
                    min={0.1}
                    step={0.1}
                  />
                </div>

                <div className="form-control order-7">
                  <label className="label">
                    <span className="label-text font-medium">KV Size (KB/token)</span>
                    <span 
                      className="tooltip tooltip-bottom" 
                      data-tip={`KV cache size per token (in KB) for ${quantType.toUpperCase()} quantization. Auto-inferred from layers × KV heads × head dim. Click restore to recalc.`}
                    >
                      <Info className="w-4 h-4 text-gray-400 hover:text-gray-600 cursor-pointer transition-colors" />
                    </span>
                  </label>
                  <div className="flex gap-2">
                    <input
                      type="number"
                      value={(customPerKVsizeFp8 / 1024).toFixed(0)}
                      onChange={e => {
                        const kb = Math.max(1, +e.target.value);
                        setCustomPerKVsizeFp8(kb * 1024);
                        setKvSizeUserModified(true);
                      }}
                      className="input input-bordered flex-1"
                      min={1}
                      step={1}
                    />
                    <button
                      type="button"
                      onClick={() => {
                        const estimatedKvSize = computeKvSizeFp8FromParams(customLayers, customNumKVHeads, customHeadDim);
                        setCustomPerKVsizeFp8(estimatedKvSize);
                        setKvSizeUserModified(false);
                      }}
                      className="btn btn-ghost btn-sm"
                      title="Restore estimated value"
                    >
                      ↻
                    </button>
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    Est: {(computeKvSizeFp8FromParams(customLayers, customNumKVHeads, customHeadDim) / 1024).toFixed(0)} KB (
                      {quantType === 'int4' ? '4-bit' :
                       (quantType === 'fp8' || quantType === 'int8') ? '8-bit' : '16-bit'}
                    )
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Custom GPU Parameters */}
        {isAdvanced && useCustomGPU && (
          <div className="mt-6 p-4 bg-base-200 rounded-lg">
            <div className="relative mb-4 flex items-center">
              {/* centred title */}
              <h3 className="text-lg font-semibold text-primary text-center flex-grow">
                Custom GPU Parameters
              </h3>
              {/* expand / collapse button */}
              <button
                type="button"
                onClick={() => setShowCustomGPUParams(s => !s)}
                className="btn btn-ghost btn-xs absolute right-0"
              >
                {showCustomGPUParams ? (
                  <ChevronUp className="w-4 h-4" />
                ) : (
                  <ChevronDown className="w-4 h-4" />
                )}
              </button>
            </div>
            {showCustomGPUParams && (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="form-control">
                  <label className="label">
                    <span className="label-text font-medium">VRAM (GB)</span>
                    <span 
                      className="tooltip tooltip-bottom" 
                      data-tip="Total VRAM capacity per GPU in GB."
                    >
                      <Info className="w-4 h-4 text-gray-400 hover:text-gray-600 cursor-pointer transition-colors" />
                    </span>
                  </label>
                  <input
                    type="number"
                    value={customVramGB}
                    onChange={e => setCustomVramGB(Math.max(1, +e.target.value))}
                    className="input input-bordered w-full"
                    min={1}
                    step={1}
                  />
                </div>
                
                <div className="form-control">
                  <label className="label">
                    <span className="label-text font-medium">Memory BW (GB/s)</span>
                    <span 
                      className="tooltip tooltip-bottom" 
                      data-tip="Memory bandwidth in GB/s (e.g., 900 for H100, 600 for A100)."
                    >
                      <Info className="w-4 h-4 text-gray-400 hover:text-gray-600 cursor-pointer transition-colors" />
                    </span>
                  </label>
                  <input
                    type="number"
                    value={customMemoryBandwidthGBs}
                    onChange={e => setCustomMemoryBandwidthGBs(Math.max(1, +e.target.value))}
                    className="input input-bordered w-full"
                    min={10}
                    step={10}
                  />
                </div>
                
                <div className="form-control">
                  <label className="label">
                    <span className="label-text font-medium">FP16 TFLOPS</span>
                    <span 
                      className="tooltip tooltip-bottom" 
                      data-tip="FP16 compute performance in TFLOPS (e.g., 989 for H100, 312 for A100)."
                    >
                      <Info className="w-4 h-4 text-gray-400 hover:text-gray-600 cursor-pointer transition-colors" />
                    </span>
                  </label>
                  <input
                    type="number"
                    value={customProcessPowerFP16}
                    onChange={e => setCustomProcessPowerFP16(Math.max(1, +e.target.value))}
                    className="input input-bordered w-full"
                    min={1}
                    step={1}
                  />
                </div>
                
                <div className="form-control">
                  <label className="label">
                    <span className="label-text font-medium">KV Cache Quant</span>
                    <span 
                      className="tooltip tooltip-bottom" 
                      data-tip="KV cache quantization support (newer GPUs support FP8)."
                    >
                      <Info className="w-4 h-4 text-gray-400 hover:text-gray-600 cursor-pointer transition-colors" />
                    </span>
                  </label>
                  <select
                    value={customKvQuantType}
                    onChange={e => {
                      const val = e.target.value as 'fp16' | 'fp8' | 'int8' | 'int4';
                      setCustomKvQuantType(val);
                      _setKvQuantType(val); // synchronize overall kv quant when using custom GPU
                    }}
                    className="select select-bordered w-full"
                  >
                    <option value="fp8">FP8</option>
                    <option value="fp16">FP16</option>
                    <option value="int8">INT8</option>
                    <option value="int4">INT4</option>
                  </select>
                </div>
              </div>
            )}
          </div>
        )}
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
                      useCustomGPU ? (
                        customKvQuantType === 'fp8' ? '8-bit' :
                        customKvQuantType === 'fp16' ? '16-bit' :
                        customKvQuantType === 'int8' ? '8-bit' :
                        customKvQuantType === 'int4' ? '4-bit' : '32-bit'
                      ) : (
                        selectedCard?.kvQuantType === 'fp8' ? '8-bit' :
                        selectedCard?.kvQuantType === 'fp16' ? '16-bit' :
                        selectedCard?.kvQuantType === 'int8' ? '8-bit' :
                        selectedCard?.kvQuantType === 'int4' ? '4-bit' : '32-bit'
                      )
                    } ({kvQuantType.toUpperCase()})
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

            {((selectedCard && selectedModel) || (useCustomGPU && useCustomModel)) && (
                <div className="stat">
                  <div className="stat-title">{useCustomGPU ? 'Custom GPU' : selectedCard?.name}</div>
                  <div className="stat-value">{useCustomGPU ? customMemoryBandwidthGBs : selectedCard?.memoryBandwidthGBs} GB/s</div>
                  <div className="stat-desc">
                    FP16: {useCustomGPU ? customProcessPowerFP16 : (selectedCard?.processPower.fp16 || selectedCard?.processPower.fp32)} TFLOPS
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
