import { useState, useEffect } from 'react';

// --- CONFIG: please populate these arrays from your spreadsheet export ---

import type { GPUCard, ModelDef, CalcResults } from '../types/index.ts'; // Assuming GPUCard is defined in another file

import { gpuCards } from '../data/gpuCards.ts'; // Import GPU cards data
import { modelDefs } from '../data/modelDefs.ts'; // Import model definitions

function computeModelVramGB(model: ModelDef, quant: 'fp16' | 'fp8' | 'int8' | 'int4'): number {
  // If using the model's native quantization type
  if (quant === model.quantType) {
    return model.modelSizeGB;
  }

  // For other quantization types
  const bytesPerParam = quant === 'fp16' ? 2
    : quant === 'fp8' ? 1
    : quant === 'int8' ? 1
    : 0.5; // int4

  return model.totalParamsB * 1e9 * bytesPerParam / (1024 ** 3);
}

function computeKvCacheVramGB(hiddenSize: number, maxLength: number, quant: 'fp16' | 'fp8' | 'int8' | 'int4', model: ModelDef, card: GPUCard): number {
  const bytesPerValue = quant === 'fp16' ? 2
    : quant === 'fp8' ? 1
    : quant === 'int8' ? 1
    : 0.5;

  // Check if GPU supports FP8 for KV cache (Ampere and newer architectures)
  const supportsFp8KV = card.processPower.fp8 !== undefined;
  
  // If GPU doesn't support FP8 KV cache, use precalculated FP8 value * 2 for FP16
  if (!supportsFp8KV && model.perKVsizeFp8) {
    const baseKVSize = model.perKVsizeFp8; // This is the FP8 size per token
    const sizeMultiplier = quant === 'fp16' ? 2 : 
                          quant === 'int8' ? 1 : 
                          quant === 'int4' ? 0.5 : 1;
    // Calculate total KV cache size in GB
    // Notes: both k and v are stored, so we multiply by 2
    // Also, maxLength is the number of tokens
    // and baseKVSize is in bytes per token
    // Finally, we convert bytes to gigabytes
    return (baseKVSize * sizeMultiplier * maxLength * 2) / (1024 * 1024 * 1024);
  }

  // For GPUs that support FP8 or when using other quantization
  const totalBytes = 2 * hiddenSize * bytesPerValue * maxLength;
  return totalBytes / (1024 ** 3);  // in gigabytes
}

export default function LLMVRAMCalculator() {
  // selections
  const [selectedCard, setSelectedCard] = useState<GPUCard | null>(null);
  const [selectedModel, setSelectedModel] = useState<ModelDef | null>(null);
  const [quantType, setQuantType] = useState<'fp16' | 'fp8' | 'int8' | 'int4'>('fp16');
  const [maxLength, setMaxLength] = useState<number>(1024);
  const [userCount, setUserCount] = useState<number>(20);

  // outputs
  const [results, setResults] = useState<CalcResults | null>(null);

  // Add these to the state declarations in LLMVRAMCalculator:
  const [vramUtilProportion, setVramUtilProportion] = useState<number>(0.9); // 90% default
  const [minReserveVramGB, setMinReserveVramGB] = useState<number>(2); // 2GB default
  const [parallelGPUs, setParallelGPUs] = useState<number>(1);

  useEffect(() => {
    if (selectedCard && selectedModel) {
      // compute base values
      const modelVram = computeModelVramGB(selectedModel, quantType);
      const kvVram = computeKvCacheVramGB(selectedModel.hiddenSize, maxLength, quantType, selectedModel, selectedCard);
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

      const ppow = selectedCard.processPower[quantType] ?? selectedCard.processPower.fp16 ?? 0;
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
    <div className="max-w-2xl mx-auto p-4">
      <h2 className="text-2xl font-bold mb-4">LLM VRAM & Speed Calculator</h2>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-6">
        <div>
          <label className="block mb-1">GPU Card</label>
          <select
            className="w-full p-2 rounded border"
            onChange={(e) => {
              const card = gpuCards.find(c => c.name === e.target.value) || null;
              setSelectedCard(card);
            }}
          >
            <option value="">Select a card</option>
            {gpuCards.map(c => (
              <option key={c.name} value={c.name}>{c.name}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="block mb-1">Parallel GPUs</label>
          <input
            type="range"
            min="1"
            max="8"
            value={parallelGPUs}
            onChange={e => setParallelGPUs(Number(e.target.value))}
            className="w-full"
          />
          <span className="text-sm text-gray-600">{parallelGPUs} GPU{parallelGPUs > 1 ? 's' : ''}</span>
        </div>
        <div>
          <label className="block mb-1">VRAM Utilization (%)</label>
          <input
            type="range"
            min="10"
            max="100"
            value={vramUtilProportion * 100}
            onChange={e => setVramUtilProportion(Number(e.target.value) / 100)}
            className="w-full"
          />
          <span className="text-sm text-gray-600">{(vramUtilProportion * 100).toFixed(0)}%</span>
        </div>
        <div>
          <label className="block mb-1">Min Reserve VRAM (GB)</label>
          <input
            type="number"
            min="0"
            step="0.5"
            value={minReserveVramGB}
            onChange={e => setMinReserveVramGB(Number(e.target.value))}
            className="w-full p-2 rounded border"
          />
        </div>
        <div>
          <label className="block mb-1">Model</label>
          <select
            className="w-full p-2 rounded border"
            onChange={(e) => {
              const m = modelDefs.find(m => m.name === e.target.value) || null;
              setSelectedModel(m);
            }}
          >
            <option value="">Select a model</option>
            {modelDefs.map(m => (
              <option key={m.name} value={m.name}>{m.name}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="block mb-1">Quantization</label>
          <select
            value={quantType}
            onChange={(e) => setQuantType(e.target.value as any)}
            className="w-full p-2 rounded border"
          >
            <option value="fp16">FP16</option>
            <option value="fp8">FP8</option>
            <option value="int8">INT8</option>
            <option value="int4">INT4</option>
          </select>
        </div>
        <div>
          <label className="block mb-1">Max Length (tokens)</label>
          <input
            type="number"
            value={maxLength}
            onChange={e => setMaxLength(+e.target.value)}
            className="w-full p-2 rounded border"
          />
        </div>
        <div>
          <label className="block mb-1">Concurrent Users</label>
          <input
            type="number"
            value={userCount}
            onChange={e => setUserCount(+e.target.value)}
            className="w-full p-2 rounded border"
          />
        </div>
      </div>

      {results && (
        <div className="space-y-2">
          <p>Total GPUs: {parallelGPUs}</p>
          <p>Total VRAM: {results.totalVram.toFixed(2)} GB </p>
          
          <p className={results.totalVram > results.usableVram ? "text-red-600 font-bold" : ""}>
            Usable VRAM: {results.usableVram.toFixed(2)} GB ({(results.usableVram/parallelGPUs).toFixed(2)} GB/GPU)
          </p>

          <p>Model Required VRAM (Total): {results.modelVram.toFixed(2)} GB ({(results.modelVram/parallelGPUs).toFixed(2)} GB/GPU)</p>
          <p>Full-length KV Cache Required VRAM (Total): {results.kvCacheVram.toFixed(2)} GB ({(results.kvCacheVram/parallelGPUs).toFixed(2)} GB/GPU)</p>
          
          <p>Usable KV Cache VRAM: {results.usableKvCacheVram.toFixed(2)} GB ({(results.usableKvCacheVram/parallelGPUs).toFixed(2)} GB/GPU)</p>
          
          {/* estimation */}
          <p>Max Concurrent Full-length Requests: {results.fullLengthGenCount.toFixed(2)}</p>
          <p>Max Simultaneous Tokens: {results.maxTokenCountSimultaneous.toFixed(0)}</p>

          <p>Generate Speed ({}× scaling ~ {results.membwScaling.toFixed(2)}): {results.genSpeed.toFixed(0)} tok/s</p>
          <p>Prompt Speed ({}× scaling ~ {results.ppScaling.toFixed(2)}): {results.promptSpeed.toFixed(0)} tok/s</p>
          <p>Shared Gen Speed  ( ÷ {userCount} ): {results.sharedGen.toFixed(0)} tok/s/user</p>
          <p>Shared Prompt Speed ( ÷ {userCount} ): {results.sharedPrompt.toFixed(0)} tok/s/user</p>
        </div>
      )}
      {!results && <p className="text-gray-500">Select card and model to see results.</p>}
    </div>
  );
}
