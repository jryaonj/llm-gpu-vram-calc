import { useEffect, useState } from 'react';
import { Calculator, Github } from 'lucide-react';
import LLMVRAMCalculator from './components/LlmGpuVramCalculator';

function App() {
  const [_isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    setIsLoaded(true);
    
    // Update document title and meta tags for SEO
    document.title = 'LLM GPU VRAM Calculator - VRAM requirements and performance estimation for LLM inference';
    
    // Add meta description for SEO
    const metaDescription = document.querySelector('meta[name="description"]');
    if (metaDescription) {
      metaDescription.setAttribute('content', 'Calculate GPU memory requirements and performance for Large Language Models. Professional hardware calculator for AI developers.');
    } else {
      const meta = document.createElement('meta');
      meta.name = 'description';
      meta.content = 'Calculate GPU memory requirements and performance for Large Language Models. Professional hardware calculator for AI developers.';
      document.head.appendChild(meta);
    }
  }, []);

  return (
    <div className="flex flex-col min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center">
                <Calculator className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-semibold text-gray-900">LLM VRAM GPU Calculator</h1>
                <p className="text-sm text-gray-500">VRAM requirements and performance estimation for LLM inference</p>
              </div>
            </div>
            
            <a
              href="https://github.com/jryaonj/llm-vram-gpu-calculator"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-4 py-2 text-gray-600 hover:text-gray-900 hover:bg-gray-50 rounded-lg transition-colors"
            >
              <Github className="w-4 h-4" />
              <span className="hidden sm:inline text-sm">GitHub</span>
            </a>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-grow max-w-7xl mx-auto px-6 py-8">
        <LLMVRAMCalculator />
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 py-6">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-1 text-sm text-gray-500">
              <span>© 2025 LLM VRAM GPU Calculator</span>
              <span className="text-gray-300">•</span>
              <span>Calculations for AI developers</span>
            </div>
            
            <div className="flex items-center gap-4 text-sm text-gray-500">
              <span>Powered by</span>
              <div className="flex items-center gap-2">
                <span className="font-medium text-blue-600">React</span>
                <span className="text-gray-300">•</span>
                <span className="font-medium text-blue-500">DaisyUI</span>
              </div>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
