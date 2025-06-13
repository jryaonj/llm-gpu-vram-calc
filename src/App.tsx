import { useEffect, useState } from 'react';
import { Calculator, Github } from 'lucide-react';
import LLMVRAMCalculator from './components/LlmGpuVramCalculator.emoji';

function App() {
  const [_isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    setIsLoaded(true);
    
    // Update document title and meta tags for SEO
    document.title = '🚀 LLM GPU VRAM Calculator 🔥 - Epic VRAM Beast Mode Calculator! 💪⚡';
    
    // Add meta description for SEO
    const metaDescription = document.querySelector('meta[name="description"]');
    if (metaDescription) {
      metaDescription.setAttribute('content', '🔥 Calculate GPU memory like a boss! 💻⚡ Professional hardware calculator for AI legends! 🚀🤖💎');
    } else {
      const meta = document.createElement('meta');
      meta.name = 'description';
      meta.content = '🔥 Calculate GPU memory like a boss! 💻⚡ Professional hardware calculator for AI legends! 🚀🤖💎';
      document.head.appendChild(meta);
    }
  }, []);

  return (
    <div className="flex flex-col min-h-screen bg-gradient-to-br from-purple-100 via-pink-50 to-indigo-100">
      {/* Header */}
      <header className="bg-gradient-to-r from-purple-600 via-pink-500 to-indigo-600 border-b-4 border-yellow-400 sticky top-0 z-50 shadow-xl">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 bg-gradient-to-br from-yellow-400 to-orange-500 rounded-xl flex items-center justify-center shadow-lg transform hover:scale-110 transition-transform">
                <span className="text-2xl">🚀</span>
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white flex items-center gap-2">
                  🔥 LLM VRAM GPU Calculator 💪
                </h1>
                <p className="text-purple-100 flex items-center gap-1">
                  ⚡ Epic VRAM Beast Mode Calculator! 🤖💎✨
                </p>
              </div>
            </div>
            
            <a
              href="https://github.com/jryaonj/llm-vram-gpu-calculator"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-6 py-3 bg-white/20 hover:bg-white/30 text-white rounded-xl transition-all hover:scale-105 shadow-lg"
            >
              <Github className="w-4 h-4" />
              <span className="hidden sm:inline text-sm font-medium">⭐ GitHub</span>
            </a>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-grow max-w-7xl mx-auto px-6 py-8">
        <LLMVRAMCalculator />
      </main>

      {/* Footer */}
      <footer className="bg-gradient-to-r from-gray-800 via-gray-900 to-black border-t-4 border-yellow-400 py-6">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-2 text-sm text-gray-300">
              <span>© 2025 🔥 LLM VRAM GPU Calculator 💪</span>
              <span className="text-yellow-400">•</span>
              <span>⚡ Made for AI Legends! 🚀</span>
            </div>
            
            <div className="flex items-center gap-4 text-sm text-gray-300">
              <span>🛠️ Powered by</span>
              <div className="flex items-center gap-2">
                <span className="font-medium text-blue-400">⚛️ React</span>
                <span className="text-yellow-400">•</span>
                <span className="font-medium text-purple-400">🎨 DaisyUI</span>
              </div>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
