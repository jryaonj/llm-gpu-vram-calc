import { useEffect, useState } from 'react';
import { Calculator, Github } from 'lucide-react';
import LLMVRAMCalculator from './components/LlmGpuVramCalculator';
import LLMVRAMCalculatorEmoji from './components/LlmGpuVramCalculatorEmoji';

function App() {
  const [_isLoaded, setIsLoaded] = useState(false);
  const [useEmoji, setUseEmoji] = useState(false);

  useEffect(() => {
    setIsLoaded(true);
    
    // Update document title and meta tags for SEO
    document.title = useEmoji
      ? '🚀 LLM GPU VRAM Calculator 🔥 - Epic VRAM Beast Mode Calculator! 💪⚡'
      : 'LLM GPU VRAM Calculator - VRAM requirements and performance estimation for LLM inference';
    
    // Add meta description for SEO
    const metaDescription = document.querySelector('meta[name="description"]');
    if (metaDescription) {
      metaDescription.setAttribute(
        'content',
        useEmoji
          ? '🔥 Calculate GPU memory like a boss! 💻⚡ Professional hardware calculator for AI legends! 🚀🤖💎'
          : 'Calculate GPU memory requirements and performance for Large Language Models. Professional hardware calculator for AI developers.'
      );
    } else {
      const meta = document.createElement('meta');
      meta.name = 'description';
      meta.content = useEmoji
        ? '🔥 Calculate GPU memory like a boss! 💻⚡ Professional hardware calculator for AI legends! 🚀🤖💎'
        : 'Calculate GPU memory requirements and performance for Large Language Models. Professional hardware calculator for AI developers.';
      document.head.appendChild(meta);
    }
  }, [useEmoji]);

  return (
    <div
      className={
        useEmoji
          ? 'flex flex-col min-h-screen bg-gradient-to-br from-purple-100 via-pink-50 to-indigo-100'
          : 'flex flex-col min-h-screen bg-gray-50'
      }
    >
      {/* Header */}
      <header
        className={
          useEmoji
            ? 'bg-gradient-to-r from-purple-600 via-pink-500 to-indigo-600 border-b-4 border-yellow-400 sticky top-0 z-50 shadow-xl'
            : 'bg-white border-b border-gray-200 sticky top-0 z-50'
        }
      >
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              {useEmoji ? (
                <div className="w-12 h-12 bg-gradient-to-br from-yellow-400 to-orange-500 rounded-xl flex items-center justify-center shadow-lg transform hover:scale-110 transition-transform">
                  <span className="text-2xl">🚀</span>
                </div>
              ) : (
                <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center">
                  <Calculator className="w-5 h-5 text-white" />
                </div>
              )}
              <div>
                {useEmoji ? (
                  <>
                    <h1 className="text-2xl font-bold text-white flex items-center gap-2">
                      🔥 LLM GPU VRAM Calculator 💪
                    </h1>
                    <p className="text-purple-100 flex items-center gap-1">
                      ⚡ Epic VRAM Beast Mode Calculator! 🤖💎✨
                    </p>
                  </>
                ) : (
                  <>
                    <h1 className="text-xl font-semibold text-gray-900">LLM GPU VRAM Calculator</h1>
                    <p className="text-sm text-gray-500">VRAM requirements and performance estimation for LLM inference</p>
                  </>
                )}
              </div>
            </div>
            
            <div className="flex items-center gap-2">
              <button
                onClick={() => setUseEmoji((prev) => !prev)}
                className="flex items-center gap-2 px-4 py-2 text-gray-600 hover:text-gray-900 hover:bg-gray-50 rounded-lg transition-colors"
              >
                {useEmoji ? '🙂 Normal' : '🎉 Emoji'}
              </button>
              <a
                href="https://github.com/jryaonj/llm-gpu-vram-calculator"
                target="_blank"
                rel="noopener noreferrer"
                className={
                  useEmoji
                    ? 'flex items-center gap-2 px-6 py-3 bg-white/20 hover:bg-white/30 text-white rounded-xl transition-all hover:scale-105 shadow-lg'
                    : 'flex items-center gap-2 px-4 py-2 text-gray-600 hover:text-gray-900 hover:bg-gray-50 rounded-lg transition-colors'
                }
              >
                <Github className="w-4 h-4" />
                <span className={useEmoji ? 'hidden sm:inline text-sm font-medium' : 'hidden sm:inline text-sm'}>{useEmoji ? '⭐ GitHub' : 'GitHub'}</span>
              </a>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-grow max-w-7xl mx-auto px-6 py-8">
        {useEmoji ? <LLMVRAMCalculatorEmoji /> : <LLMVRAMCalculator />}
      </main>

      {/* Footer */}
      <footer
        className={
          useEmoji
            ? 'bg-gradient-to-r from-gray-800 via-gray-900 to-black border-t-4 border-yellow-400 py-6'
            : 'bg-white border-t border-gray-200 py-6'
        }
      >
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-1 text-sm text-gray-500">
              {useEmoji ? (
                <>
                  <span className="text-gray-300">© 2025 🔥 LLM GPU VRAM Calculator 💪</span>
                  <span className="text-yellow-400">•</span>
                  <a href="https://jryaonj.github.io" target="_blank" rel="noopener noreferrer" className="hover:underline">
                    jryaonj
                  </a>
                  <span className="text-yellow-400">•</span>
                  <span className="text-gray-300">⚡ Made for AI Legends! 🚀</span>
                </>
              ) : (
                <>
                  <span>© 2025 LLM GPU VRAM Calculator</span>
                  <span className="text-gray-300">•</span>
                  <a href="https://jryaonj.github.io" target="_blank" rel="noopener noreferrer" className="hover:underline">
                    jryaonj
                  </a>
                  <span className="text-gray-300">•</span>
                  <span>Calculations for AI developers</span>
                  <span className="text-gray-300">•</span>
                </>
              )}
            </div>
            
            <div className="flex items-center gap-4 text-sm text-gray-500">
              {useEmoji ? (
                <>
                  <span className="text-gray-300">🛠️ Powered by</span>
                  <div className="flex items-center gap-2">
                    <a href="https://react.dev" className="font-medium text-blue-400 hover:underline">⚛️ React</a>
                    <span className="text-yellow-400">•</span>
                    <a href="https://daisyui.com" className="font-medium text-purple-400 hover:underline">🎨 DaisyUI</a>
                    <span className="text-yellow-400">•</span>
                    <a href="https://cursor.com" className="font-medium text-purple-400 hover:underline">🤖 CursorAI</a>
                  </div>
                </>
              ) : (
                <>
                  <span>Powered by</span>
                  <div className="flex items-center gap-2">
                    <a href="https://react.dev" className="font-medium text-blue-600 hover:underline">React</a>
                    <span className="text-gray-300">•</span>
                    <a href="https://daisyui.com" className="font-medium text-blue-500 hover:underline">DaisyUI</a>
                    <span className="text-gray-300">•</span>
                    <a href="https://cursor.com" className="font-medium text-blue-500 hover:underline">CursorAI</a>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
