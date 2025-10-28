import { useState } from 'react';
import { Upload, Link, FileText, Info } from 'lucide-react';

export function UploadDocuments() {
  const [username, setUsername] = useState('');
  const [botName, setBotName] = useState('');
  const [websiteUrl, setWebsiteUrl] = useState('');
  const [urlType, setUrlType] = useState<'single' | 'sitemap'>('single');
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    // Handle file drop
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-white mb-6">Create a New Knowledge Base</h2>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left Column */}
        <div className="space-y-6">
          {/* User Information */}
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-zinc-300 mb-2">
                Username
              </label>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Enter your username"
                className="w-full bg-zinc-900/50 border border-zinc-800 rounded-lg px-4 py-2.5 text-white placeholder:text-zinc-600 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-transparent transition-all"
              />
            </div>

            <div>
              <label className="block text-sm text-zinc-300 mb-2">
                Bot Name
              </label>
              <input
                type="text"
                value={botName}
                onChange={(e) => setBotName(e.target.value)}
                placeholder="Enter bot name"
                className="w-full bg-zinc-900/50 border border-zinc-800 rounded-lg px-4 py-2.5 text-white placeholder:text-zinc-600 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-transparent transition-all"
              />
            </div>
          </div>

          {/* Upload Documents */}
          <div>
            <label className="block text-sm text-zinc-300 mb-2">
              Upload Documents
            </label>
            <div
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              className={`relative border-2 border-dashed rounded-lg p-12 text-center transition-all ${
                dragActive
                  ? 'border-indigo-500 bg-indigo-500/5'
                  : 'border-zinc-800 hover:border-zinc-700'
              }`}
            >
              <input
                type="file"
                multiple
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
              <Upload className="w-10 h-10 text-zinc-600 mx-auto mb-4" />
              <p className="text-zinc-400 mb-1">Drop File Here</p>
              <p className="text-zinc-600 text-sm mb-4">or</p>
              <button className="text-indigo-400 hover:text-indigo-300 text-sm transition-colors">
                Click to Upload
              </button>
            </div>
          </div>

          {/* Website URL Section */}
          <div className="pt-6 border-t border-zinc-800">
            <div className="text-center mb-6">
              <span className="text-zinc-500 text-sm">OR</span>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-sm text-zinc-300 mb-2">
                  Website URL
                </label>
                <p className="text-xs text-zinc-500 mb-3">
                  Supports single pages or sitemap.xml for entire websites
                </p>
                <input
                  type="url"
                  value={websiteUrl}
                  onChange={(e) => setWebsiteUrl(e.target.value)}
                  placeholder="Enter website URL (e.g., https://example.com)"
                  className="w-full bg-zinc-900/50 border border-zinc-800 rounded-lg px-4 py-2.5 text-white placeholder:text-zinc-600 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-transparent transition-all"
                />
              </div>

              <div>
                <label className="block text-sm text-zinc-300 mb-3">
                  URL Type
                </label>
                <p className="text-xs text-zinc-500 mb-3">
                  Single page, Process one webpage | Sitemap: Process entire website
                </p>
                <div className="flex gap-3">
                  <button
                    onClick={() => setUrlType('single')}
                    className={`flex-1 px-4 py-2.5 rounded-lg text-sm transition-all ${
                      urlType === 'single'
                        ? 'bg-indigo-600 text-white'
                        : 'bg-zinc-900/50 text-zinc-400 border border-zinc-800 hover:border-zinc-700'
                    }`}
                  >
                    Single Page
                  </button>
                  <button
                    onClick={() => setUrlType('sitemap')}
                    className={`flex-1 px-4 py-2.5 rounded-lg text-sm transition-all ${
                      urlType === 'sitemap'
                        ? 'bg-indigo-600 text-white'
                        : 'bg-zinc-900/50 text-zinc-400 border border-zinc-800 hover:border-zinc-700'
                    }`}
                  >
                    Entire Website (Sitemap)
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* URL Processing Tips */}
          <div className="bg-amber-500/5 border border-amber-500/20 rounded-lg p-4">
            <div className="flex gap-3">
              <Info className="w-5 h-5 text-amber-500 flex-shrink-0 mt-0.5" />
              <div>
                <h3 className="text-amber-500 text-sm mb-2">URL Processing Tips:</h3>
                <ul className="text-zinc-400 text-sm space-y-1 list-disc list-inside">
                  <li>For single pages, enter the complete URL</li>
                  <li>For sitemaps, ensure sitemap.xml is accessible</li>
                  <li>Large websites may take longer to process</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Create Button */}
          <button className="w-full bg-indigo-600 hover:bg-indigo-700 text-white py-3 rounded-lg transition-colors flex items-center justify-center gap-2">
            <FileText className="w-4 h-4" />
            Create Knowledge Base
          </button>
        </div>

        {/* Right Column - Upload Status */}
        <div>
          <label className="block text-sm text-zinc-300 mb-2">
            Upload Status
          </label>
          <div className="bg-zinc-900/30 border border-zinc-800 rounded-lg p-6 h-[400px] flex items-center justify-center">
            <p className="text-zinc-600 text-sm">Upload status will appear here...</p>
          </div>
        </div>
      </div>
    </div>
  );
}
