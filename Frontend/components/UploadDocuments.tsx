import { useState } from 'react';
import { Upload, Link, FileText, Info, CheckCircle, AlertCircle, Loader2, BarChart3, Database } from 'lucide-react';
import { api, UploadResponse } from '../services/api';

export function UploadDocuments() {
  const [username, setUsername] = useState('');
  const [botName, setBotName] = useState('');
  const [websiteUrl, setWebsiteUrl] = useState('');
  const [urlType, setUrlType] = useState<'single' | 'sitemap'>('single');
  const [dragActive, setDragActive] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string>('');
  const [isUploading, setIsUploading] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);

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
    
    const files = Array.from(e.dataTransfer.files);
    const validFiles = files.filter(file => 
      ['.pdf', '.docx', '.txt', '.md'].some(ext => 
        file.name.toLowerCase().endsWith(ext)
      )
    );
    
    setSelectedFiles(prev => [...prev, ...validFiles]);
    
    if (validFiles.length !== files.length) {
      setUploadStatus('Some files were skipped (only PDF, DOCX, TXT, MD files are supported)');
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    const validFiles = files.filter(file => 
      ['.pdf', '.docx', '.txt', '.md'].some(ext => 
        file.name.toLowerCase().endsWith(ext)
      )
    );
    
    setSelectedFiles(prev => [...prev, ...validFiles]);
  };

  const removeFile = (index: number) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleUpload = async () => {
    if (!username.trim() || !botName.trim()) {
      setUploadStatus('Please enter both username and bot name');
      return;
    }

    if (selectedFiles.length === 0 && !websiteUrl.trim()) {
      setUploadStatus('Please upload documents OR enter a website URL');
      return;
    }

    if (selectedFiles.length > 0 && websiteUrl.trim()) {
      setUploadStatus('Please choose either file upload OR URL processing, not both');
      return;
    }

    setIsUploading(true);
    setUploadStatus('Processing...');

    try {
      let response: UploadResponse;

      if (selectedFiles.length > 0) {
        // Upload files
        response = await api.uploadDocuments(selectedFiles, username, botName);
        setUploadStatus(`Successfully uploaded ${selectedFiles.length} file(s)\n${response.chunks_stored} chunks stored\nCollection: ${response.qdrant_collection}`);
      } else {
        // Upload from URL
        const url = urlType === 'sitemap' && !websiteUrl.endsWith('sitemap.xml') 
          ? websiteUrl.replace(/\/$/, '') + '/sitemap.xml'
          : websiteUrl;
        
        response = await api.uploadFromUrl(url, urlType, username, botName);
        setUploadStatus(`Successfully processed URL: ${url}\n${response.chunks_stored} chunks stored\nCollection: ${response.qdrant_collection}`);
      }

      // Clear form after successful upload
      setSelectedFiles([]);
      setWebsiteUrl('');
      
    } catch (error) {
      setUploadStatus(`Upload failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div>
        <h2 style={{ color: '#ffffff', marginBottom: '24px', fontSize: '24px', fontWeight: 'bold' }}>
          Create a New Knowledge Base
        </h2>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
        {/* Left Column */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          {/* User Information */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div>
              <label style={{ display: 'block', fontSize: '14px', color: '#d4d4d8', marginBottom: '8px' }}>
                Username
              </label>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Enter your username"
                style={{
                  width: '100%',
                  backgroundColor: 'rgba(24, 24, 27, 0.5)',
                  border: '1px solid #27272a',
                  borderRadius: '8px',
                  padding: '10px 16px',
                  color: '#ffffff',
                  fontSize: '14px'
                }}
              />
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '14px', color: '#d4d4d8', marginBottom: '8px' }}>
                Bot Name
              </label>
              <input
                type="text"
                value={botName}
                onChange={(e) => setBotName(e.target.value)}
                placeholder="Enter bot name"
                style={{
                  width: '100%',
                  backgroundColor: 'rgba(24, 24, 27, 0.5)',
                  border: '1px solid #27272a',
                  borderRadius: '8px',
                  padding: '10px 16px',
                  color: '#ffffff',
                  fontSize: '14px'
                }}
              />
            </div>
          </div>

          {/* Upload Documents */}
          <div>
            <label style={{ display: 'block', fontSize: '14px', color: '#d4d4d8', marginBottom: '8px' }}>
              Upload Documents
            </label>
            <div
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              style={{
                position: 'relative',
                border: `2px dashed ${dragActive ? '#4f46e5' : '#27272a'}`,
                borderRadius: '8px',
                padding: '48px',
                textAlign: 'center',
                backgroundColor: dragActive ? 'rgba(79, 70, 229, 0.05)' : 'transparent',
                transition: 'all 0.2s',
                cursor: 'pointer'
              }}
              onMouseEnter={(e) => {
                if (!dragActive) {
                  e.currentTarget.style.borderColor = '#3f3f46';
                  e.currentTarget.style.backgroundColor = 'rgba(39, 39, 42, 0.1)';
                }
              }}
              onMouseLeave={(e) => {
                if (!dragActive) {
                  e.currentTarget.style.borderColor = '#27272a';
                  e.currentTarget.style.backgroundColor = 'transparent';
                }
              }}
            >
              <input
                type="file"
                multiple
                accept=".pdf,.docx,.txt,.md"
                onChange={handleFileInput}
                style={{
                  position: 'absolute',
                  inset: 0,
                  width: '100%',
                  height: '100%',
                  opacity: 0,
                  cursor: 'pointer'
                }}
              />
              <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '16px' }}>
                <svg 
                  style={{ width: '40px', height: '40px', color: '#71717a' }}
                  viewBox="0 0 24 24" 
                  fill="none" 
                  stroke="currentColor" 
                  strokeWidth="2" 
                  strokeLinecap="round" 
                  strokeLinejoin="round"
                >
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                  <polyline points="7,10 12,15 17,10"/>
                  <line x1="12" y1="15" x2="12" y2="3"/>
                </svg>
              </div>
              <p style={{ color: '#a1a1aa', marginBottom: '4px' }}>Drop File Here</p>
              <p style={{ color: '#71717a', fontSize: '14px', marginBottom: '16px' }}>or</p>
              <button style={{ 
                color: '#818cf8', 
                fontSize: '14px',
                background: 'none',
                border: 'none',
                cursor: 'pointer'
              }}>
                Click to Upload
              </button>
            </div>
            
            {/* Selected Files List */}
            {selectedFiles.length > 0 && (
              <div style={{ marginTop: '16px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <p style={{ fontSize: '14px', color: '#d4d4d8' }}>Selected Files:</p>
                {selectedFiles.map((file, index) => (
                  <div key={index} style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    backgroundColor: 'rgba(24, 24, 27, 0.3)',
                    border: '1px solid #27272a',
                    borderRadius: '8px',
                    padding: '12px'
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <FileText style={{ width: '16px', height: '16px', color: '#a1a1aa' }} />
                      <span style={{ fontSize: '14px', color: '#d4d4d8' }}>{file.name}</span>
                      <span style={{ fontSize: '12px', color: '#71717a' }}>
                        ({(file.size / 1024 / 1024).toFixed(2)} MB)
                      </span>
                    </div>
                    <button
                      onClick={() => removeFile(index)}
                      style={{
                        color: '#f87171',
                        fontSize: '14px',
                        background: 'none',
                        border: 'none',
                        cursor: 'pointer'
                      }}
                    >
                      Remove
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Website URL Section */}
          <div style={{ paddingTop: '24px', borderTop: '1px solid #27272a' }}>
            <div style={{ textAlign: 'center', marginBottom: '24px' }}>
              <span style={{ color: '#71717a', fontSize: '14px' }}>OR</span>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div>
                <label style={{ display: 'block', fontSize: '14px', color: '#d4d4d8', marginBottom: '8px' }}>
                  Website URL
                </label>
                <p style={{ fontSize: '12px', color: '#71717a', marginBottom: '12px' }}>
                  Supports single pages or sitemap.xml for entire websites
                </p>
                <input
                  type="url"
                  value={websiteUrl}
                  onChange={(e) => setWebsiteUrl(e.target.value)}
                  placeholder="Enter website URL (e.g., https://example.com)"
                  style={{
                    width: '100%',
                    backgroundColor: 'rgba(24, 24, 27, 0.5)',
                    border: '1px solid #27272a',
                    borderRadius: '8px',
                    padding: '10px 16px',
                    color: '#ffffff',
                    fontSize: '14px'
                  }}
                />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '14px', color: '#d4d4d8', marginBottom: '12px' }}>
                  URL Type
                </label>
                <p style={{ fontSize: '12px', color: '#71717a', marginBottom: '12px' }}>
                  Single page, Process one webpage | Sitemap: Process entire website
                </p>
                <div style={{ display: 'flex', gap: '12px' }}>
                  <button
                    onClick={() => setUrlType('single')}
                    style={{
                      flex: 1,
                      padding: '10px 16px',
                      borderRadius: '8px',
                      fontSize: '14px',
                      backgroundColor: urlType === 'single' ? '#4f46e5' : 'rgba(24, 24, 27, 0.5)',
                      color: urlType === 'single' ? '#ffffff' : '#a1a1aa',
                      border: urlType === 'single' ? 'none' : '1px solid #27272a',
                      cursor: 'pointer',
                      transition: 'all 0.2s'
                    }}
                  >
                    Single Page
                  </button>
                  <button
                    onClick={() => setUrlType('sitemap')}
                    style={{
                      flex: 1,
                      padding: '10px 16px',
                      borderRadius: '8px',
                      fontSize: '14px',
                      backgroundColor: urlType === 'sitemap' ? '#4f46e5' : 'rgba(24, 24, 27, 0.5)',
                      color: urlType === 'sitemap' ? '#ffffff' : '#a1a1aa',
                      border: urlType === 'sitemap' ? 'none' : '1px solid #27272a',
                      cursor: 'pointer',
                      transition: 'all 0.2s'
                    }}
                  >
                    Entire Website (Sitemap)
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* URL Processing Tips */}
          <div style={{
            backgroundColor: 'rgba(245, 158, 11, 0.05)',
            border: '1px solid rgba(245, 158, 11, 0.2)',
            borderRadius: '8px',
            padding: '16px'
          }}>
            <div style={{ display: 'flex', gap: '12px' }}>
              <Info style={{ width: '20px', height: '20px', color: '#f59e0b', flexShrink: 0, marginTop: '2px' }} />
              <div>
                <h3 style={{ color: '#f59e0b', fontSize: '14px', marginBottom: '8px' }}>URL Processing Tips:</h3>
                <ul style={{ color: '#a1a1aa', fontSize: '14px', margin: 0, paddingLeft: '20px' }}>
                  <li style={{ marginBottom: '4px' }}>For single pages, enter the complete URL</li>
                  <li style={{ marginBottom: '4px' }}>For sitemaps, ensure sitemap.xml is accessible</li>
                  <li>Large websites may take longer to process</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Create Button */}
          <button 
            onClick={handleUpload}
            disabled={isUploading}
            style={{
              width: '100%',
              padding: '12px 24px',
              borderRadius: '8px',
              backgroundColor: isUploading ? '#52525b' : '#4f46e5',
              color: isUploading ? '#a1a1aa' : '#ffffff',
              border: 'none',
              cursor: isUploading ? 'not-allowed' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px',
              fontSize: '14px',
              fontWeight: '500',
              transition: 'all 0.2s'
            }}
          >
            {isUploading ? (
              <>
                <div style={{
                  width: '16px',
                  height: '16px',
                  border: '2px solid #a1a1aa',
                  borderTop: '2px solid transparent',
                  borderRadius: '50%',
                  animation: 'spin 1s linear infinite'
                }} />
                Processing...
              </>
            ) : (
              <>
                <FileText style={{ width: '16px', height: '16px' }} />
            Create Knowledge Base
              </>
            )}
          </button>
        </div>

        {/* Right Column - Upload Status */}
        <div>
          <label style={{ display: 'block', fontSize: '14px', color: '#d4d4d8', marginBottom: '8px' }}>
            Upload Status
          </label>
          <div style={{
            backgroundColor: 'rgba(24, 24, 27, 0.3)',
            border: '1px solid #27272a',
            borderRadius: '8px',
            padding: '24px',
            height: '400px',
            overflowY: 'auto'
          }}>
            {uploadStatus ? (
              <div style={{ whiteSpace: 'pre-wrap', fontSize: '14px' }}>
                {uploadStatus.split('\n').map((line, index) => (
                  <div key={index} style={{ marginBottom: '4px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    {line.startsWith('Successfully') ? (
                      <>
                        <CheckCircle style={{ width: '16px', height: '16px', color: '#4ade80', flexShrink: 0 }} />
                        <span style={{ color: '#4ade80' }}>{line}</span>
                      </>
                    ) : line.startsWith('Upload failed') || line.startsWith('Please') ? (
                      <>
                        <AlertCircle style={{ width: '16px', height: '16px', color: '#f87171', flexShrink: 0 }} />
                        <span style={{ color: '#f87171' }}>{line}</span>
                      </>
                    ) : line.startsWith('Processing') ? (
                      <>
                        <Loader2 style={{ width: '16px', height: '16px', color: '#3b82f6', flexShrink: 0, animation: 'spin 1s linear infinite' }} />
                        <span style={{ color: '#3b82f6' }}>{line}</span>
                      </>
                    ) : line.includes('chunks stored') ? (
                      <>
                        <BarChart3 style={{ width: '16px', height: '16px', color: '#60a5fa', flexShrink: 0 }} />
                        <span style={{ color: '#60a5fa' }}>{line}</span>
                      </>
                    ) : line.includes('Collection:') ? (
                      <>
                        <Database style={{ width: '16px', height: '16px', color: '#a78bfa', flexShrink: 0 }} />
                        <span style={{ color: '#a78bfa' }}>{line}</span>
                      </>
                    ) : (
                      <span style={{ color: '#d4d4d8' }}>{line}</span>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <p style={{ color: '#71717a', fontSize: '14px' }}>Upload status will appear here...</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
