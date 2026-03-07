import React, { useState } from 'react';
import './App.css';
import AuthModal from './components/AuthModal';
import Dashboard from './components/Dashboard';
import logoImage from './assets/logo.png';

const imgIcon = "http://localhost:3845/assets/9bfd8d5b194302753ebf462860c36f7913fd48a4.svg";
const imgIcon1 = "http://localhost:3845/assets/ad219a5dce32b80937c204324fe07e427f4775ba.svg";
const imgIcon2 = "http://localhost:3845/assets/44391ec517491a54d88907ef2ea4645e999151cf.svg";
const imgIcon3 = "http://localhost:3845/assets/6d41778a58b16b2f60d2ba41132d1df9921bdfb1.svg";
const imgIcon4 = "http://localhost:3845/assets/daf2a1a86e8ef232d0f73b34c2dbbea8ff04abb5.svg";

function App() {
  const [authModal, setAuthModal] = useState({ isOpen: false, type: 'login' });
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  const openModal = (type) => setAuthModal({ isOpen: true, type });
  const closeModal = () => setAuthModal((prev) => ({ ...prev, isOpen: false }));
  const switchModalType = () => setAuthModal((prev) => ({ ...prev, type: prev.type === 'login' ? 'signup' : 'login' }));
  const handleLogin = () => {
    setIsLoggedIn(true);
    setAuthModal({ isOpen: false, type: 'login' });
  };

  if (isLoggedIn) {
    return <Dashboard />;
  }

  return (
    <div className="page-container">
      {/* Header */}
      <header className="header">
        <div className="header-inner">
          <div className="logo-container">
            <img src={logoImage} alt="Fusor AI Logo" className="logo-image" />
          </div>
          <button className="btn-secondary" onClick={() => openModal('login')}>Login</button>
        </div>
      </header>

      {/* Hero Section */}
      <section className="hero">
        <div className="badge">
          <span className="badge-dot"></span>
          Build AI Chatbots Without Code
        </div>
        <h1 className="hero-title">Create Intelligent<br />AI Chatbots in Minutes</h1>
        <p className="hero-subtitle">
          The most powerful no-code platform for building, training, and deploying AI chatbots.
          Transform your business with conversational AI.
        </p>
        <div className="hero-actions">
          <button className="btn-primary" onClick={() => openModal('signup')}>Get Started Free</button>
          <button className="btn-secondary">View Demo</button>
        </div>
      </section>

      {/* Stats Section */}
      <section className="stats">
        <div className="stat-item">
          <div className="stat-value">10K+</div>
          <div className="stat-label">Active Chatbots</div>
        </div>
        <div className="stat-item">
          <div className="stat-value">1M+</div>
          <div className="stat-label">Conversations</div>
        </div>
        <div className="stat-item">
          <div className="stat-value">99.9%</div>
          <div className="stat-label">Uptime</div>
        </div>
      </section>

      {/* Features Section */}
      <section className="features-section">
        <div className="features-inner">
          <div className="features-header">
            <h2 className="features-title">Built for Modern Teams</h2>
            <p className="features-subtitle">
              Everything you need to create powerful AI experiences
            </p>
          </div>

          <div className="features-grid">
            <div className="feature-card">
              <div className="feature-icon-wrapper">
                <img src={imgIcon1} alt="Lightning Fast" className="feature-icon" />
              </div>
              <h3 className="feature-card-title">Lightning Fast</h3>
              <p className="feature-card-desc">
                Deploy chatbots in minutes with our intuitive drag-and-drop builder.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon-wrapper">
                <img src={imgIcon2} alt="Enterprise Security" className="feature-icon" />
              </div>
              <h3 className="feature-card-title">Enterprise Security</h3>
              <p className="feature-card-desc">
                Bank-grade encryption and compliance with SOC 2, GDPR, and HIPAA.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon-wrapper">
                <img src={imgIcon3} alt="Multi-Channel" className="feature-icon" />
              </div>
              <h3 className="feature-card-title">Multi-Channel</h3>
              <p className="feature-card-desc">
                Deploy across web, WhatsApp, Messenger, Slack, and more.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon-wrapper">
                <img src={imgIcon} alt="Smart AI" className="feature-icon" />
              </div>
              <h3 className="feature-card-title">Smart AI</h3>
              <p className="feature-card-desc">
                Powered by GPT-4 and custom models for intelligent responses.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="cta-section">
        <h2 className="cta-title">Ready to Get Started?</h2>
        <p className="cta-subtitle">
          Join thousands of teams building better customer experiences with AI
        </p>
        <button className="btn-primary" onClick={() => openModal('signup')}>Create Your First Chatbot</button>
      </section>

      {/* Footer */}
      <footer className="footer">
        <div className="footer-inner">
          <div className="footer-logo">
            <img src={logoImage} alt="Fusor AI Logo" className="footer-logo-image" />
          </div>
          <div className="footer-copyright">
            © 2025 Fusor AI. All rights reserved.
          </div>
        </div>
      </footer>

      <AuthModal
        isOpen={authModal.isOpen}
        type={authModal.type}
        onClose={closeModal}
        onSwitchType={switchModalType}
        onLogin={handleLogin}
      />
    </div>
  );
}

export default App;
