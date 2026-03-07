import React from 'react';
import './Dashboard.css';
import logoImage from '../assets/logo.png';

export default function Dashboard() {
    return (
        <div className="dashboard-layout">
            {/* Sidebar */}
            <aside className="sidebar">
                <div className="sidebar-header">
                    <div className="sidebar-logo">
                        <div className="sidebar-logo-icon">
                            <img src="http://localhost:3845/assets/d06d9cb206fb7e408a5f141d9f6f3111219199cc.svg" alt="Icon" />
                        </div>
                        <span className="sidebar-logo-text">Fusor AI</span>
                    </div>
                    <button className="sidebar-collapse">
                        <img src="http://localhost:3845/assets/ce53ab79d3121d75595285bb157d5a6fe9f8a68c.svg" alt="Collapse" />
                    </button>
                </div>

                <nav className="sidebar-nav">
                    <a href="#" className="nav-item active">
                        <img src="http://localhost:3845/assets/1a271db5e417a92742bdf0ec6a96060da153638c.svg" alt="Dashboard" />
                        <span>Dashboard</span>
                    </a>
                    <a href="#" className="nav-item">
                        <img src="http://localhost:3845/assets/83634fe46d6ceaef91804c027c2586320a2151d7.svg" alt="Chatbots" />
                        <span>Chatbots</span>
                    </a>
                    <a href="#" className="nav-item">
                        <img src="http://localhost:3845/assets/be082518d91c89758ab232f02f04876c83801e.svg" alt="Knowledge Base" />
                        <span>Knowledge Base</span>
                    </a>
                    <a href="#" className="nav-item">
                        <img src="http://localhost:3845/assets/0ae4fb18810a3398c156d536869067c546900fa2.svg" alt="Analytics" />
                        <span>Analytics</span>
                    </a>
                    <a href="#" className="nav-item">
                        <img src="http://localhost:3845/assets/12de7085ca750306edcc794c6f8c63242739b64c.svg" alt="Settings" />
                        <span>Settings</span>
                    </a>
                </nav>

                <div className="sidebar-footer">
                    <div className="user-profile">
                        <div className="user-avatar">JD</div>
                        <div className="user-info">
                            <span className="user-name">John Doe</span>
                            <span className="user-email">john@company.com</span>
                        </div>
                    </div>
                </div>
            </aside>

            {/* Main Content */}
            <main className="main-content">
                {/* Top Header */}
                <header className="top-header">
                    <div className="search-bar">
                        <img src="http://localhost:3845/assets/416ba753270f75d72460ddcf3a437a5bf3bc8cc0.svg" alt="Search" className="search-icon" />
                        <input type="text" placeholder="Search chatbots, documents, analytics..." />
                    </div>
                    <div className="header-actions">
                        <button className="btn-notification">
                            <img src="http://localhost:3845/assets/ea7f706f7fec1c9b9a1a5d1cc93cec733b9da1d3.svg" alt="Notifications" />
                            <span className="notification-dot"></span>
                        </button>
                        <div className="header-profile">
                            <div className="header-avatar">JD</div>
                            <span className="header-name">John Doe</span>
                        </div>
                    </div>
                </header>

                {/* Dashboard Area */}
                <div className="dashboard-area">
                    <div className="dashboard-header">
                        <div>
                            <h1 className="dashboard-title">Dashboard</h1>
                            <p className="dashboard-subtitle">Manage your chatbots and view analytics</p>
                        </div>
                        <button className="btn-create">
                            <img src="http://localhost:3845/assets/8c665db6ff4867201dd6d007ef88963acff5f097.svg" alt="Add" />
                            Create New Chatbot
                        </button>
                    </div>

                    <div className="stats-grid">
                        <div className="stat-card">
                            <div className="stat-card-header">
                                <span>Total Chatbots</span>
                                <img src="http://localhost:3845/assets/9efcbd55d4db9e3f51b5f2290dc387e732e3a21f.svg" alt="Icon" />
                            </div>
                            <div className="stat-card-value">3</div>
                            <div className="stat-card-trend">+1 from last month</div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-card-header">
                                <span>Conversations</span>
                                <img src="http://localhost:3845/assets/5c71b75c0a199e210c01d3fb1311b4d1ddf4fafe.svg" alt="Icon" />
                            </div>
                            <div className="stat-card-value">2,101</div>
                            <div className="stat-card-trend">+180 from last month</div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-card-header">
                                <span>Active Users</span>
                                <img src="http://localhost:3845/assets/40569a238c7ca45ad2aab24d525bcfc7a5db61b6.svg" alt="Icon" />
                            </div>
                            <div className="stat-card-value">1,847</div>
                            <div className="stat-card-trend">+12% from last month</div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-card-header">
                                <span>Success Rate</span>
                                <img src="http://localhost:3845/assets/fe79544297aa6918df243909e78bb8353a76c7a8.svg" alt="Icon" />
                            </div>
                            <div className="stat-card-value">94.2%</div>
                            <div className="stat-card-trend">+2.1% from last month</div>
                        </div>
                    </div>

                    <div className="chatbots-section">
                        <h2 className="section-title">Your Chatbots</h2>
                        <div className="chatbots-list">

                            <div className="chatbot-row">
                                <div className="chatbot-info">
                                    <div className="chatbot-icon">
                                        <img src="http://localhost:3845/assets/6c9311a7b5f6bb00f9324d0938d4aad0222b27d6.svg" alt="Bot" />
                                    </div>
                                    <div className="chatbot-details">
                                        <h3 className="chatbot-name">Customer Support Bot</h3>
                                        <div className="chatbot-meta">
                                            <span className="meta-item">
                                                <img src="http://localhost:3845/assets/4aabce6071b40bb71c74e56dabf3793d46f1a671.svg" alt="Msg" />
                                                1,234 conversations
                                            </span>
                                            <span className="meta-item">Updated 2 days ago</span>
                                        </div>
                                    </div>
                                </div>
                                <div className="chatbot-actions">
                                    <span className="badge-active">active</span>
                                    <span className="badge-type">Support</span>
                                    <button className="btn-icon">
                                        <img src="http://localhost:3845/assets/15aef1705a32946a04249c0c0a1f1f6b5e6ab924.svg" alt="Web" />
                                    </button>
                                    <button className="btn-icon">
                                        <img src="http://localhost:3845/assets/4f3c1f804d20912c03f91eb3d2892e2e376bd353.svg" alt="More" />
                                    </button>
                                </div>
                            </div>

                            <div className="chatbot-row">
                                <div className="chatbot-info">
                                    <div className="chatbot-icon">
                                        <img src="http://localhost:3845/assets/6c9311a7b5f6bb00f9324d0938d4aad0222b27d6.svg" alt="Bot" />
                                    </div>
                                    <div className="chatbot-details">
                                        <h3 className="chatbot-name">Sales Assistant</h3>
                                        <div className="chatbot-meta">
                                            <span className="meta-item">
                                                <img src="http://localhost:3845/assets/4aabce6071b40bb71c74e56dabf3793d46f1a671.svg" alt="Msg" />
                                                834 conversations
                                            </span>
                                            <span className="meta-item">Updated 1 week ago</span>
                                        </div>
                                    </div>
                                </div>
                                <div className="chatbot-actions">
                                    <span className="badge-active">active</span>
                                    <span className="badge-type">Sales</span>
                                    <button className="btn-icon">
                                        <img src="http://localhost:3845/assets/15aef1705a32946a04249c0c0a1f1f6b5e6ab924.svg" alt="Web" />
                                    </button>
                                    <button className="btn-icon">
                                        <img src="http://localhost:3845/assets/4f3c1f804d20912c03f91eb3d2892e2e376bd353.svg" alt="More" />
                                    </button>
                                </div>
                            </div>

                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
}
