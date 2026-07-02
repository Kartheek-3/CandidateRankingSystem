import { Link } from 'react-router-dom';
import { Search, Zap, Shield, ChevronRight } from 'lucide-react';
import '../index.css';

const LandingPage = () => {
  return (
    <div className="landing-page">
      {/* Navigation */}
      <nav className="landing-nav">
        <div className="nav-logo">
          <h2>Nexus AI</h2>
        </div>
        <div className="nav-actions">
          <Link to="/login" className="nav-link">Log in</Link>
          <Link to="/signup" className="btn btn-primary nav-btn">Sign up</Link>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-content">
          <h1 className="hero-title">
            Intelligent Candidate <span className="text-gradient">Discovery</span>
          </h1>
          <p className="hero-subtitle">
            Find the perfect engineering talent in seconds. Our advanced AI pipeline uses semantic search, 
            vector embeddings, and behavioral scoring to rank thousands of profiles against your exact needs.
          </p>
          <div className="hero-actions">
            <Link to="/signup" className="btn btn-primary hero-btn">
              Start Discovering <ChevronRight size={18} />
            </Link>
            <Link to="/login" className="btn btn-secondary hero-btn-outline">
              View Demo
            </Link>
          </div>
        </div>
        <div className="hero-visual">
          <div className="mock-chat">
            <div className="mock-user">Senior ML Engineer, PyTorch, FAISS</div>
            <div className="mock-assistant">
              <div className="mock-result">
                <strong>#1 Match</strong>
                <p>Found candidate with 6 years experience in FAISS & Vector DBs.</p>
              </div>
              <div className="mock-result">
                <strong>#2 Match</strong>
                <p>Found candidate with 8 years experience building retrieval systems.</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="features-section">
        <div className="feature-card">
          <div className="feature-icon"><Search size={24} /></div>
          <h3>Semantic Understanding</h3>
          <p>We don't just match keywords. Our Sentence Transformers model understands the deep context of your job descriptions.</p>
        </div>
        <div className="feature-card">
          <div className="feature-icon"><Zap size={24} /></div>
          <h3>Lightning Fast</h3>
          <p>Powered by FAISS index clustering, we search through hundreds of thousands of candidate profiles in under a second.</p>
        </div>
        <div className="feature-card">
          <div className="feature-icon"><Shield size={24} /></div>
          <h3>Smart Scoring</h3>
          <p>We combine semantic similarity with hard-coded heuristics like consulting experience and recruiter response rates.</p>
        </div>
      </section>

      {/* Footer */}
      <footer className="landing-footer">
        <div className="footer-content">
          <div className="footer-brand">
            <h3>Nexus AI</h3>
            <p>Building the future of talent discovery.</p>
          </div>
          <div className="footer-links">
            <a href="#">Privacy</a>
            <a href="#">Terms</a>
            <a href="#">Contact</a>
          </div>
        </div>
        <div className="footer-bottom">
          <p>&copy; {new Date().getFullYear()} Nexus AI. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;
