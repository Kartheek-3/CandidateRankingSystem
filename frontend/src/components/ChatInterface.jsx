import { useState, useRef, useEffect, useCallback } from 'react';
import axios from 'axios';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://127.0.0.1:5001';
import {
  Search, Loader2, Bookmark, CheckCircle, Send, Bot, User, LogOut,
  Menu, MessageSquare, Plus, Mic, MicOff, Paperclip, X, FileText,
  Sun, Moon, Settings, AlertTriangle, Zap, Brain, Shield, Activity,
  ChevronDown, ChevronUp, ArrowUp, Image as ImageIcon, Clock, Edit, Check, Filter, FileCode, Store, RefreshCw, Download, Upload,
  MoreHorizontal, Share, Pin, Archive, Trash2
} from 'lucide-react';
import { supabase } from '../supabase';
import { auth } from '../firebase';
import { signOut, onAuthStateChanged } from 'firebase/auth';
import { useNavigate } from 'react-router-dom';
import '../index.css';

// ---------------------------------------------------------------------------
// Stage progress labels
// ---------------------------------------------------------------------------
const STAGE_LABELS = {
  idle:              { label: 'Ready',                 pct: 0   },
  starting:          { label: 'Starting pipeline…',   pct: 5   },
  stage1_decomposing:{ label: 'Stage 1 · Decomposing JD…',    pct: 15  },
  stage2_retrieving: { label: 'Stage 2 · Semantic Retrieval…', pct: 35  },
  loading_profiles:  { label: 'Loading candidate profiles…',  pct: 45  },
  stage3_evaluating: { label: 'Stage 3 · Evaluating candidates…', pct: 65 },
  stage4_critiquing: { label: 'Stage 4 · Critic reviewing…',  pct: 82  },
  stage5_synthesizing:{ label: 'Stage 5 · Synthesizing ranks…', pct: 94 },
  done:              { label: 'Complete',              pct: 100 },
  error:             { label: 'Error',                 pct: 0   },
};

// ---------------------------------------------------------------------------
// Sub-score bar component
// ---------------------------------------------------------------------------
const SubScoreBar = ({ label, value, color }) => {
  const pct = Math.round(Math.min(Math.max(value * 100, 0), 100));
  return (
    <div className="sub-score-row">
      <span className="sub-score-label">{label}</span>
      <div className="sub-score-track">
        <div
          className="sub-score-fill"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
      <span className="sub-score-pct">{pct}%</span>
    </div>
  );
};

// ---------------------------------------------------------------------------
// Candidate card component
// ---------------------------------------------------------------------------
const CandidateCard = ({ cand, isSaved, onSave }) => {
  const [expanded, setExpanded] = useState(false);
  const overallPct = Math.min(Math.max((cand.score / 1.0) * 100, 0), 100);
  const sub = cand.sub_scores || {};

  return (
    <div className="chat-candidate-card">
      <div className="card-top">
        <div className="rank-badge">#{cand.rank}</div>
        <div className="card-top-right">
          {cand.pipeline_mode === 'agentic' && (
            <span className="mode-badge agentic-badge">
              <Brain size={10} /> AI
            </span>
          )}
          <button
            className={`save-btn ${isSaved ? 'saved' : ''}`}
            onClick={() => onSave(cand)}
            disabled={isSaved}
            title="Save candidate"
          >
            {isSaved ? <CheckCircle size={16} /> : <Bookmark size={16} />}
          </button>
        </div>
      </div>

      <div className="cand-identity">
        <h3>{cand.candidate_id}</h3>
        <span>
          {cand.metadata?.profile?.current_title || cand.metadata?.current_title || 'Engineer'} &bull;{' '}
          {cand.metadata?.profile?.years_of_experience ?? cand.metadata?.years_of_experience ?? '?'} YOE
        </span>
      </div>

      <div className="score-main-row">
        <span className="score-main-label">Match</span>
        <div className="score-mini">
          <div className="score-fill" style={{ width: `${overallPct}%` }} />
        </div>
        <span className="score-main-pct">{cand.score?.toFixed ? (cand.score * 100).toFixed(1) : '--'}%</span>
      </div>

      {sub && Object.keys(sub).length > 0 && (
        <div className="sub-scores-section">
          <button
            className="sub-scores-toggle"
            onClick={() => setExpanded(p => !p)}
          >
            <span>Score breakdown</span>
            {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>

          {expanded && (
            <div className="sub-scores-grid">
              {sub.technical   !== undefined && <SubScoreBar label="Technical"   value={sub.technical}   color="linear-gradient(90deg,#a8c7fa,#669df6)" />}
              {sub.experience  !== undefined && <SubScoreBar label="Experience"  value={sub.experience}  color="linear-gradient(90deg,#81c995,#34a853)" />}
              {sub.domain      !== undefined && <SubScoreBar label="Domain Fit"  value={sub.domain}      color="linear-gradient(90deg,#fdd663,#f29900)" />}
              {sub.behavioral  !== undefined && <SubScoreBar label="Behavioral"  value={sub.behavioral}  color="linear-gradient(90deg,#c58af9,#9334e6)" />}
              {sub.semantic    !== undefined && <SubScoreBar label="Semantic"    value={sub.semantic}    color="linear-gradient(90deg,#f28b82,#e8453c)" />}
            </div>
          )}
        </div>
      )}

      {cand.risk_flag && (
        <div className="risk-flag-chip">
          <AlertTriangle size={12} />
          <span>{cand.risk_flag.slice(0, 100)}{cand.risk_flag.length > 100 ? '…' : ''}</span>
        </div>
      )}

      <p className="cand-reasoning">{cand.reasoning}</p>
    </div>
  );
};

// ---------------------------------------------------------------------------
// Pipeline progress bar component
// ---------------------------------------------------------------------------
const PipelineProgress = ({ stage }) => {
  const info = STAGE_LABELS[stage] || STAGE_LABELS.starting;
  if (stage === 'idle' || !stage) return null;

  return (
    <div className="pipeline-progress-bar">
      <div className="pipeline-progress-track">
        <div
          className="pipeline-progress-fill"
          style={{ width: `${info.pct}%` }}
        />
      </div>
      <span className="pipeline-progress-label">{info.label}</span>
    </div>
  );
};

// ---------------------------------------------------------------------------
// Main ChatInterface
// ---------------------------------------------------------------------------
const ChatInterface = () => {
  const [messages, setMessages]           = useState([]);
  const [input, setInput]                 = useState('');
  const [loading, setLoading]             = useState(false);
  const [pipelineStage, setPipelineStage] = useState('idle');
  const [isListening, setIsListening]     = useState(false);
  const [attachedFile, setAttachedFile]   = useState(null);
  const [isUploading, setIsUploading]     = useState(false);
  
  const recognitionRef = useRef(null);
  const textareaRef    = useRef(null);
  const pollRef        = useRef(null);
  const messagesEndRef = useRef(null);
  const navigate       = useNavigate();
  const candidatesFileInputRef = useRef(null);
  const [isUploadingCandidates, setIsUploadingCandidates] = useState(false);

  const [savedIds, setSavedIds]                 = useState(new Set());
  const [chatHistory, setChatHistory]           = useState([]);
  const [currentChatId, setCurrentChatId]       = useState(null);
  const [isSidebarOpen, setIsSidebarOpen]       = useState(true);
  const [showProfileModal, setShowProfileModal] = useState(false);
  const [showSearchModal, setShowSearchModal] = useState(false);
  const [showGalleryModal, setShowGalleryModal] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [showSavedView, setShowSavedView]       = useState(false);
  const [showAnalyzerView, setShowAnalyzerView] = useState(false);
  const [theme, setTheme]                       = useState(() => localStorage.getItem('theme') || 'light');
  const [savedCandidatesData, setSavedCandidatesData] = useState([]);
  const [galleryFiles, setGalleryFiles] = useState([]);
  const [galleryLoading, setGalleryLoading] = useState(false);
  const [compareList, setCompareList] = useState([]);
  const [showCompareModal, setShowCompareModal] = useState(false);
  const [activeBatch, setActiveBatch] = useState(null);
  
  // New States for Pin/Archive
  const [pinnedChats, setPinnedChats] = useState([]);
  const [archivedChats, setArchivedChats] = useState([]);
  const [chatMenuOpen, setChatMenuOpen] = useState(null);
  
  // Analyzer states
  const [analyzerInput, setAnalyzerInput] = useState('');
  const [analyzerFile, setAnalyzerFile] = useState(null);
  const [parsedResume, setParsedResume] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [groqApiKey, setGroqApiKey] = useState(() => localStorage.getItem('groqApiKey') || '');

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 250)}px`;
    }
  }, [input]);

  useEffect(() => {
    document.body.classList.toggle('light-theme', theme === 'light');
    localStorage.setItem('theme', theme);
  }, [theme]);

  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = 'en-US';
      recognition.onresult = (event) => {
        let finalTranscript = '';
        for (let i = event.resultIndex; i < event.results.length; ++i) {
          if (event.results[i].isFinal) finalTranscript += event.results[i][0].transcript;
        }
        if (finalTranscript) {
          setInput(prev => prev + (prev.length > 0 && !prev.endsWith(' ') ? ' ' : '') + finalTranscript);
        }
      };
      recognition.onerror = () => setIsListening(false);
      recognition.onend   = () => setIsListening(false);
      recognitionRef.current = recognition;
    }
  }, []);

  const toggleListening = () => {
    if (isListening) { recognitionRef.current?.stop(); setIsListening(false); }
    else {
      if (recognitionRef.current) { recognitionRef.current.start(); setIsListening(true); }
      else alert('Voice input is not supported in this browser.');
    }
  };

  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  useEffect(() => {
    const unsub = onAuthStateChanged(auth, user => { 
      if (user) {
        fetchChats(user.uid);
        try {
          const pinned = JSON.parse(localStorage.getItem(`pinnedChats_${user.uid}`) || '[]');
          const archived = JSON.parse(localStorage.getItem(`archivedChats_${user.uid}`) || '[]');
          setPinnedChats(pinned);
          setArchivedChats(archived);
        } catch (e) { console.warn("Failed to load local prefs", e); }
      }
    });
    return () => unsub();
  }, []);

  useEffect(() => {
    if (loading) {
      pollRef.current = setInterval(async () => {
        try {
          const res = await axios.get(`${BACKEND_URL}/api/status`);
          setPipelineStage(res.data.stage || 'idle');
        } catch {/* server may not be ready yet */}
      }, 1500);
    } else {
      clearInterval(pollRef.current);
      setPipelineStage('idle');
    }
    return () => clearInterval(pollRef.current);
  }, [loading]);

  const fetchChats = async (uid) => {
    const { data } = await supabase.from('chats').select('*').eq('user_id', uid).order('created_at', { ascending: false });
    if (data) setChatHistory(data);
  };

  const startNewChat = () => { setShowSavedView(false); setShowAnalyzerView(false); setCurrentChatId(null); setMessages([]); };

  const loadChat = async (chatId) => {
    setShowSavedView(false);
    setShowAnalyzerView(false);
    if (chatId === currentChatId) return;
    setCurrentChatId(chatId);
    setMessages([]);
    const { data } = await supabase.from('chat_messages').select('*').eq('chat_id', chatId).order('created_at', { ascending: true });
    if (data && data.length > 0) {
      setMessages(data.map(msg => ({ role: msg.role, content: msg.content, results: msg.results, timeTaken: msg.time_taken, pipelineMode: msg.pipeline_mode })));
    } else {
      setMessages([]);
    }
  };

  const handleUploadCandidates = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    setIsUploadingCandidates(true);
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const res = await axios.post(`${BACKEND_URL}/api/upload_candidates`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      // Save file to Supabase Storage so it is visible in the Gallery
      const uid = auth.currentUser?.uid;
      if (uid) {
        const uniqueName = `candidates_${Date.now()}_${file.name}`;
        const formDataAttach = new FormData();
        formDataAttach.append('file', file);
        formDataAttach.append('file_path', `${uid}/${uniqueName}`);
        await axios.post(`${BACKEND_URL}/api/upload_attachment`, formDataAttach).catch(e => console.error("Gallery upload error:", e));
      }
      
      if (res.data.batch_id) {
        setActiveBatch({ id: res.data.batch_id, name: file.name });
      }
      
      alert(res.data.message || 'Upload successful! File added to Gallery.');
    } catch (err) {
      console.error(err);
      alert('Upload failed: ' + (err.response?.data?.error || err.message));
    } finally {
      setIsUploadingCandidates(false);
      e.target.value = null;
    }
  };

  const loadSavedCandidates = async () => {
    setShowSavedView(true);
    setShowAnalyzerView(false);
    setCompareList([]); // Reset compare list on load
    const uid = auth.currentUser?.uid;
    if (!uid) return;
    setLoading(true);
    const { data } = await supabase.from('saved_candidates').select('*').eq('user_id', uid).order('created_at', { ascending: false });
    if (data) setSavedCandidatesData(data);
    setLoading(false);
  };

  const loadGalleryFiles = async () => {
    setShowGalleryModal(true);
    const uid = auth.currentUser?.uid;
    if (!uid) return;
    
    setGalleryLoading(true);
    try {
      const { data, error } = await supabase.storage.from('chat-attachments').list(uid, {
        limit: 100,
        sortBy: { column: 'created_at', order: 'desc' }
      });
      if (data && !error) {
        // Filter out the empty placeholder folder object if it exists
        const validFiles = data.filter(f => f.name !== '.emptyFolderPlaceholder');
        const filesWithUrls = validFiles.map(file => {
          const { data: pub } = supabase.storage.from('chat-attachments').getPublicUrl(`${uid}/${file.name}`);
          return { ...file, url: pub.publicUrl };
        });
        setGalleryFiles(filesWithUrls);
      }
    } catch (e) {
      console.error('Error loading gallery:', e);
    }
    setGalleryLoading(false);
  };

  const handleClearHistory = async () => {
    if (!window.confirm('Are you sure you want to delete all chat history? This cannot be undone.')) return;
    const uid = auth.currentUser?.uid;
    if (!uid) return;
    
    setLoading(true);
    try {
      await supabase.from('chats').delete().eq('user_id', uid);
      setChatHistory([]);
      startNewChat();
    } catch (e) {
      console.error('Failed to clear history:', e);
    }
    setLoading(false);
    setShowProfileModal(false);
  };

  const handleLogout = async () => { await signOut(auth); navigate('/login'); };

  const handlePinChat = (e, chatId) => {
    e.stopPropagation();
    const uid = auth.currentUser?.uid;
    setPinnedChats(prev => {
      const next = prev.includes(chatId) ? prev.filter(id => id !== chatId) : [...prev, chatId];
      if (uid) localStorage.setItem(`pinnedChats_${uid}`, JSON.stringify(next));
      return next;
    });
    setChatMenuOpen(null);
  };

  const handleArchiveChat = (e, chatId) => {
    e.stopPropagation();
    const uid = auth.currentUser?.uid;
    setArchivedChats(prev => {
      const next = prev.includes(chatId) ? prev.filter(id => id !== chatId) : [...prev, chatId];
      if (uid) localStorage.setItem(`archivedChats_${uid}`, JSON.stringify(next));
      return next;
    });
    // Remove from pinned if archiving
    setPinnedChats(prev => {
      const next = prev.filter(id => id !== chatId);
      if (uid) localStorage.setItem(`pinnedChats_${uid}`, JSON.stringify(next));
      return next;
    });
    if (chatId === currentChatId) startNewChat();
    setChatMenuOpen(null);
  };

  const handleDeleteChat = async (e, chatId) => {
    e.stopPropagation();
    if (!window.confirm("Delete this chat?")) return;
    
    setChatMenuOpen(null);
    try {
      await supabase.from('chats').delete().eq('id', chatId);
      setChatHistory(prev => prev.filter(c => c.id !== chatId));
      if (currentChatId === chatId) startNewChat();
    } catch (err) {
      console.error("Failed to delete chat", err);
    }
  };

  const handleFileChange = (e) => { if (e.target.files?.[0]) setAttachedFile(e.target.files[0]); };
  const removeAttachedFile = () => setAttachedFile(null);

  const handleToggleCompare = (cand) => {
    setCompareList(prev => {
      if (prev.find(c => c.id === cand.id)) return prev.filter(c => c.id !== cand.id);
      if (prev.length >= 2) return [prev[1], cand];
      return [...prev, cand];
    });
  };

  const handleSend = async (overrideText = null) => {
    if (loading) return; // Prevent concurrent submissions
    const textToSend = overrideText || input;
    if (!textToSend.trim() && !attachedFile) return;

    let finalJdText = textToSend.trim();
    let fileUrl = null;
    let fileName = null;

    setLoading(true);

    let currentBatchId = activeBatch ? activeBatch.id : null;

    if (attachedFile) {
      setIsUploading(true);
      fileName = attachedFile.name;
      const fileExt = fileName.split('.').pop().toLowerCase();
      // Preserve the original file name so it's readable in the gallery
      const filePath = `${auth.currentUser?.uid || 'guest'}/${Date.now()}_${fileName}`;
      
      const formDataAttach = new FormData();
      formDataAttach.append('file', attachedFile);
      formDataAttach.append('file_path', filePath);
      
      try {
        const attachRes = await axios.post(`${BACKEND_URL}/api/upload_attachment`, formDataAttach, {
          headers: { 'Content-Type': 'multipart/form-data' }
        });
        fileUrl = attachRes.data.url;
      } catch (err) {
        console.error("Gallery upload error:", err);
      }
      
      // Auto-upload dataset files to the candidates table
      if (['csv', 'json', 'jsonl'].includes(fileExt)) {
        const formData = new FormData();
        formData.append('file', attachedFile);
        try {
          const res = await axios.post(`${BACKEND_URL}/api/upload_candidates`, formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
          });
          if (res.data.batch_id) {
            currentBatchId = res.data.batch_id;
            setActiveBatch({ id: res.data.batch_id, name: fileName });
          }
        } catch (err) {
          console.error("Failed to upload attached dataset", err);
        }
      } else if (attachedFile.type === 'text/plain') {
        const text = await attachedFile.text();
        finalJdText = finalJdText + '\n\n' + text;
      }
      
      setIsUploading(false);
    }

    const jd = finalJdText || '(See attached file)';
    if (!overrideText) setInput('');
    setAttachedFile(null);
    setMessages(prev => [...prev, { role: 'user', content: jd, fileName, fileUrl }]);

    try {
      let activeChatId = currentChatId;
      const uid = auth.currentUser?.uid;

      if (!activeChatId && uid) {
        const title = jd.substring(0, 30) + (jd.length > 30 ? '...' : '');
        const { data: newChat } = await supabase.from('chats').insert([{ user_id: uid, title }]).select().single();
        if (newChat) { 
          activeChatId = newChat.id; 
          setCurrentChatId(newChat.id); 
          setChatHistory(prev => [newChat, ...prev]); 
        }
      }

      if (activeChatId) {
        await supabase.from('chat_messages').insert([{ chat_id: activeChatId, role: 'user', content: jd, file_url: fileUrl, file_name: fileName }]);
      }

      const payload = { jd };
      if (currentBatchId) {
        payload.batch_id = currentBatchId;
      }

      const response  = await axios.post(`${BACKEND_URL}/api/rank`, payload);
      const results   = response.data.results;
      const timeTaken = response.data.time_seconds;
      const pipelineMode = response.data.pipeline_mode;
      const stageTimings = response.data.stage_times;

      setMessages(prev => [...prev, { role: 'assistant', timeTaken, results, pipelineMode, stageTimings }]);

      if (activeChatId) {
        const { error: dbErr } = await supabase.from('chat_messages').insert([{ 
          chat_id: activeChatId, 
          role: 'assistant', 
          results: results, 
          time_taken: timeTaken 
        }]);
        if (dbErr) console.error("Supabase insert error (assistant):", dbErr);
      }
    } catch (error) {
      console.error('Error fetching ranking:', error);
      const errMsg = error.response?.data?.error || 'Error contacting the backend. Is the Flask server running?';
      setMessages(prev => [...prev, { role: 'assistant', isError: true, content: errMsg }]);
    }
    setLoading(false);
  };

  const handleRetry = async () => {
    if (loading || messages.length === 0) return;
    const lastMsg = messages[messages.length - 1];
    if (lastMsg.role !== 'user') return;

    setLoading(true);
    try {
      const response = await axios.post(`${BACKEND_URL}/api/rank`, { jd: lastMsg.content });
      const results = response.data.results;
      const timeTaken = response.data.time_seconds;
      const pipelineMode = response.data.pipeline_mode;
      const stageTimings = response.data.stage_times;

      setMessages(prev => [...prev, { role: 'assistant', timeTaken, results, pipelineMode, stageTimings }]);

      if (currentChatId) {
        const { error: dbErr } = await supabase.from('chat_messages').insert([{ 
          chat_id: currentChatId, 
          role: 'assistant', 
          results: results, 
          time_taken: timeTaken 
        }]);
        if (dbErr) console.error("Supabase insert error (retry):", dbErr);
      }
    } catch (error) {
      console.error('Error fetching ranking on retry:', error);
      const errMsg = error.response?.data?.error || 'Error contacting the backend. Is the Flask server running?';
      setMessages(prev => [...prev, { role: 'assistant', isError: true, content: errMsg }]);
    }
    setLoading(false);
  };

  const handleAnalyzeResume = async () => {
    let textToSend = analyzerInput;
    if (!textToSend.trim() && !analyzerFile) return;

    setIsAnalyzing(true);
    setParsedResume(null);

    try {
      const formData = new FormData();
      formData.append('api_key', groqApiKey);
      if (textToSend) formData.append('resume', textToSend);
      
      if (analyzerFile) {
        // Upload to Supabase so it shows up in Gallery
        const uid = auth.currentUser?.uid;
        if (uid) {
          const uniqueName = `resume_${Date.now()}_${analyzerFile.name}`;
          const formDataAttach = new FormData();
          formDataAttach.append('file', analyzerFile);
          formDataAttach.append('file_path', `${uid}/${uniqueName}`);
          await axios.post(`${BACKEND_URL}/api/upload_attachment`, formDataAttach).catch(e => console.error("Gallery upload error:", e));
        }

        formData.append('file', analyzerFile);
      }
      
      const response = await axios.post(`${BACKEND_URL}/api/analyze_resume`, formData);
      setParsedResume(response.data);
    } catch (err) {
      console.error('Error analyzing resume:', err);
      alert('Failed to analyze resume. Check backend logs.');
    }
    setIsAnalyzing(false);
  };

  const handleKeyDown = (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); } };

  const handleSave = async (candidate) => {
    if (savedIds.has(candidate.candidate_id)) return;
    setSavedIds(prev => new Set(prev).add(candidate.candidate_id));
    try {
      const { error } = await supabase.from('saved_candidates').insert([{
        user_id: auth.currentUser?.uid,
        candidate_id: candidate.candidate_id,
        score: candidate.score,
        reasoning: candidate.reasoning
      }]);
      if (error) throw error;
    } catch (e) {
      console.warn('Supabase save failed:', e.message);
    }
  };

  const handleExportCSV = (results) => {
    if (!results || results.length === 0) return;
    
    const headers = ['Rank', 'Candidate ID', 'Match Score', 'Current Title', 'Years of Experience', 'Response Rate', 'Reasoning'];
    const csvRows = [headers.join(',')];
    
    results.forEach(cand => {
      const row = [
        cand.rank,
        cand.candidate_id,
        cand.score.toFixed(4),
        `"${(cand.metadata?.profile?.current_title || cand.metadata?.current_title || 'Engineer').replace(/"/g, '""')}"`,
        cand.metadata?.profile?.years_of_experience ?? cand.metadata?.years_of_experience ?? 0,
        `${((cand.metadata?.recruiter_response_rate || 0) * 100).toFixed(0)}%`,
        `"${(cand.reasoning || '').replace(/"/g, '""')}"`
      ];
      csvRows.push(row.join(','));
    });
    
    const blob = new Blob([csvRows.join('\n')], { type: 'text/csv;charset=utf-8;' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.setAttribute('href', url);
    a.setAttribute('download', `nexus_candidates_${Date.now()}.csv`);
    a.click();
    window.URL.revokeObjectURL(url);
  };

  return (
    <div className="chat-layout">
      {/* Sidebar */}
      <aside className={`sidebar ${isSidebarOpen ? '' : 'collapsed'}`}>
        <div className="sidebar-header">
          <button className="menu-toggle-btn" onClick={() => setIsSidebarOpen(!isSidebarOpen)}>
            <Menu size={20} />
          </button>
        </div>
        
        <div className="sidebar-content">
          <div className="sidebar-section" style={{ marginTop: '0.5rem' }}>
            <button className={`new-chat-btn ${!showSavedView && currentChatId === null ? 'active' : ''}`} onClick={startNewChat}>
              <Edit size={18} /> {isSidebarOpen && <span>New chat</span>}
            </button>
            
            <div className="sidebar-menu-group" style={{ marginTop: '0.5rem' }}>
              <button className="sidebar-btn sidebar-menu-item" onClick={() => setShowSearchModal(true)}>
                <Search size={18} /> {isSidebarOpen && <span>Search chats</span>}
              </button>
              <button className="sidebar-btn sidebar-menu-item" onClick={loadGalleryFiles}>
                <ImageIcon size={18} /> {isSidebarOpen && <span>Gallery</span>}
              </button>
              <button className={`sidebar-btn sidebar-menu-item ${showAnalyzerView ? 'active' : ''}`} onClick={() => { setShowSavedView(false); setShowAnalyzerView(true); setCurrentChatId(null); }}>
                <FileCode size={18} /> {isSidebarOpen && <span>Resume Analyzer</span>}
              </button>
            </div>
            
            <button className={`sidebar-btn gemini-pill-btn ${showSavedView ? 'active' : ''}`} style={{ marginTop: '1rem' }} onClick={loadSavedCandidates}>
              <Bookmark size={18} /> {isSidebarOpen && <span>Saved Candidates</span>}
            </button>
            <button className={`sidebar-btn gemini-pill-btn`} style={{ marginTop: '0.5rem', background: 'rgba(16, 163, 127, 0.1)', color: '#10a37f' }} onClick={() => candidatesFileInputRef.current?.click()}>
              {isUploadingCandidates ? <Loader2 size={18} className="animate-spin" /> : <Upload size={18} />}
              {isSidebarOpen && <span>{isUploadingCandidates ? 'Uploading...' : 'Upload Candidates'}</span>}
            </button>
            <input type="file" ref={candidatesFileInputRef} onChange={handleUploadCandidates} accept=".json,.jsonl,.csv" style={{ display: 'none' }} />
          </div>

          <div className="sidebar-section history-section" style={{ marginTop: '1rem', flex: 1, overflowY: 'auto' }}>
            {isSidebarOpen && <h3 style={{ color: 'var(--text-main)', fontSize: '1rem', margin: '0 0 0.5rem 0.5rem', fontWeight: 'bold' }}>Chats</h3>}
            <div className="history-list">
              {chatHistory.length === 0 ? (
                isSidebarOpen && <div className="empty-history">No past chats</div>
              ) : (
                (() => {
                  const pinned = chatHistory.filter(c => pinnedChats.includes(c.id));
                  const archived = chatHistory.filter(c => archivedChats.includes(c.id) && !pinnedChats.includes(c.id));
                  const recent = chatHistory.filter(c => !pinnedChats.includes(c.id) && !archivedChats.includes(c.id));
                  
                  const renderChatItem = (chat) => (
                    <div key={chat.id} style={{ position: 'relative' }} onMouseLeave={() => setChatMenuOpen(null)}>
                      <button 
                        style={{ 
                          display: 'flex', justifyContent: 'space-between', alignItems: 'center', 
                          padding: '0.6rem 0.5rem', color: 'var(--text-main)', fontSize: '0.9rem', 
                          background: chat.id === currentChatId ? 'rgba(128,128,128,0.1)' : 'transparent', 
                          border: 'none', cursor: 'pointer', textAlign: 'left', borderRadius: '8px',
                          transition: 'background 0.2s', width: '100%'
                        }} 
                        onMouseEnter={e => e.currentTarget.style.background = 'rgba(128,128,128,0.15)'}
                        onMouseLeave={e => e.currentTarget.style.background = chat.id === currentChatId ? 'rgba(128,128,128,0.1)' : 'transparent'}
                        onClick={() => loadChat(chat.id)} 
                        title={chat.title}
                      >
                        <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', paddingRight: '1rem' }}>{chat.title}</span>
                        {isSidebarOpen && (
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                            {chat.id === currentChatId && <div style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: '#0052cc', flexShrink: 0 }}></div>}
                            <div 
                              className="chat-menu-trigger" 
                              onClick={(e) => { e.stopPropagation(); setChatMenuOpen(chatMenuOpen === chat.id ? null : chat.id); }}
                              style={{ padding: '2px', borderRadius: '4px', display: 'flex', alignItems: 'center', color: 'var(--text-muted)' }}
                            >
                              <MoreHorizontal size={16} />
                            </div>
                          </div>
                        )}
                      </button>
                      
                      {chatMenuOpen === chat.id && isSidebarOpen && (
                        <div className="chat-dropdown-menu fade-in">
                          <button onClick={(e) => handlePinChat(e, chat.id)}>
                            <Pin size={14} /> {pinnedChats.includes(chat.id) ? 'Unpin' : 'Pin'} Chat
                          </button>
                          <button onClick={(e) => handleArchiveChat(e, chat.id)}>
                            <Archive size={14} /> {archivedChats.includes(chat.id) ? 'Unarchive' : 'Archive'}
                          </button>
                          <div className="menu-divider"></div>
                          <button onClick={(e) => handleDeleteChat(e, chat.id)} className="delete-option">
                            <Trash2 size={14} /> Delete
                          </button>
                        </div>
                      )}
                    </div>
                  );
                  
                  return (
                    <>
                      {pinned.length > 0 && isSidebarOpen && <div className="sidebar-group-label">Pinned</div>}
                      {pinned.map(renderChatItem)}
                      
                      {recent.length > 0 && isSidebarOpen && (pinned.length > 0 || archived.length > 0) && <div className="sidebar-group-label" style={{ marginTop: '0.5rem' }}>Recent</div>}
                      {recent.map(renderChatItem)}
                      
                      {archived.length > 0 && isSidebarOpen && <div className="sidebar-group-label" style={{ marginTop: '0.5rem' }}>Archived</div>}
                      {archived.map(renderChatItem)}
                    </>
                  );
                })()
              )}
            </div>
          </div>
        </div>
        <div className="sidebar-footer" style={{ padding: '1rem', borderTop: '1px solid var(--border-color)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', background: 'var(--bg-sidebar)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{ width: '32px', height: '32px', borderRadius: '50%', backgroundColor: 'var(--accent)', color: 'white', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.85rem', fontWeight: 'bold' }}>
              {auth.currentUser?.email ? auth.currentUser.email.substring(0, 2).toUpperCase() : 'U'}
            </div>
            {isSidebarOpen && (
              <div style={{ display: 'flex', flexDirection: 'column' }}>
                <span style={{ fontSize: '0.9rem', color: 'var(--text-main)', fontWeight: 500, textTransform: 'capitalize' }}>
                  {auth.currentUser?.email ? auth.currentUser.email.split('@')[0].replace(/[^a-zA-Z]/g, ' ') : 'User'}
                </span>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'capitalize' }}>
                  {auth.currentUser?.email ? auth.currentUser.email.split('@')[1].split('.')[0] : 'App'}
                </span>
              </div>
            )}
          </div>
          <button className="settings-icon-btn" onClick={() => setShowProfileModal(true)} title="Settings" style={{ background: 'transparent', border: 'none', cursor: 'pointer', padding: '0.25rem' }}>
            <Store size={20} color="#888" />
          </button>
        </div>
      </aside>

      {/* Profile Modal */}
      {showProfileModal && (
        <div className="modal-overlay" onClick={() => setShowProfileModal(false)}>
          <div className="profile-modal" onClick={e => e.stopPropagation()}>
            <div className="profile-modal-header">
              <h3>Account Profile</h3>
              <button className="close-btn" onClick={() => setShowProfileModal(false)}>×</button>
            </div>
            <div className="profile-modal-content">
              <div className="profile-avatar-large"><User size={48} /></div>
              <h2 className="profile-email">{auth.currentUser?.email}</h2>
              <p className="profile-uid">User ID: {auth.currentUser?.uid}</p>
              <div className="profile-stats">
                <div className="stat-box">
                  <span className="stat-val">{chatHistory.length}</span>
                  <span className="stat-label">Total Chats</span>
                </div>
              </div>
              <div className="theme-toggle-section">
                <span>App Theme</span>
                <button className="theme-toggle-btn" onClick={() => setTheme(prev => prev === 'dark' ? 'light' : 'dark')}>
                  {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
                  {theme === 'dark' ? 'Light Mode' : 'Dark Mode'}
                </button>
              </div>
              <div className="theme-toggle-section" style={{ borderTop: 'none', paddingTop: 0 }}>
                <span>Chat History</span>
                <button className="theme-toggle-btn" style={{ color: 'var(--error)', borderColor: 'rgba(239, 68, 68, 0.3)' }} onClick={handleClearHistory} disabled={loading}>
                  Clear All Chats
                </button>
              </div>
              <div className="theme-toggle-section" style={{ borderTop: 'none', paddingTop: 0, flexDirection: 'column', alignItems: 'flex-start', gap: '0.5rem' }}>
                <span style={{ fontSize: '0.9rem', fontWeight: 500 }}>Groq API Key (Resume Analyzer)</span>
                <input 
                  type="password"
                  className="search-input"
                  placeholder="Enter Groq API Key..."
                  value={groqApiKey}
                  onChange={(e) => {
                    setGroqApiKey(e.target.value);
                    localStorage.setItem('groqApiKey', e.target.value);
                  }}
                  style={{ width: '100%' }}
                />
              </div>
              <button onClick={handleLogout} className="btn btn-primary logout-large-btn">
                <LogOut size={18} /> Sign Out
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Search Modal */}
      {showSearchModal && (
        <div className="modal-overlay" onClick={() => setShowSearchModal(false)}>
          <div className="profile-modal search-modal" onClick={e => e.stopPropagation()}>
            <div className="profile-modal-header">
              <h3>Search Chats</h3>
              <button className="close-btn" onClick={() => setShowSearchModal(false)}>×</button>
            </div>
            <div className="profile-modal-content">
              <input 
                type="text" 
                className="search-input" 
                placeholder="Search past conversations..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                autoFocus
              />
              <div className="search-results" style={{ marginTop: '1rem', maxHeight: '300px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                {searchQuery.trim() === '' ? (
                  <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Type to search your previous conversations...</p>
                ) : chatHistory.filter(chat => chat.title.toLowerCase().includes(searchQuery.toLowerCase())).length === 0 ? (
                  <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>No results found for "{searchQuery}".</p>
                ) : (
                  chatHistory.filter(chat => chat.title.toLowerCase().includes(searchQuery.toLowerCase())).map(chat => (
                    <button 
                      key={chat.id} 
                      className="history-btn recent-chat-link" 
                      style={{ width: '100%', textAlign: 'left', justifyContent: 'flex-start' }}
                      onClick={() => {
                        loadChat(chat.id);
                        setShowSearchModal(false);
                        setSearchQuery('');
                      }}
                    >
                      <MessageSquare size={16} className="history-icon" />
                      <span className="history-title">{chat.title}</span>
                    </button>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Compare Modal */}
      {showCompareModal && compareList.length === 2 && (
        <div className="modal-overlay" onClick={() => setShowCompareModal(false)}>
          <div className="profile-modal compare-modal fade-in" style={{ maxWidth: '800px', display: 'flex', flexDirection: 'column' }} onClick={e => e.stopPropagation()}>
            <div className="profile-modal-header">
              <h3>Compare Candidates</h3>
              <button className="close-btn" onClick={() => setShowCompareModal(false)}>×</button>
            </div>
            
            <div style={{ margin: '1rem', padding: '1rem', background: 'rgba(16, 163, 127, 0.1)', border: '1px solid rgba(16, 163, 127, 0.3)', borderRadius: '8px', display: 'flex', alignItems: 'center', gap: '0.75rem', color: 'var(--text-main)' }}>
              <div style={{ background: '#10a37f', color: 'white', padding: '0.25rem', borderRadius: '50%', display: 'flex' }}><Check size={16} /></div>
              <span><strong>{compareList[0].score >= compareList[1].score ? compareList[0].candidate_id : compareList[1].candidate_id}</strong> is the recommended choice based on the AI Match Score.</span>
            </div>

            <div className="profile-modal-content" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', padding: '0 1rem 1.5rem 1rem' }}>
              {compareList.map(cand => {
                const isBest = cand.score === Math.max(compareList[0].score, compareList[1].score);
                return (
                  <div key={cand.id} style={{ position: 'relative', padding: '1.5rem', background: 'var(--bg-chat)', borderRadius: '12px', border: `2px solid ${isBest ? '#10a37f' : 'var(--border-color)'}`, display: 'flex', flexDirection: 'column', gap: '1rem', transition: 'all 0.2s' }}>
                    {isBest && <div style={{ position: 'absolute', top: '-12px', left: '1rem', background: '#10a37f', color: 'white', fontSize: '0.75rem', fontWeight: 'bold', padding: '0.2rem 0.6rem', borderRadius: '12px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>⭐ Top Pick</div>}
                    <h2 style={{ marginTop: isBest ? '0.5rem' : '0' }}>{cand.candidate_id}</h2>
                    <div>
                      <div style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>Match Score</div>
                      <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: isBest ? '#10a37f' : 'var(--accent)' }}>{(cand.score * 100).toFixed(1)}%</div>
                    </div>
                    <div>
                      <div style={{ fontSize: '0.9rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>AI Reasoning</div>
                      <p style={{ fontSize: '0.95rem', lineHeight: '1.6' }}>{cand.reasoning}</p>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Gallery Modal */}
      {showGalleryModal && (
        <div className="modal-overlay" onClick={() => setShowGalleryModal(false)}>
          <div className="profile-modal gallery-modal" style={{ maxWidth: '600px' }} onClick={e => e.stopPropagation()}>
            <div className="profile-modal-header">
              <h3>Gallery</h3>
              <button className="close-btn" onClick={() => setShowGalleryModal(false)}>×</button>
            </div>
            <div className="profile-modal-content">
              <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginBottom: '1rem' }}>Your attached images and files.</p>
              <div className="gallery-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(120px, 1fr))', gap: '1rem', maxHeight: '60vh', overflowY: 'auto' }}>
                {galleryLoading ? (
                  <div style={{ gridColumn: '1 / -1', textAlign: 'center', padding: '2rem' }}>
                    <Loader2 className="spinner" size={32} style={{ margin: '0 auto', color: 'var(--accent)' }} />
                  </div>
                ) : galleryFiles.length > 0 ? (
                  galleryFiles.map((file, idx) => {
                    const isImage = file.metadata?.mimetype?.startsWith('image/');
                    return (
                      <a key={idx} href={file.url} target="_blank" rel="noreferrer" className="gallery-item" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '1rem', background: 'var(--bg-input)', borderRadius: '12px', textDecoration: 'none', color: 'var(--text-main)', border: '1px solid var(--border-color)' }}>
                        {isImage ? (
                          <img src={file.url} alt={file.name} style={{ width: '100%', height: '80px', objectFit: 'cover', borderRadius: '8px', marginBottom: '0.5rem' }} />
                        ) : (
                          <FileText size={48} style={{ color: 'var(--text-muted)', marginBottom: '0.5rem', height: '80px' }} />
                        )}
                        <span style={{ fontSize: '0.8rem', textAlign: 'center', wordBreak: 'break-all', display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>{file.name}</span>
                      </a>
                    );
                  })
                ) : (
                  <div className="gallery-placeholder" style={{ gridColumn: '1 / -1' }}>
                    <ImageIcon size={32} opacity={0.2} />
                    <span>No files uploaded yet</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Main Chat Area */}
      <main className="chat-main">
        <header className="desktop-chat-header">
          <div className="header-left">
            {!isSidebarOpen && <button className="menu-btn" onClick={() => setIsSidebarOpen(true)} style={{ marginRight: '0.5rem', background: 'transparent', border: 'none', color: 'var(--text-main)', cursor: 'pointer' }}><Menu size={24} /></button>}
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
              <h2 style={{ fontSize: '1.25rem', fontWeight: '600', color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '0.25rem', margin: 0 }}>
                NEXUS AI <ChevronDown size={20} color="var(--text-muted)" />
              </h2>
            </div>
          </div>
          
          <div className="header-right">
            <button className="header-share-btn" onClick={() => {
              navigator.clipboard.writeText(window.location.href);
              alert("Chat link copied to clipboard!");
            }}>
              <Share size={18} /> Share
            </button>
            <div style={{ position: 'relative' }}>
              <button className="header-more-btn" onClick={() => setChatMenuOpen(chatMenuOpen === 'header' ? null : 'header')}>
                <MoreHorizontal size={24} />
              </button>
              
              {chatMenuOpen === 'header' && currentChatId && (
                <div className="chat-dropdown-menu fade-in" style={{ top: '100%', right: 0, marginTop: '0.5rem', width: '200px' }}>
                  <button onClick={(e) => handlePinChat(e, currentChatId)}>
                    <Pin size={16} /> {pinnedChats.includes(currentChatId) ? 'Unpin chat' : 'Pin chat'}
                  </button>
                  <button onClick={(e) => handleArchiveChat(e, currentChatId)}>
                    <Archive size={16} /> {archivedChats.includes(currentChatId) ? 'Unarchive' : 'Archive'}
                  </button>
                  <div className="menu-divider"></div>
                  <button onClick={(e) => handleDeleteChat(e, currentChatId)} className="delete-option">
                    <Trash2 size={16} /> Delete
                  </button>
                </div>
              )}
            </div>
          </div>
        </header>

        {showAnalyzerView ? (
          <div className="analyzer-view" style={{ flex: 1, padding: '2rem', overflowY: 'auto' }}>
            <div style={{ maxWidth: '800px', margin: '0 auto' }}>
              <h2 style={{ fontSize: '1.75rem', fontWeight: '500', marginBottom: '0.5rem' }}>Resume Analyzer</h2>
              <p style={{ color: 'var(--text-muted)', marginBottom: '2rem' }}>Paste a resume below to instantly extract skills, experience, and receive an AI SWOT analysis.</p>

              <div className="input-box centered" style={{ marginBottom: '2rem' }}>
                <label className="paperclip-btn">
                  <Plus size={24} />
                  <input type="file" onChange={(e) => { if (e.target.files?.[0]) setAnalyzerFile(e.target.files[0]); }} accept=".txt,.pdf" style={{ display: 'none' }} />
                </label>
                {analyzerFile && (
                  <div className="file-preview-pill" style={{ position: 'absolute', top: '-2rem', left: '1rem' }}>
                    <FileText size={16} /> <span>{analyzerFile.name}</span>
                    <button onClick={() => setAnalyzerFile(null)}><X size={14} /></button>
                  </div>
                )}
                <textarea
                  value={analyzerInput}
                  onChange={e => setAnalyzerInput(e.target.value)}
                  placeholder="Paste resume text here..."
                  rows={1}
                  style={{ resize: 'none', overflowY: 'auto', maxHeight: '250px' }}
                />
                <button 
                  className={`send-btn ${analyzerInput.trim() || analyzerFile ? 'active' : ''}`} 
                  onClick={handleAnalyzeResume} 
                  disabled={isAnalyzing || (!analyzerInput.trim() && !analyzerFile)}
                >
                  {isAnalyzing ? <Loader2 className="spinner" size={22} /> : <Check size={22} />}
                </button>
              </div>

              {parsedResume && (
                <div className="parsed-resume-card" style={{ background: 'var(--bg-card)', padding: '2rem', borderRadius: '16px', border: '1px solid var(--border-color)', boxShadow: '0 4px 15px rgba(0,0,0,0.05)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
                    <h3 style={{ margin: 0, fontSize: '1.25rem' }}>Analysis Results</h3>
                    <div style={{ background: 'var(--accent)', color: 'white', padding: '0.25rem 0.75rem', borderRadius: '16px', fontSize: '0.85rem', fontWeight: 600 }}>
                      {parsedResume.experience_years} Years Experience
                    </div>
                  </div>
                  
                  <div style={{ marginBottom: '1.5rem' }}>
                    <h4 style={{ color: 'var(--text-muted)', fontSize: '0.85rem', textTransform: 'uppercase', marginBottom: '0.5rem', letterSpacing: '0.5px' }}>Summary</h4>
                    <p style={{ lineHeight: '1.5' }}>{parsedResume.summary}</p>
                  </div>
                  
                  <div style={{ marginBottom: '1.5rem' }}>
                    <h4 style={{ color: 'var(--text-muted)', fontSize: '0.85rem', textTransform: 'uppercase', marginBottom: '0.5rem', letterSpacing: '0.5px' }}>Skills Extracted</h4>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                      {parsedResume.skills?.map(s => (
                        <span key={s} style={{ background: 'rgba(0,0,0,0.05)', border: '1px solid var(--border-color)', padding: '0.25rem 0.75rem', borderRadius: '4px', fontSize: '0.9rem' }}>{s}</span>
                      ))}
                    </div>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                    <div style={{ background: 'rgba(34, 197, 94, 0.1)', border: '1px solid rgba(34, 197, 94, 0.2)', padding: '1rem', borderRadius: '8px' }}>
                      <h4 style={{ color: '#16a34a', marginTop: 0, marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}><CheckCircle size={16}/> Strengths</h4>
                      <ul style={{ margin: 0, paddingLeft: '1.25rem', color: 'var(--text-main)' }}>
                        {parsedResume.strengths?.map(s => <li key={s} style={{ marginBottom: '0.25rem' }}>{s}</li>)}
                      </ul>
                    </div>
                    <div style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.2)', padding: '1rem', borderRadius: '8px' }}>
                      <h4 style={{ color: '#dc2626', marginTop: 0, marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}><AlertTriangle size={16}/> Weaknesses</h4>
                      <ul style={{ margin: 0, paddingLeft: '1.25rem', color: 'var(--text-main)' }}>
                        {parsedResume.weaknesses?.map(w => <li key={w} style={{ marginBottom: '0.25rem' }}>{w}</li>)}
                      </ul>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        ) : showSavedView ? (
          <div className="saved-candidates-view">
            <div className="saved-header">
              <h2>Saved Candidates</h2>
              <p>{savedCandidatesData.length} candidates bookmarked</p>
            </div>
            {loading ? (
              <div className="loading-centered"><Loader2 className="spinner" size={32} /></div>
            ) : (
              <div className="saved-grid">
                {savedCandidatesData.map(cand => (
                  <div key={cand.id} className="chat-candidate-card" style={compareList.some(c => c.id === cand.id) ? { borderColor: 'var(--accent)' } : {}}>
                    <div className="card-top" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <div className="rank-badge saved-badge"><Bookmark size={14} style={{ marginRight: '6px' }} /> Saved</div>
                      <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.85rem', cursor: 'pointer', color: 'var(--text-main)' }}>
                        <input 
                          type="checkbox" 
                          checked={compareList.some(c => c.id === cand.id)} 
                          onChange={() => handleToggleCompare(cand)} 
                          style={{ accentColor: 'var(--accent)', width: '16px', height: '16px', cursor: 'pointer' }}
                        /> 
                        Compare
                      </label>
                    </div>
                    <div className="cand-identity"><h3>{cand.candidate_id}</h3></div>
                    <div className="score-main-row">
                      <span className="score-main-label">Match</span>
                      <div className="score-mini">
                        <div className="score-fill" style={{ width: `${Math.min(Math.max(cand.score * 100, 0), 100)}%` }} />
                      </div>
                      <span className="score-main-pct">{(cand.score * 100).toFixed(1)}%</span>
                    </div>
                    <p className="cand-reasoning">{cand.reasoning}</p>
                  </div>
                ))}
                {savedCandidatesData.length === 0 && <div className="empty-history">No saved candidates yet. Bookmark some from the chat!</div>}
              </div>
            )}
            
            {compareList.length > 0 && (
              <div className="fade-in" style={{ 
                position: 'fixed', bottom: '30px', left: '50%', transform: 'translateX(-50%)', 
                background: 'var(--bg-input)', color: 'var(--text-main)', padding: '1rem 1.5rem', borderRadius: '16px', 
                border: '1px solid var(--border-color)', boxShadow: '0 10px 40px rgba(0,0,0,0.3)', 
                display: 'flex', alignItems: 'center', gap: '1.5rem', zIndex: 100 
              }}>
                <span style={{ fontWeight: '500' }}>{compareList.length} / 2 selected for comparison</span>
                <button 
                  onClick={() => setShowCompareModal(true)}
                  disabled={compareList.length !== 2}
                  style={{ 
                    padding: '0.6rem 1.2rem', background: 'var(--accent)', color: 'white', 
                    borderRadius: '8px', opacity: compareList.length === 2 ? 1 : 0.5, 
                    cursor: compareList.length === 2 ? 'pointer' : 'not-allowed', 
                    border: 'none', fontWeight: 'bold' 
                  }}
                >
                  Compare Now
                </button>
              </div>
            )}
          </div>
        ) : (
          <div className="chat-container">
            {messages.length === 0 ? (
              <div className="hero-layout">
                <img src="/logo.png" alt="Nexus AI Logo" style={{ width: '64px', height: '64px', marginBottom: '1rem', borderRadius: '8px', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }} />
                <h2>What kind of candidate are you looking for?</h2>
                <div className="hero-input-wrapper" style={{ maxWidth: '48rem' }}>
                  {attachedFile && (
                    <div className="file-preview-pill" style={{ marginBottom: '0.5rem', alignSelf: 'flex-start', marginLeft: '1.5rem' }}>
                      <FileText size={16} />
                      <span>{attachedFile.name}</span>
                      <button onClick={removeAttachedFile}><X size={14} /></button>
                    </div>
                  )}
                  {activeBatch && (
                    <div className="file-preview-pill" style={{ marginBottom: '0.5rem', alignSelf: 'flex-start', marginLeft: '1.5rem', background: 'rgba(59, 130, 246, 0.1)', color: '#3b82f6', border: '1px solid rgba(59, 130, 246, 0.2)' }}>
                      <FileText size={16} />
                      <span>Filtering by: {activeBatch.name}</span>
                      <button onClick={() => setActiveBatch(null)}><X size={14} /></button>
                    </div>
                  )}
                  <div className="input-box centered">
                    <label className="paperclip-btn">
                      <Plus size={24} />
                      <input type="file" onChange={handleFileChange} accept=".txt,.pdf,.doc,.docx" style={{ display: 'none' }} />
                    </label>
                    <textarea
                      ref={textareaRef}
                      value={input}
                      onChange={e => setInput(e.target.value)}
                      onKeyDown={handleKeyDown}
                      placeholder="Paste a Job Description..."
                      rows={1}
                      style={{ resize: 'none', overflowY: 'auto', maxHeight: '250px' }}
                    />
                    <button className={`mic-btn ${isListening ? 'listening' : ''}`} onClick={toggleListening} title={isListening ? 'Stop listening' : 'Start Voice Input'}>
                      {isListening ? <MicOff size={24} /> : <Mic size={24} />}
                    </button>
                    <button className={`send-btn ${input.trim() || attachedFile ? 'active' : ''}`} onClick={() => handleSend()} disabled={loading || (!input.trim() && !attachedFile)}>
                      {loading ? <Loader2 className="spinner" size={22} /> : <ArrowUp size={22} />}
                    </button>
                  </div>
                  <div className="suggestion-pills">
                    <button disabled={loading} onClick={() => handleSend("Senior Python Backend Engineer with Django and AWS experience")}>
                      <FileText size={16}/> Backend Engineer
                    </button>
                    <button disabled={loading} onClick={() => handleSend("Product Manager with FinTech and API integration experience")}>
                      <FileText size={16}/> Product Manager
                    </button>
                    <button disabled={loading} onClick={() => handleSend("Frontend Developer with React, Redux, and Tailwind")}>
                      <FileText size={16}/> Frontend Developer
                    </button>
                  </div>
                  
                  <div className="how-it-works-cards fade-in" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1.5rem', marginTop: '4rem', width: '100%', textAlign: 'left', animationDelay: '0.2s' }}>
                    <div className="hiw-card" style={{ background: 'var(--bg-card)', padding: '1.5rem', borderRadius: '12px', border: '1px solid var(--border-color)', boxShadow: '0 4px 15px rgba(0,0,0,0.02)' }}>
                      <div style={{ background: 'rgba(16, 163, 127, 0.1)', width: '40px', height: '40px', borderRadius: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '1rem', color: '#10a37f' }}><Upload size={20} /></div>
                      <h4 style={{ margin: '0 0 0.5rem 0', fontSize: '1.05rem' }}>1. Upload Data</h4>
                      <p style={{ margin: 0, fontSize: '0.9rem', color: 'var(--text-muted)', lineHeight: '1.45' }}>Use the sidebar to securely inject your own candidate pool directly into the cloud vector database.</p>
                    </div>
                    <div className="hiw-card" style={{ background: 'var(--bg-card)', padding: '1.5rem', borderRadius: '12px', border: '1px solid var(--border-color)', boxShadow: '0 4px 15px rgba(0,0,0,0.02)' }}>
                      <div style={{ background: 'rgba(59, 130, 246, 0.1)', width: '40px', height: '40px', borderRadius: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '1rem', color: '#3b82f6' }}><Search size={20} /></div>
                      <h4 style={{ margin: '0 0 0.5rem 0', fontSize: '1.05rem' }}>2. Paste a JD</h4>
                      <p style={{ margin: 0, fontSize: '0.9rem', color: 'var(--text-muted)', lineHeight: '1.45' }}>Paste any job description and let our multi-agent AI instantly rank the absolute best matches.</p>
                    </div>
                    <div className="hiw-card" style={{ background: 'var(--bg-card)', padding: '1.5rem', borderRadius: '12px', border: '1px solid var(--border-color)', boxShadow: '0 4px 15px rgba(0,0,0,0.02)' }}>
                      <div style={{ background: 'rgba(168, 85, 247, 0.1)', width: '40px', height: '40px', borderRadius: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '1rem', color: '#a855f7' }}><CheckCircle size={20} /></div>
                      <h4 style={{ margin: '0 0 0.5rem 0', fontSize: '1.05rem' }}>3. Compare & Save</h4>
                      <p style={{ margin: 0, fontSize: '0.9rem', color: 'var(--text-muted)', lineHeight: '1.45' }}>Bookmark your favorites and compare their skills side-by-side to make the final call.</p>
                    </div>
                  </div>
                  
                </div>
              </div>
            ) : (
              <div className="messages-container">
                {messages.map((msg, idx) => (
                  <div key={idx} className={`message-wrapper ${msg.role}`}>
                    <div className="message-content-block">
                      <div className="message-avatar">
                        {msg.role === 'assistant' ? <Bot size={24} /> : <User size={24} />}
                      </div>
                      <div className="message-body">
                        <div className="message-sender">{msg.role === 'assistant' ? 'Nexus AI' : 'You'}</div>

                        {msg.content && (
                          <div className={`message-text ${msg.isError ? 'error-text' : ''}`}>{msg.content}</div>
                        )}
                        {msg.fileName && (
                          <div className="message-attachment">
                            <FileText size={16} />
                            <a href={msg.fileUrl} target="_blank" rel="noreferrer">{msg.fileName}</a>
                          </div>
                        )}

                        {msg.results && (
                          <div className="results-block fade-in">
                            <div className="results-meta" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                <CheckCircle size={16} className="text-green" />
                                <span>Found top {msg.results.length} candidates in {msg.timeTaken}s</span>
                              </div>
                              <button 
                                onClick={() => handleExportCSV(msg.results)}
                                style={{
                                  display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.4rem 0.8rem',
                                  backgroundColor: 'var(--bg-input)', color: 'var(--accent)', border: '1px solid var(--border-color)', borderRadius: '8px',
                                  cursor: 'pointer', fontSize: '0.8rem', fontWeight: '600', transition: 'all 0.2s'
                                }}
                                onMouseOver={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-chat)'}
                                onMouseOut={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-input)'}
                              >
                                <Download size={14} /> Export CSV
                              </button>
                            </div>
                            {msg.pipelineMode && (
                              <span className={`mode-badge ${msg.pipelineMode === 'agentic' ? 'agentic-badge' : 'heuristic-badge'}`}>
                                {msg.pipelineMode === 'agentic' ? <><Brain size={11} /> Multi-Agent</> : <><Zap size={11} /> Heuristic</>}
                              </span>
                            )}
                            {msg.stageTimings && (
                              <div className="stage-timing-row">
                                {Object.entries(msg.stageTimings).map(([k, v]) => (
                                  <span key={k} className="stage-timing-chip">{k}: {v}s</span>
                                ))}
                              </div>
                            )}
                            <div className="candidate-carousel">
                              {msg.results.map(cand => (
                                <CandidateCard
                                  key={cand.candidate_id}
                                  cand={cand}
                                  isSaved={savedIds.has(cand.candidate_id)}
                                  onSave={handleSave}
                                />
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ))}

                {/* Loading state with pipeline progress */}
                {loading && (
                  <div className="message-wrapper assistant">
                    <div className="message-content-block">
                      <div className="message-avatar"><Bot size={24} /></div>
                      <div className="message-body">
                        <div className="message-sender">Nexus AI</div>
                        <PipelineProgress stage={pipelineStage} />
                        <div className="typing-indicator">
                          <div className="typing-dot" />
                          <div className="typing-dot" />
                          <div className="typing-dot" />
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </div>
            )}
            
            {/* Input Area (Bottom) - Only show when there are messages */}
            {messages.length > 0 && (
              <div className="input-area bottom">
                {attachedFile && (
                  <div className="file-preview-pill">
                    <FileText size={16} />
                    <span>{attachedFile.name}</span>
                    <button onClick={removeAttachedFile}><X size={14} /></button>
                  </div>
                )}
                {activeBatch && (
                  <div className="file-preview-pill" style={{ background: 'rgba(59, 130, 246, 0.1)', color: '#3b82f6', border: '1px solid rgba(59, 130, 246, 0.2)', marginLeft: '1rem', marginBottom: '0.5rem' }}>
                    <FileText size={16} />
                    <span>Filtering by: {activeBatch.name}</span>
                    <button onClick={() => setActiveBatch(null)}><X size={14} /></button>
                  </div>
                )}
                <div className="input-box bottom">
                  <label className="paperclip-btn">
                    <Plus size={24} />
                    <input type="file" onChange={handleFileChange} accept=".txt,.pdf,.doc,.docx" style={{ display: 'none' }} />
                  </label>
                  <textarea
                    ref={textareaRef}
                    value={input}
                    onChange={e => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Message Nexus AI..."
                    rows={1}
                    style={{ resize: 'none', overflowY: 'auto', maxHeight: '250px' }}
                  />
                  <button className={`mic-btn ${isListening ? 'listening' : ''}`} onClick={toggleListening} title={isListening ? 'Stop listening' : 'Start Voice Input'}>
                    {isListening ? <MicOff size={24} /> : <Mic size={24} />}
                  </button>
                  <button className={`send-btn ${input.trim() || attachedFile ? 'active' : ''}`} onClick={() => handleSend()} disabled={loading || (!input.trim() && !attachedFile)}>
                    {loading ? <Loader2 className="spinner" size={22} /> : <ArrowUp size={22} />}
                  </button>
                </div>
                <div className="input-footer">Nexus AI can make mistakes. Verify important candidates independently.</div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
};

export default ChatInterface;
