import { useState, useEffect, useCallback, useRef } from "react";
import { healthCheck, ingest, sendQuestion, compare, generateGraphs, getResults, uploadFtResults } from "./api";

const EXEMPLES = [
  "What is 5G core network architecture?",
  "Explain network slicing in 5G",
  "What is the role of AMF in 5G?",
  "How does beamforming work in 5G?",
  "What are the key 3GPP Release 18 features?",
];

const MODELS = [
  { key: "mistral-small-latest", label: "Mistral Small" },
  { key: "mistral-large-latest", label: "Mistral Large" },
  { key: "gpt-4o-mini", label: "GPT-4o-mini" },
];

const APPROACHES = [
  { key: "LLM seul", label: "LLM seul" },
  { key: "RAG simple", label: "RAG simple" },
  { key: "RAG+Reranking", label: "RAG+Reranking" },
  { key: "Hybrid Search", label: "Hybrid Search" },
  { key: "Fine-tuning", label: "Fine-tuning" },
  { key: "Fine-tuning+RAG", label: "Fine-tuning+RAG" },
];

export default function App() {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("Connexion...");
  const [connected, setConnected] = useState(false);
  const [indexed, setIndexed] = useState(false);
  const [selectedModel, setSelectedModel] = useState(MODELS[0].key);
  const [selectedApproach, setSelectedApproach] = useState(APPROACHES[0].key);
  const [results, setResults] = useState(null);
  const [graphFiles, setGraphFiles] = useState([]);
  const [tab, setTab] = useState("chat");
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    healthCheck().then(async () => {
      setConnected(true);
      const r = await getResults();
      if (r?.resume) setResults(r);
      setStatus("Prêt – indexez la base TeleQnA");
    }).catch(() => {
      setConnected(false);
      setStatus("API non disponible");
    });
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const handleIngest = useCallback(async () => {
    setStatus("Indexation...");
    try {
      const data = await ingest();
      setIndexed(true);
      setStatus(`${data.indexed} chunks indexés (${data.documents} documents)`);
    } catch (err) {
      setStatus(err.message);
    }
  }, []);

  const handleNewChat = useCallback(() => setMessages([]), []);

  const handleCompare = useCallback(async () => {
    setStatus("Comparaison en cours...");
    try {
      await compare(10);
      const r = await getResults();
      if (r?.resume) setResults(r);
      setStatus(`Comparaison terminee: ${r?.resume?.length || 0} resultats`);
    } catch (err) {
      setStatus(err.message);
    }
  }, []);

  const handleGraphs = useCallback(async () => {
    setStatus("Generation des graphiques...");
    try {
      const data = await generateGraphs();
      setGraphFiles(data.files);
      const r = await getResults();
      if (r?.resume) setResults(r);
      setStatus(`Graphiques generes: ${data.files?.join(", ")}`);
    } catch (err) {
      setStatus(err.message);
    }
  }, []);

  const handleSend = useCallback(async (text) => {
    if (!text.trim()) return;
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setLoading(true);
    const modelLabel = MODELS.find(m => m.key === selectedModel)?.label;
    const approachLabel = APPROACHES.find(a => a.key === selectedApproach)?.label;
    setStatus(`${approachLabel} – ${modelLabel}...`);

    try {
      const data = await sendQuestion(text, 3, selectedModel, null, selectedApproach);
      setMessages((prev) => [
        ...prev,
        {
          role: "bot",
          content: data.answer,
          sources: data.sources || [],
          model: selectedModel,
          approach: selectedApproach,
        },
      ]);
      setStatus("Prêt");
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "bot", content: `Erreur : ${err.message}` },
      ]);
      setStatus("Erreur");
    } finally {
      setLoading(false);
    }
  }, [selectedModel, selectedApproach]);

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="sidebar-header">
          <h2>📡 Telecom RAG - Comparaison d'approches</h2>
          <p className="sidebar-subtitle">Normes 3GPP – TeleQnA</p>
        </div>

        <button className="btn-new-chat" onClick={handleNewChat}>
          + Nouvelle discussion
        </button>

        <button className="btn-ingest" onClick={handleIngest} disabled={loading}>
          {indexed ? "Reindexer le corpus" : "1. Indexer TeleQnA"}
        </button>

        <button className="btn-ingest" onClick={handleCompare} disabled={loading} style={{background:"#8b5cf6"}}>
          Comparer les approches
        </button>

        <button className="btn-ingest" onClick={handleGraphs} disabled={loading} style={{background:"#059669"}}>
          Generer les graphiques
        </button>

        <button className="btn-ingest" onClick={() => fileInputRef.current?.click()} disabled={loading} style={{background:"#d97706"}}>
          Importer resultats FT
        </button>
        <input ref={fileInputRef} type="file" accept=".json" style={{display:"none"}}
          onChange={async (e) => {
            const file = e.target.files?.[0];
            if (!file) return;
            setStatus("Importation des resultats FT...");
            try {
              await uploadFtResults(file);
              const r = await getResults();
              if (r?.resume) setResults(r);
              setStatus("Resultats FT importes. Cliquez sur Generer les graphiques.");
            } catch (err) {
              setStatus(err.message);
            }
            e.target.value = "";
          }}
        />

        <div className="selector-row">
          <div className="selector-group">
            <span className="selector-label">Approche</span>
            <select
              value={selectedApproach}
              onChange={(e) => setSelectedApproach(e.target.value)}
            >
              {APPROACHES.map((a) => (
                <option key={a.key} value={a.key}>{a.label}</option>
              ))}
            </select>
          </div>
          <div className="selector-group">
            <span className="selector-label">Modèle</span>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
            >
              {MODELS.map((m) => (
                <option key={m.key} value={m.key}>{m.label}</option>
              ))}
            </select>
          </div>
        </div>

        {messages.length > 0 && (
          <div className="sidebar-stats">
            <p className="selector-label">Statistiques session :</p>
            <p className="stat-line">Messages : {messages.length}</p>
          </div>
        )}

        <div className="sidebar-footer">
          <div className={`status-dot ${connected ? "online" : "offline"}`} />
          <span className="status-text">{status}</span>
        </div>
      </aside>

      <div className="main-area">
        <header className="chat-header">
          <nav className="tabs">
            <button className={`tab ${tab === "chat" ? "active" : ""}`} onClick={() => setTab("chat")}>
              Chat
            </button>
            <button className={`tab ${tab === "results" ? "active" : ""}`} onClick={() => setTab("results")} disabled={!results}>
              Resultats
            </button>
            <button className={`tab ${tab === "graphs" ? "active" : ""}`} onClick={() => setTab("graphs")} disabled={!results}>
              Graphiques
            </button>
          </nav>
          <div className="header-badge">
            {APPROACHES.find(a => a.key === selectedApproach)?.label}
            {" | "}
            {MODELS.find(m => m.key === selectedModel)?.label}
          </div>
        </header>

        {tab === "chat" && (
          <div className="chat-container">
            <div className="messages">
              {messages.length === 0 && !loading && (
                <div className="welcome">
                  <div className="welcome-icon">📡</div>
                  <h2>Bienvenue sur le Chatbot Telecom</h2>
                  <p>
                    Posez vos questions sur les normes 3GPP (5G, LTE, NB-IoT...).
                    Le systeme utilise le dataset TeleQnA pour repondre.
                  </p>
                  {!indexed && (
                    <p className="welcome-hint">
                      Cliquez sur <strong>"Indexer TeleQnA"</strong> pour commencer.
                    </p>
                  )}
                  <div className="examples">
                    <p className="examples-title">Questions suggerees :</p>
                    {EXEMPLES.map((q) => (
                      <button key={q} className="example-btn" onClick={() => handleSend(q)}>
                        {q}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {messages.map((msg, i) => (
                <div key={i} className={`message ${msg.role}`}>
                  <div className="message-avatar">{msg.role === "user" ? "👤" : "📡"}</div>
                  <div className="message-content">
                    <div className="message-model-badge">
                      {msg.approach && <span>Approche: {msg.approach}</span>}
                      {msg.approach && msg.model && <span>{" | "}</span>}
                      {msg.model && <span>Modele: {MODELS.find(m => m.key === msg.model)?.label || msg.model}</span>}
                    </div>
                    <div className="message-text">{msg.content}</div>
                    {msg.sources?.length > 0 && (
                      <div className="sources">
                        <p className="sources-title">Sources utilisees :</p>
                        {msg.sources.map((s, j) => (
                          <div key={j} className="source-item">
                            <span className="source-score">Score : {s.score.toFixed(2)}</span>
                            <span className="source-title">{s.titre} – {s.id_article}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {loading && (
                <div className="message bot">
                  <div className="message-avatar">📡</div>
                  <div className="message-content">
                    <div className="typing">Generation en cours...</div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            <div className="chat-input-area">
              <form
                className="chat-input-form"
                onSubmit={(e) => {
                  e.preventDefault();
                  const input = e.target.querySelector("input");
                  handleSend(input.value.trim());
                  input.value = "";
                }}
              >
                <input
                  type="text"
                  placeholder="Ask a telecom question..."
                  disabled={loading}
                  className="chat-input"
                />
                <button type="submit" className="btn-send" disabled={loading}>
                  Ask
                </button>
              </form>
              <p className="disclaimer">
                Responses based on TeleQnA dataset (3GPP standards).
              </p>
            </div>
          </div>
        )}

        {tab === "results" && results && (
          <div className="tab-page">
            <h2>Resultats de la comparaison</h2>
            {results.resume?.length > 0 && (
              <table className="result-table">
                <thead>
                  <tr>
                    <th>Modele / Approche</th>
                    <th>EM</th>
                    <th>F1</th>
                    <th>Temps</th>
                  </tr>
                </thead>
                <tbody>
                  {results.resume.map((r, i) => (
                    <tr key={i}>
                      <td>{r.modele} / {r.approche}</td>
                      <td>{(r.exact_match_rate * 100).toFixed(0)}%</td>
                      <td>{(r.f1_moyen * 100).toFixed(0)}%</td>
                      <td>{r.temps_moyen.toFixed(1)}s</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        )}

        {tab === "graphs" && results && (
          <div className="tab-page">
            <h2>Graphiques</h2>
            {graphFiles.length > 0 ? (
              <div className="graph-grid">
                {graphFiles.map((f) => (
                  <div key={f} className="graph-card">
                    <img src={`/graphs/${f}`} alt={f} />
                  </div>
                ))}
              </div>
            ) : (
              <p>Cliquez sur "Generer les graphiques" d'abord.</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
