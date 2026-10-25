import React, { useState } from 'react';
import { Flower2, Sparkles, Menu } from 'lucide-react';

// Define a URL base dinamicamente (Vercel/Render) com fallback
const API_BASE_URL = (import.meta.env.VITE_API_URL || 'https://flower-classifier-backend-mqdr.onrender.com').replace(/\/+$/, '');

// O plano free do Render "dorme"; a primeira chamada pode levar ~50s.
const fetchWithTimeout = (url, options = {}, ms = 70000) => {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ms);
  return fetch(url, { ...options, signal: controller.signal }).finally(() => clearTimeout(timer));
};

export default function App() {
  const [inputs, setInputs] = useState({
    sepal_length: '',
    sepal_width: '',
    petal_length: '',
    petal_width: ''
  });

  const [prediction, setPrediction] = useState(null);
  const [classifyError, setClassifyError] = useState('');
  const [loading, setLoading] = useState(false);
  const [floraThinking, setFloraThinking] = useState(false);
  const [chatMessage, setChatMessage] = useState('');
  const [floraResponses, setFloraResponses] = useState([
    "Oi! 🌸 Posso explicar as espécies de Iris, ajudar você a interpretar uma previsão e responder perguntas sobre o modelo."
  ]);

  const handleInputChange = (e) => {
    setInputs({ ...inputs, [e.target.name]: e.target.value });
  };

  const handleClassify = async () => {
    const values = {
      sepal_length: parseFloat(String(inputs.sepal_length).replace(',', '.')),
      sepal_width: parseFloat(String(inputs.sepal_width).replace(',', '.')),
      petal_length: parseFloat(String(inputs.petal_length).replace(',', '.')),
      petal_width: parseFloat(String(inputs.petal_width).replace(',', '.'))
    };

    if (Object.values(values).some((v) => isNaN(v) || v <= 0)) {
      setPrediction(null);
      setClassifyError('Preencha as quatro medidas com valores maiores que zero.');
      return;
    }

    setLoading(true);
    setClassifyError('');
    try {
      const res = await fetchWithTimeout(`${API_BASE_URL}/api/v1/classify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(values)
      });
      const data = await res.json();
      if (!res.ok || !(data.predicted_species || data.species)) {
        throw new Error(typeof data.detail === 'string' ? data.detail : `Erro ${res.status} da API`);
      }
      setPrediction(data);
    } catch (err) {
      console.error('Erro ao classificar:', err);
      setPrediction(null);
      setClassifyError(
        err.name === 'AbortError'
          ? 'O servidor demorou demais para responder. Tente de novo em instantes.'
          : `Não consegui classificar: ${err.message}`
      );
    } finally {
      setLoading(false);
    }
  };

  const handleChatSubmit = async (e) => {
    if (e.key === 'Enter' && chatMessage.trim() && !floraThinking) {
      const userMsg = chatMessage;
      setChatMessage('');
      setFloraResponses((prev) => [...prev, `Você: ${userMsg}`]);

      setFloraThinking(true);
      try {
        const res = await fetchWithTimeout(`${API_BASE_URL}/api/v1/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: userMsg })
        });
        const data = await res.json();
        
        // Trata todas as possíveis chaves do JSON para nunca exibir undefined
        const replyText = data.response || data.reply || data.message || data.data?.response || "Não consegui obter uma resposta.";
        
        setFloraResponses((prev) => [...prev, `Flora 🌸: ${replyText}`]);
      } catch (err) {
        setFloraResponses((prev) => [...prev, "Flora 🌸: Desculpe, tive um problema ao se conectar ao servidor."]);
      } finally {
        setFloraThinking(false);
      }
    }
  };

  // Helper para formatar a confiança com segurança sem multiplicar por 100 em duplicidade
  const getFormattedConfidence = () => {
    if (!prediction) return '0.0';
    const rawConf = prediction.confidence ?? prediction.probability ?? 0;
    let confNum = parseFloat(rawConf);
    if (isNaN(confNum)) return '0.0';
    if (confNum <= 1.0 && confNum > 0) {
      confNum = confNum * 100;
    }
    return confNum.toFixed(1);
  };

  return (
    <div className="container">
      <header>
        <div className="brand-title">
          <div className="brand-icon">
            <Flower2 size={24} />
          </div>
          <div className="brand-text">
            <h1>Flower Classifier</h1>
            <p>Descubra qual é a sua Iris</p>
          </div>
        </div>
        <Menu size={20} color="#2B2240" style={{ cursor: 'pointer' }} />
      </header>

      <main>
        <div className="card-section">
          <h2 className="section-title">
            Vamos classificar uma flor? 🌷
          </h2>
          <p className="section-subtitle">
            Informe as quatro medidas para obter uma previsão do modelo.
          </p>

          <div className="grid-inputs">
            <div className="input-group">
              <label>Comprimento da sépala</label>
              <input
                type="number"
                step="0.1"
                name="sepal_length"
                placeholder="0,0 cm"
                value={inputs.sepal_length}
                onChange={handleInputChange}
              />
            </div>
            <div className="input-group">
              <label>Largura da sépala</label>
              <input
                type="number"
                step="0.1"
                name="sepal_width"
                placeholder="0,0 cm"
                value={inputs.sepal_width}
                onChange={handleInputChange}
              />
            </div>
            <div className="input-group">
              <label>Comprimento da pétala</label>
              <input
                type="number"
                step="0.1"
                name="petal_length"
                placeholder="0,0 cm"
                value={inputs.petal_length}
                onChange={handleInputChange}
              />
            </div>
            <div className="input-group">
              <label>Largura da pétala</label>
              <input
                type="number"
                step="0.1"
                name="petal_width"
                placeholder="0,0 cm"
                value={inputs.petal_width}
                onChange={handleInputChange}
              />
            </div>
          </div>

          <button className="btn-primary" onClick={handleClassify} disabled={loading}>
            {loading ? 'Classificando… (o servidor pode demorar ~1 min na primeira vez)' : 'Classificar flor'}
          </button>

          {classifyError && (
            <p style={{ color: '#C0392B', marginTop: 12 }}>{classifyError}</p>
          )}

          {prediction && (
            <div className="result-box">
              <h3>
                Espécie identificada: Iris {prediction.predicted_species || prediction.species}
              </h3>
              <p>Confiança do modelo: {getFormattedConfidence()}%</p>
            </div>
          )}
        </div>

        <div className="flora-card">
          <div className="flora-header">
            <div className="flora-profile">
              <div className="flora-avatar">
                <Flower2 size={20} />
              </div>
              <div>
                <div className="flora-title">Flora · Assistente de IA</div>
                <div className="flora-subtitle">Seu guia de classificação</div>
              </div>
            </div>
            <Sparkles size={18} color="#9E77F2" />
          </div>

          <div className="flora-messages">
            {floraResponses.map((msg, index) => (
              <p key={index} className="flora-msg">{msg}</p>
            ))}
            {floraThinking && (
              <p className="flora-msg" style={{ opacity: 0.7, fontStyle: 'italic' }}>
                Flora 🌸 está pensando…
              </p>
            )}
          </div>

          <div className="flora-input-wrapper">
            <input
              type="text"
              placeholder="Pergunte algo sobre as flores..."
              value={chatMessage}
              onChange={(e) => setChatMessage(e.target.value)}
              onKeyDown={handleChatSubmit}
            />
          </div>
        </div>
      </main>

      <div className="footer-text">
        Prévia funcional conectada à API do Flower Classifier.
      </div>
    </div>
  );
}