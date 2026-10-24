import React, { useState } from 'react';
import { Flower2, Sparkles, Menu } from 'lucide-react';

export default function App() {
  const [inputs, setInputs] = useState({
    sepal_length: '',
    sepal_width: '',
    petal_length: '',
    petal_width: ''
  });

  const [prediction, setPrediction] = useState(null);
  const [chatMessage, setChatMessage] = useState('');
  const [floraResponses, setFloraResponses] = useState([
    "Oi! 🌸 Posso explicar as espécies de Iris, ajudar você a interpretar uma previsão e responder perguntas sobre o modelo."
  ]);

  const handleInputChange = (e) => {
    setInputs({ ...inputs, [e.target.name]: e.target.value });
  };

  const handleClassify = async () => {
    try {
      const res = await fetch('http://127.0.0.1:8000/api/v1/classify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sepal_length: parseFloat(inputs.sepal_length) || 0,
          sepal_width: parseFloat(inputs.sepal_width) || 0,
          petal_length: parseFloat(inputs.petal_length) || 0,
          petal_width: parseFloat(inputs.petal_width) || 0
        })
      });
      const data = await res.json();
      setPrediction(data);
    } catch (err) {
      console.error("Erro ao classificar:", err);
    }
  };

  const handleChatSubmit = async (e) => {
    if (e.key === 'Enter' && chatMessage.trim()) {
      const userMsg = chatMessage;
      setChatMessage('');
      setFloraResponses((prev) => [...prev, `Você: ${userMsg}`]);

      try {
        const res = await fetch('http://127.0.0.1:8000/api/v1/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: userMsg })
        });
        const data = await res.json();
        setFloraResponses((prev) => [...prev, `Flora 🌸: ${data.reply}`]);
      } catch (err) {
        setFloraResponses((prev) => [...prev, "Flora 🌸: Desculpe, tive um problema ao se conectar ao servidor."]);
      }
    }
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

          <button className="btn-primary" onClick={handleClassify}>
            Classificar flor
          </button>

          {prediction && (
            <div className="result-box">
              <h3>Espécie identificada: Iris {prediction.predicted_species}</h3>
              <p>Confiança do modelo: {(prediction.confidence * 100).toFixed(1)}%</p>
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