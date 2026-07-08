import { useRef, useState } from 'react'
import { Bot, Leaf, SendHorizontal } from 'lucide-react'
import { askChatbot } from '../services/api.js'

const QUICK_PROMPTS = [
  '¿Qué materiales son reciclables?',
  '¿Qué pasa con un tetrapack o jugo en caja?',
  '¿Cómo clasifica la IA un utensilio?',
]

function buildReply(question) {
  const normalized = question
    .normalize('NFKD')
    .replace(/[^\w\s]/g, '')
    .toLowerCase()

  if (/(hola|ayuda|qué puedes hacer|como funciona)/.test(normalized)) {
    return 'Puedo responder preguntas sobre reciclaje, clasificación de residuos y las categorías que usa este proyecto: reciclable y no reciclable.'
  }

  if (/(reciclable|reciclaje|reciclar|papel|carton|vidrio|metal|botella|lata|envase)/.test(normalized)) {
    return 'En general, los materiales como vidrio, latas, botellas limpias y papel o cartón secos suelen ser reciclables, siempre que estén limpios y separados correctamente.'
  }

  if (/(tetrapack|caja|jugo|leche|carton|envase carton)/.test(normalized)) {
    return 'Los envases tipo tetrapack o cajas de jugo suelen requerir atención especial. Muchas veces se reciclan, pero dependen del tipo de material y de la infraestructura local.'
  }

  if (/(styrofoam|espuma|poliestireno)/.test(normalized)) {
    return 'El styrofoam o espuma de poliestireno suele considerarse difícil de reciclar y, en muchos casos, se clasifica como no reciclable para este proyecto.'
  }

  if (/(utensilio|cuchara|tenedor|cubiertos|plástico de un solo uso)/.test(normalized)) {
    return 'Los utensilios y los plásticos de un solo uso suelen entrar en la categoría de no reciclable cuando están muy contaminados o cuando no son aceptados por la red de reciclaje local.'
  }

  if (/(ia|modelo|cnn|clasificador|proyecto|clase|categoría)/.test(normalized)) {
    return 'Este proyecto usa una IA para clasificar residuos en dos grandes categorías: reciclable y no reciclable, con ejemplos como botellas, latas, cajas, utensilios y empaques especiales.'
  }

  return 'Solo puedo ayudar con preguntas relacionadas con reciclaje, residuos y las clases que se manejan en este proyecto. Si quieres, pregúntame por botellas, latas, tetrapack, utensilios o styrofoam.'
}

export default function Chat() {
  const nextMessageId = useRef(2)
  const [messages, setMessages] = useState([
    {
      id: 1,
      role: 'assistant',
      content: 'Hola. Puedo ayudarte con dudas sobre reciclaje y cómo se clasifican los residuos en este proyecto.',
    },
  ])
  const [input, setInput] = useState('')
  const [isSending, setIsSending] = useState(false)

  function createMessage(role, content) {
    const message = {
      id: nextMessageId.current,
      role,
      content,
    }

    nextMessageId.current += 1
    return message
  }

  async function submitMessage(messageText = input) {
    const trimmed = messageText.trim()

    if (!trimmed || isSending) {
      return
    }

    const userMessage = createMessage('user', trimmed)

    setMessages((current) => [...current, userMessage])
    setInput('')
    setIsSending(true)

    try {
      const response = await askChatbot(trimmed)
      const assistantMessage = createMessage('assistant', response.reply || buildReply(trimmed))

      setMessages((current) => [...current, assistantMessage])
    } catch {
      const assistantMessage = createMessage('assistant', buildReply(trimmed))

      setMessages((current) => [...current, assistantMessage])
    } finally {
      setIsSending(false)
    }
  }

  return (
    <section className="panel chat-panel" aria-label="Asistente de reciclaje">
      <div className="panel__header">
        <div className="panel__icon" aria-hidden="true">
          <Bot size={28} />
        </div>
        <div>
          <h2>Asistente EcoChat</h2>
          <p>Responde solo sobre reciclaje y las clases del proyecto.</p>
        </div>
      </div>

      <div className="panel__body chat-panel__body">
        <div className="chat-messages" role="log" aria-live="polite">
          {messages.map((message) => (
            <div key={message.id} className={`message message--${message.role}`}>
              <div className="message__icon" aria-hidden="true">
                {message.role === 'assistant' ? <Leaf size={16} /> : <Bot size={16} />}
              </div>
              <p>{message.content}</p>
            </div>
          ))}
        </div>

        <div className="chat-quick-actions">
          {QUICK_PROMPTS.map((prompt) => (
            <button key={prompt} type="button" onClick={() => submitMessage(prompt)} disabled={isSending}>
              {prompt}
            </button>
          ))}
        </div>

        <form
          className="chat-input-row"
          onSubmit={(event) => {
            event.preventDefault()
            submitMessage()
          }}
        >
          <input
            type="text"
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="Escribe una pregunta sobre reciclaje..."
            aria-label="Pregunta para el asistente"
            disabled={isSending}
          />
          <button type="submit" aria-label="Enviar pregunta" disabled={isSending}>
            <SendHorizontal size={18} />
          </button>
        </form>

        <p className="chat-hint">Ejemplos: ¿Qué es reciclable?, ¿Qué pasa con un tetrapack? o ¿Cómo se clasifica un utensilio?</p>
      </div>
    </section>
  )
}
