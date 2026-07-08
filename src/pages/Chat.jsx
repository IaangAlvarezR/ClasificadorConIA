import { useRef, useState } from 'react'
import { Bot, Leaf, SendHorizontal } from 'lucide-react'
import { askChatbot } from '../services/api.js'

const QUICK_PROMPTS = [
  'Que materiales son reciclables?',
  'Que hago con una pila?',
  'Como separo residuos en casa?',
]

const LOCAL_REPLIES = [
  {
    patterns: ['hola', 'ayuda', 'que puedes hacer', 'como funciona'],
    reply:
      'Puedo ayudarte con dudas sobre reciclaje, separacion de residuos, limpieza de envases y las categorias del proyecto: reciclable y no reciclable.',
  },
  {
    patterns: ['limpiar', 'lavar', 'sucio', 'grasoso', 'comida', 'contaminado'],
    reply:
      'Si un envase tiene restos de comida, aceite o liquidos, conviene vaciarlo y enjuagarlo. Un residuo limpio tiene mas probabilidades de ser reciclable.',
  },
  {
    patterns: ['papel', 'carton', 'periodico', 'cuaderno', 'caja de carton'],
    reply:
      'Papel y carton suelen ser reciclables si estan secos y limpios. Si estan mojados, con grasa o con restos de comida, pueden dejar de aceptarse.',
  },
  {
    patterns: ['vidrio', 'frasco', 'botella de vidrio'],
    reply:
      'El vidrio limpio, como botellas y frascos, suele ser reciclable. Evita mezclarlo con ceramica, espejos o focos, porque normalmente requieren manejo especial.',
  },
  {
    patterns: ['metal', 'aluminio', 'lata', 'conserva'],
    reply:
      'Las latas de aluminio o metal limpias suelen ser reciclables. Lo mejor es vaciarlas, enjuagarlas y separarlas de residuos organicos.',
  },
  {
    patterns: ['plastico', 'pet', 'bolsa', 'botella plastica', 'envase plastico'],
    reply:
      'Muchos plasticos, como botellas PET limpias, pueden reciclarse. Bolsas, envolturas flexibles y plasticos sucios dependen mucho del centro de acopio local.',
  },
  {
    patterns: ['tetrapack', 'tetrabrik', 'jugo', 'leche', 'caja'],
    reply:
      'Los envases tipo tetrapack pueden reciclarse en lugares que aceptan materiales multicapa. Vacialos, enjuagalos y aplastalos antes de separarlos.',
  },
  {
    patterns: ['organico', 'comida', 'fruta', 'verdura', 'cascaras', 'jardin'],
    reply:
      'Los residuos organicos, como restos de fruta, verdura y poda, no van con reciclables secos. Pueden aprovecharse en composta si estan separados.',
  },
  {
    patterns: ['pila', 'bateria', 'electronico', 'celular', 'cargador', 'cable'],
    reply:
      'Pilas, baterias y electronicos no deben tirarse con basura comun. Llevan metales y componentes que requieren puntos de recoleccion especializados.',
  },
  {
    patterns: ['aceite', 'cocina', 'fritura'],
    reply:
      'El aceite usado no debe ir al drenaje. Guardalo frio en una botella cerrada y llevalo a un punto de recoleccion si existe en tu localidad.',
  },
  {
    patterns: ['medicina', 'medicamento', 'pastilla', 'jarabe'],
    reply:
      'Los medicamentos caducos no deben mezclarse con reciclables ni tirarse al drenaje. Lo ideal es llevarlos a un contenedor o farmacia con programa de acopio.',
  },
  {
    patterns: ['ropa', 'textil', 'zapato', 'tela'],
    reply:
      'La ropa y textiles no suelen clasificarse como reciclables comunes. Si estan en buen estado, donarlos o reutilizarlos suele ser mejor opcion.',
  },
  {
    patterns: ['unicel', 'styrofoam', 'espuma', 'poliestireno'],
    reply:
      'El unicel o espuma de poliestireno suele ser dificil de reciclar y muchas redes no lo aceptan, especialmente si tiene restos de comida.',
  },
  {
    patterns: ['contenedor', 'color', 'verde', 'amarillo', 'azul', 'separar'],
    reply:
      'Los colores de contenedores cambian por ciudad, pero la regla practica es separar reciclables limpios y secos, organicos, residuos sanitarios y residuos especiales.',
  },
  {
    patterns: ['sanitario', 'papel higienico', 'panal', 'toalla femenina', 'cubrebocas'],
    reply:
      'Los residuos sanitarios no se consideran reciclables. Deben ir cerrados en una bolsa y separados de materiales limpios como papel, carton, vidrio o metal.',
  },
  {
    patterns: ['foco', 'lampara', 'bombilla', 'fluorescente'],
    reply:
      'Focos, lamparas y fluorescentes requieren manejo especial. No los mezcles con vidrio comun porque pueden contener componentes peligrosos.',
  },
  {
    patterns: ['ia', 'modelo', 'cnn', 'clasificador', 'proyecto', 'clase', 'categoria'],
    reply:
      'El proyecto usa un modelo de IA para distinguir residuos reciclables y no reciclables. Si la imagen no parece un residuo, el sistema puede rechazarla por baja confianza.',
  },
]

function buildReply(question) {
  const normalized = question
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^\w\s]/g, '')
    .toLowerCase()

  const match = LOCAL_REPLIES.find(({ patterns }) => patterns.some((pattern) => normalized.includes(pattern)))
  if (match) {
    return match.reply
  }

  return 'Puedo orientarte sobre residuos reciclables, no reciclables y manejo responsable. Dame el material o ejemplo concreto, como botella, lata, carton, pila, aceite, ropa o tetrapack.'
}

export default function Chat() {
  const nextMessageId = useRef(2)
  const [messages, setMessages] = useState([
    {
      id: 1,
      role: 'assistant',
      content: 'Hola. Puedo ayudarte con dudas sobre reciclaje y como se clasifican los residuos en este proyecto.',
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
          {isSending ? (
            <div className="message message--assistant message--thinking">
              <div className="message__icon" aria-hidden="true">
                <Leaf size={16} />
              </div>
              <p>
                EcoChat esta pensando
                <span className="typing-dots" aria-hidden="true">
                  <span />
                  <span />
                  <span />
                </span>
              </p>
            </div>
          ) : null}
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

        <p className="chat-hint">Ejemplos: que es reciclable, que hago con una pila o como separo residuos en casa.</p>
      </div>
    </section>
  )
}
