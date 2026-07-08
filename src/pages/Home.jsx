import { useEffect, useState } from 'react'
import { ShieldCheck } from 'lucide-react'
import Footer from '../components/Footer.jsx'
import Header from '../components/Header.jsx'
import ResultPanel from '../components/ResultPanel.jsx'
import UploadPanel from '../components/UploadPanel.jsx'
import Chat from './Chat.jsx'
import { classifyImage } from '../services/api.js'

const MAX_UPLOAD_SIZE = 5 * 1024 * 1024
const ALLOWED_IMAGE_TYPES = new Set(['image/jpeg', 'image/jpg', 'image/png', 'image/webp'])
const ALLOWED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp']

function validateImageFile(file) {
  if (!file) {
    return 'No se selecciono ninguna imagen. Elige un archivo JPG, PNG o WEBP para continuar.'
  }

  if (file.size === 0) {
    return 'La imagen esta vacia. Selecciona un archivo con contenido para poder analizarlo.'
  }

  if (file.size > MAX_UPLOAD_SIZE) {
    return 'La imagen supera el limite de 5 MB. Usa una imagen mas ligera e intenta de nuevo.'
  }

  const fileName = file.name.toLowerCase()
  const hasAllowedExtension = ALLOWED_IMAGE_EXTENSIONS.some((extension) => fileName.endsWith(extension))

  if (!ALLOWED_IMAGE_TYPES.has(file.type) && !hasAllowedExtension) {
    return 'Formato no valido. Solo se aceptan imagenes JPG, JPEG, PNG o WEBP.'
  }

  return ''
}

export default function Home() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    if (previewUrl) {
      return () => URL.revokeObjectURL(previewUrl)
    }

    return undefined
  }, [previewUrl])

  async function handleFileSelect(file) {
    setResult(null)
    setError('')

    const validationError = validateImageFile(file)
    if (validationError) {
      setSelectedFile(null)
      setPreviewUrl('')
      setIsLoading(false)
      setError(validationError)
      return
    }

    let objectUrl

    try {
      objectUrl = URL.createObjectURL(file)
    } catch {
      setSelectedFile(null)
      setPreviewUrl('')
      setIsLoading(false)
      setError('No se pudo crear la vista previa de la imagen. Intenta con otro archivo.')
      return
    }

    setSelectedFile(file)
    setPreviewUrl(objectUrl)
    setIsLoading(true)

    try {
      const prediction = await classifyImage(file)
      setResult(prediction)
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="app-shell">
      <Header />

      <main className="page-content">
        <section className="intro">
          <div>
            <h2>
              Clasificador <span>Inteligente</span> de Residuos
            </h2>
            <p>Sube una imagen de un residuo como plástico, vidrio, lata, tetrapack o utensilio y la IA determinará si es reciclable o no reciclable.</p>
          </div>

          <aside className="mission-card">
            <ShieldCheck size={36} aria-hidden="true" />
            <p>
              Promovemos un futuro <strong>más limpio y sostenible</strong>
            </p>
          </aside>
        </section>

        <section className="workspace-grid">
          <UploadPanel
            previewUrl={previewUrl}
            selectedFile={selectedFile}
            isLoading={isLoading}
            error={error}
            onFileSelect={handleFileSelect}
          />
          <ResultPanel previewUrl={previewUrl} result={result} isLoading={isLoading} error={error} />
        </section>

        <section className="chat-section">
          <Chat />
        </section>
      </main>

      <Footer />
    </div>
  )
}
