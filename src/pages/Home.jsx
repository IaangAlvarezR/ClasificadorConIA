import { useEffect, useState } from 'react'
import { ShieldCheck } from 'lucide-react'
import Footer from '../components/Footer.jsx'
import Header from '../components/Header.jsx'
import ResultPanel from '../components/ResultPanel.jsx'
import UploadPanel from '../components/UploadPanel.jsx'
import { classifyImage } from '../services/api.js'

export default function Home() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl('')
      setResult(null)
      setError('')
      return undefined
    }

    const objectUrl = URL.createObjectURL(selectedFile)
    setPreviewUrl(objectUrl)

    return () => URL.revokeObjectURL(objectUrl)
  }, [selectedFile])

  async function handleFileSelect(file) {
    setSelectedFile(file)
    setResult(null)
    setError('')
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
            onFileSelect={handleFileSelect}
          />
          <ResultPanel previewUrl={previewUrl} result={result} isLoading={isLoading} error={error} />
        </section>
      </main>

      <Footer />
    </div>
  )
}
