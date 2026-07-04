import { Image, ShieldCheck, UploadCloud } from 'lucide-react'

export default function UploadPanel({ previewUrl, selectedFile, isLoading, onFileSelect }) {
  function handleChange(event) {
    const [file] = event.target.files
    if (file) {
      onFileSelect(file)
    }
  }

  return (
    <section className="panel">
      <div className="panel__header">
        <div className="panel__icon" aria-hidden="true">
          <UploadCloud size={28} />
        </div>
        <div>
          <h2>1. Subir imagen</h2>
          <p>Selecciona o arrastra una imagen del residuo</p>
        </div>
      </div>

      <div className="panel__body">
        <label className="drop-zone">
          <input
            type="file"
            accept="image/png,image/jpeg,image/jpg"
            disabled={isLoading}
            onChange={handleChange}
          />
          <UploadCloud size={72} aria-hidden="true" />
          <strong>Arrastra tu imagen aquí</strong>
          <span>o haz clic para seleccionar</span>
          <small>
            <Image size={16} aria-hidden="true" />
            Formatos permitidos: JPG, PNG, JPEG
          </small>
        </label>

        <div className="preview-block">
          <h3>Vista previa</h3>
          <div className={previewUrl ? 'preview preview--image' : 'preview'}>
            {previewUrl ? (
              <>
                <img src={previewUrl} alt="Vista previa del residuo seleccionado" />
                <span>{selectedFile?.name}</span>
              </>
            ) : (
              <>
                <Image size={28} aria-hidden="true" />
                <span>Aún no has seleccionado una imagen</span>
              </>
            )}
          </div>
        </div>

        <label className={isLoading ? 'primary-action primary-action--disabled' : 'primary-action'}>
          <Image size={22} aria-hidden="true" />
          {isLoading ? 'Analizando imagen...' : 'Seleccionar imagen'}
          <input
            type="file"
            accept="image/png,image/jpeg,image/jpg"
            disabled={isLoading}
            onChange={handleChange}
          />
        </label>

        <p className="security-note">
          <ShieldCheck size={18} aria-hidden="true" />
          Tu imagen se procesa de forma segura y no se almacena.
        </p>
      </div>
    </section>
  )
}
