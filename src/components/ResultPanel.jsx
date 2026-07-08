import {
  AlertCircle,
  Brain,
  CheckCircle2,
  Leaf,
  Search,
  ShieldCheck,
  Sparkles,
  Target,
  Zap,
} from 'lucide-react'
import Loader from './Loader.jsx'

const labelNames = {
  recyclable: 'Reciclable',
  non_recyclable: 'No reciclable',
  bottle_recyclable: 'Botella: Reciclable',
  can_recyclable: 'Lata: Reciclable',
  juice_box_recyclable: 'Caja de jugo: Reciclable',
  milk_carton_recyclable: 'Carton de leche: Reciclable',
  styrofoam_non_recyclable: 'Unicel: No reciclable',
  utensil_non_recyclable: 'Utensilio: No reciclable',
  bottle: 'Botella',
  can: 'Lata',
  juice_box: 'Caja de jugo',
  milk_carton: 'Carton de leche',
  styrofoam: 'Unicel',
  utensil: 'Utensilio',
  cardboard: 'Carton',
  glass: 'Vidrio',
  metal: 'Metal',
  paper: 'Papel',
  plastic: 'Plastico',
  trash: 'Basura general',
  organic: 'Organico',
  battery: 'Pila o bateria',
  electronic: 'Electronico',
  textile: 'Textil',
  carton: 'Carton',
  vidrio: 'Vidrio',
  papel: 'Papel',
  plastico: 'Plastico',
  basura: 'Basura general',
  organico: 'Organico',
}

const binaryLabels = new Set(['recyclable', 'non_recyclable'])

function getResultTitle(result) {
  if (!result) {
    return ''
  }

  if (result.material && result.recyclability_label) {
    return `${result.material}: ${result.recyclability_label}`
  }

  if (result.material) {
    return result.material
  }

  return labelNames[result.label] ?? result.label
}

function getResultType(result) {
  if (!result) {
    return ''
  }

  return result.material ? 'Tipo de basura' : binaryLabels.has(result.label) ? 'Clasificacion' : 'Tipo de basura'
}

export default function ResultPanel({ previewUrl, result, isLoading, error }) {
  const label = getResultTitle(result)
  const resultType = getResultType(result)

  return (
    <section className="panel">
      <div className="panel__header">
        <div className="panel__icon" aria-hidden="true">
          <Brain size={28} />
        </div>
        <div>
          <h2>2. Resultado de la IA</h2>
          <p>El resultado aparecerá aquí</p>
        </div>
      </div>

      <div className="panel__body result-body">
        {isLoading ? (
          <Loader />
        ) : error ? (
          <div className="error-state">
            <AlertCircle size={88} aria-hidden="true" />
            <h3>No se pudo analizar</h3>
            <p>{error}</p>
            <small>Prueba con una foto clara de un residuo sobre un fondo sencillo.</small>
          </div>
        ) : result ? (
          <div className="result-state">
            <CheckCircle2 size={96} aria-hidden="true" />
            <span className="result-kind">{resultType}</span>
            <h3>{label}</h3>
            <p>Confianza del modelo: {result.confidence}%</p>
            {result.probabilities ? (
              <div className="probabilities">
                {Object.entries(result.probabilities).map(([name, value]) => (
                  <div className="probability" key={name}>
                    <span>{labelNames[name] ?? name}</span>
                    <strong>{value}%</strong>
                  </div>
                ))}
              </div>
            ) : null}
          </div>
        ) : (
          <div className="empty-state">
            <div className="empty-state__art" aria-hidden="true">
              <Sparkles className="spark spark--one" size={14} />
              <Sparkles className="spark spark--two" size={14} />
              <Sparkles className="spark spark--three" size={12} />
              <Search size={112} />
              <Leaf size={48} />
            </div>
            <h3>{previewUrl ? 'Imagen lista para analizar' : 'Esperando imagen'}</h3>
            <p>
              {previewUrl
                ? 'La clasificación se mostrará aquí después de consultar el modelo.'
                : 'Sube una imagen de un residuo para que nuestro modelo de IA pueda analizarlo.'}
            </p>
          </div>
        )}

        <div className="trust-row">
          <div className="trust-item">
            <Target size={38} aria-hidden="true" />
            <div>
              <strong>Precisión alta</strong>
              <span>Modelo entrenado con imágenes reales</span>
            </div>
          </div>
          <div className="trust-item trust-item--gold">
            <Zap size={38} aria-hidden="true" />
            <div>
              <strong>Análisis rápido</strong>
              <span>Resultados en segundos</span>
            </div>
          </div>
          <div className="trust-item">
            <ShieldCheck size={38} aria-hidden="true" />
            <div>
              <strong>100% Seguro</strong>
              <span>Tus datos están protegidos</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
