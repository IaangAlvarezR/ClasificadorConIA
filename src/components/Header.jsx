import { GraduationCap, Leaf } from 'lucide-react'

export default function Header() {
  return (
    <header className="site-header">
      <div className="site-header__inner">
        <div className="brand">
          <div className="brand__mark" aria-hidden="true">
            <Leaf size={32} strokeWidth={2.2} />
          </div>
          <div>
            <h1>EcoClasifica IA</h1>
            <p>Clasificador Inteligente de Residuos</p>
          </div>
        </div>

        <div className="academic-badge">
          <GraduationCap size={28} aria-hidden="true" />
          <div>
            <strong>Proyecto Académico</strong>
            <span>Deep Learning · CNN</span>
          </div>
        </div>
      </div>
    </header>
  )
}
