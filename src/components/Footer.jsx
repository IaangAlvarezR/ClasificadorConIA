import { Leaf, Recycle } from 'lucide-react'

export default function Footer() {
  return (
    <footer className="site-footer">
      <div className="site-footer__inner">
        <div className="footer-item">
          <Leaf size={42} aria-hidden="true" />
          <div>
            <strong>Proyecto Final - Redes Neuronales Convolucionales</strong>
            <span>Diplomado en Inteligencia Artificial</span>
          </div>
        </div>

        <div className="footer-item footer-item--right">
          <Recycle size={36} aria-hidden="true" />
          <div>
            <strong>Reciclar es transformar</strong>
            <span>Pequeñas acciones, grandes cambios</span>
          </div>
        </div>
      </div>
    </footer>
  )
}
