const API_URL = (import.meta.env.VITE_API_URL ?? 'http://localhost:8000').replace(/\/$/, '')

export async function classifyImage(file) {
  const formData = new FormData()
  formData.append('file', file)

  let response

  try {
    response = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      body: formData,
    })
  } catch {
    throw new Error(`No se pudo conectar con el backend: ${API_URL}`)
  }

  if (!response.ok) {
    let message = 'No se pudo clasificar la imagen'

    try {
      const error = await response.json()
      message = error.detail ?? message
    } catch {
      message = response.statusText || message
    }

    throw new Error(message)
  }

  return response.json()
}

export { API_URL }
