const API_URL = (import.meta.env.VITE_API_URL ?? 'http://localhost:8000').replace(/\/$/, '')

function getPublicErrorMessage(fallback) {
  return async (response) => {
    try {
      const error = await response.json()
      return error.detail ?? fallback
    } catch {
      return fallback
    }
  }
}

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
    throw new Error('No se pudo conectar con el servicio de analisis. Intenta de nuevo mas tarde.')
  }

  if (!response.ok) {
    const message = await getPublicErrorMessage('No se pudo analizar la imagen seleccionada.')(response)
    throw new Error(message)
  }

  return response.json()
}

export async function askChatbot(question) {
  let response

  try {
    response = await fetch(`${API_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ question: `${question}\n\nResponde breve y resumido, maximo 3 frases.` }),
    })
  } catch {
    throw new Error('No se pudo conectar con el asistente. Intenta de nuevo mas tarde.')
  }

  if (!response.ok) {
    const message = await getPublicErrorMessage('No se pudo enviar la pregunta al asistente.')(response)
    throw new Error(message)
  }

  return response.json()
}

export { API_URL }
