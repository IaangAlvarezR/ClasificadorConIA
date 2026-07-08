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
    let message

    try {
      const error = await response.json()
      message = `${error.detail ?? 'El backend rechazó la imagen'} (${API_URL})`
    } catch {
      message = `${response.statusText || 'El backend respondió con error'} (${API_URL})`
    }

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
      body: JSON.stringify({ question }),
    })
  } catch {
    throw new Error(`No se pudo conectar con el backend: ${API_URL}`)
  }

  if (!response.ok) {
    let message

    try {
      const error = await response.json()
      message = `${error.detail ?? 'El backend rechazó la pregunta'} (${API_URL})`
    } catch {
      message = `${response.statusText || 'El backend respondió con error'} (${API_URL})`
    }

    throw new Error(message)
  }

  return response.json()
}

export { API_URL }
