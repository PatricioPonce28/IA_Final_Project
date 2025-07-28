async function enviarMensaje() {
  const mensaje = document.getElementById("mensaje").value.trim();
  const chatBox = document.getElementById("chat-box");
  const button = document.querySelector(".send-button");

  if (!mensaje) return;

  // Mostrar mensaje del usuario
  const userMsg = document.createElement("div");
  userMsg.className = "message user-message";
  userMsg.innerText = mensaje;
  chatBox.appendChild(userMsg);
  chatBox.scrollTop = chatBox.scrollHeight;

  document.getElementById("mensaje").value = "";
  button.disabled = true;

  try {
    const emocionRes = await fetch('/api/emocion', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mensaje })
    });

    if (!emocionRes.ok) throw new Error('Error al detectar emoción');
    const emocionData = await emocionRes.json();

    const botchatRes = await fetch('/api/botchat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        mensaje: mensaje,
        emocion: emocionData.emocion || 'neutral'
      })
    });

    if (!botchatRes.ok) {
      const errorData = await botchatRes.json();
      throw new Error(errorData.error || 'Error en el chatbot');
    }

    const botchatData = await botchatRes.json();

    // Mostrar respuesta del bot
    const botMsg = document.createElement("div");
    botMsg.className = "message bot-message";
    botMsg.innerText = botchatData.respuesta;
    chatBox.appendChild(botMsg);
    chatBox.scrollTop = chatBox.scrollHeight;

  } catch (error) {
    const errorMsg = document.createElement("div");
    errorMsg.className = "message bot-message";
    errorMsg.innerText = `Error: ${error.message}`;
    chatBox.appendChild(errorMsg);
    chatBox.scrollTop = chatBox.scrollHeight;
  } finally {
    button.disabled = false;
  }
}
