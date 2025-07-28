async function enviarMensaje() {
    const mensaje = document.getElementById("mensaje").value;

    const emocionRes = await fetch('/api/emocion', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mensaje })
    });
    const emocionData = await emocionRes.json();

    const chatbotRes = await fetch('/api/chatbot', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mensaje, emocion: emocionData.emocion })
    });
    const chatbotData = await chatbotRes.json();

    const traduccionRes = await fetch('/api/traducir', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ texto: chatbotData.respuesta, idioma: 'es' })
    });
    const traduccionData = await traduccionRes.json();

    document.getElementById("respuesta").innerText = traduccionData.traduccion;
}
