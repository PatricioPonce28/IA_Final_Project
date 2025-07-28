async function enviarMensaje() {
    const mensaje = document.getElementById("mensaje").value;
    const button = document.querySelector("button");
    
    try {
        button.disabled = true;
        document.getElementById("respuesta").innerText = "Pensando...";

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
        document.getElementById("respuesta").innerText = botchatData.respuesta;
        
    } catch (error) {
        console.error("Error completo:", error);
        document.getElementById("respuesta").innerText = `Error: ${error.message}`;
    } finally {
        button.disabled = false;
    }
}