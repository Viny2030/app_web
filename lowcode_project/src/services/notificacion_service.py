import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from pathlib import Path
import logging
from typing import Optional, List, Union
import os
from ..config import SMTP_SERVER, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, EMAIL_FROM

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServicioNotificaciones:
    """Servicio para enviar notificaciones por correo electrónico"""
    
    def __init__(self, servidor_smtp: str = None, puerto: int = None, 
                 usuario: str = None, contraseña: str = None, 
                 email_from: str = None):
        """
        Inicializa el servicio de notificaciones
        
        Args:
            servidor_smtp: Servidor SMTP
            puerto: Puerto del servidor SMTP
            usuario: Usuario para autenticación SMTP
            contraseña: Contraseña para autenticación SMTP
            email_from: Dirección de correo del remitente
        """
        self.servidor_smtp = servidor_smtp or SMTP_SERVER
        self.puerto = puerto or SMTP_PORT
        self.usuario = usuario or SMTP_USER
        self.contraseña = contraseña or SMTP_PASSWORD
        self.email_from = email_from or EMAIL_FROM
        
        if not all([self.servidor_smtp, self.usuario, self.contraseña, self.email_from]):
            logger.warning("Faltan credenciales de correo. Las notificaciones no se enviarán.")
    
    def enviar_correo(
        self,
        destinatario: Union[str, List[str]],
        asunto: str,
        cuerpo: str,
        es_html: bool = False,
        adjuntos: Optional[List[Union[str, Path]]] = None
    ) -> bool:
        """
        Envía un correo electrónico
        
        Args:
            destinatario: Dirección de correo o lista de direcciones
            asunto: Asunto del correo
            cuerpo: Cuerpo del mensaje
            es_html: Si el cuerpo está en formato HTML
            adjuntos: Lista de rutas a archivos para adjuntar
            
        Returns:
            bool: True si el correo se envió correctamente, False en caso contrario
        """
        if not all([self.servidor_smtp, self.usuario, self.contraseña, self.email_from]):
            logger.error("No se pueden enviar correos: faltan credenciales SMTP")
            return False
            
        if not destinatario:
            logger.error("No se especificó ningún destinatario")
            return False
            
        try:
            # Crear mensaje
            msg = MIMEMultipart()
            msg['From'] = self.email_from
            msg['To'] = ', '.join(destinatario) if isinstance(destinatario, list) else destinatario
            msg['Subject'] = asunto
            
            # Adjuntar cuerpo del mensaje
            msg.attach(MIMEText(cuerpo, 'html' if es_html else 'plain'))
            
            # Adjuntar archivos si los hay
            if adjuntos:
                for ruta_adjunto in adjuntos:
                    ruta_adjunto = Path(ruta_adjunto)
                    if ruta_adjunto.exists():
                        with open(ruta_adjunto, 'rb') as f:
                            adjunto = MIMEApplication(f.read(), _subtype=ruta_adjunto.suffix[1:])
                            adjunto.add_header(
                                'Content-Disposition',
                                'attachment',
                                filename=ruta_adjunto.name
                            )
                            msg.attach(adjunto)
                    else:
                        logger.warning(f"El archivo adjunto no existe: {ruta_adjunto}")
            
            # Enviar correo
            with smtplib.SMTP(self.servidor_smtp, self.puerto) as servidor:
                servidor.starttls()
                servidor.login(self.usuario, self.contraseña)
                servidor.send_message(msg)
                
            logger.info(f"Correo enviado exitosamente a {destinatario}")
            return True
            
        except Exception as e:
            logger.error(f"Error al enviar correo: {e}")
            return False
    
    def notificar_analisis_completado(
        self,
        destinatario: Union[str, List[str]],
        nombre_cliente: str,
        enlace_resultados: str,
        resumen_analisis: str = None
    ) -> bool:
        """
        Envía una notificación de que el análisis ha sido completado
        
        Args:
            destinatario: Correo electrónico del destinatario o lista de correos
            nombre_cliente: Nombre del cliente
            enlace_resultados: URL para acceder a los resultados
            resumen_analisis: Resumen del análisis realizado (opcional)
            
        Returns:
            bool: True si la notificación se envió correctamente
        """
        asunto = f"Análisis completado - {nombre_cliente}"
        
        # Crear cuerpo del mensaje en HTML
        cuerpo_html = f"""
        <html>
            <body>
                <h2>¡Hola {nombre_cliente}!</h2>
                <p>Tu análisis de datos ha sido completado exitosamente.</p>
                
                <p>Puedes acceder a tus resultados en el siguiente enlace:</p>
                <p><a href="{enlace_resultados}">{enlace_resultados}</a></p>
                
                {f'<h3>Resumen del análisis:</h3><p>{resumen_analisis}</p>' if resumen_analisis else ''}
                
                <p>Si tienes alguna pregunta o necesitas asistencia, no dudes en responder a este correo.</p>
                
                <p>Saludos,<br>El equipo de análisis de datos</p>
                
                <hr>
                <p style="color: #666; font-size: 12px;">
                    Este es un correo automático, por favor no lo responda directamente.
                </p>
            </body>
        </html>
        """
        
        return self.enviar_correo(
            destinatario=destinatario,
            asunto=asunto,
            cuerpo=cuerpo_html,
            es_html=True
        )
    
    def notificar_error(
        self,
        destinatario: Union[str, List[str]],
        nombre_cliente: str,
        mensaje_error: str,
        detalles_tecnicos: str = None
    ) -> bool:
        """
        Envía una notificación de error
        
        Args:
            destinatario: Correo electrónico del destinatario o lista de correos
            nombre_cliente: Nombre del cliente
            mensaje_error: Mensaje de error amigable para el usuario
            detalles_tecnicos: Detalles técnicos del error (opcional)
            
        Returns:
            bool: True si la notificación se envió correctamente
        """
        asunto = f"Error en el análisis - {nombre_cliente}"
        
        # Crear cuerpo del mensaje en HTML
        cuerpo_html = f"""
        <html>
            <body>
                <h2>¡Hola {nombre_cliente}!</h2>
                <p>Lamentamos informarte que ha ocurrido un error al procesar tu solicitud de análisis.</p>
                
                <div style="background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; padding: 10px; border-radius: 5px; margin: 10px 0;">
                    <p><strong>Error:</strong> {mensaje_error}</p>
                </div>
                
                {f'<div style="background-color: #e2e3e5; color: #383d41; border: 1px solid #d6d8db; padding: 10px; border-radius: 5px; margin: 10px 0;"><p><strong>Detalles técnicos:</strong><br>{detalles_tecnicos}</p></div>' if detalles_tecnicos else ''}
                
                <p>Nuestro equipo ha sido notificado y está trabajando para resolver el problema lo antes posible.</p>
                
                <p>Si el problema persiste, por favor contáctanos respondiendo a este correo.</p>
                
                <p>Lamentamos las molestias ocasionadas.</p>
                
                <p>Saludos,<br>El equipo de análisis de datos</p>
                
                <hr>
                <p style="color: #666; font-size: 12px;">
                    Este es un correo automático, por favor no lo responda directamente.
                </p>
            </body>
        </html>
        """
        
        return self.enviar_correo(
            destinatario=destinatario,
            asunto=asunto,
            cuerpo=cuerpo_html,
            es_html=True
        )
