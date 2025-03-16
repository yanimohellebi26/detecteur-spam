from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from detection import SpamDetector
import logging
import signal
import asyncio

TOKEN = '8045145813:AAG15jASsrSSYR6Y8RNSPM9_6D55x8gvApc'
detector = SpamDetector()

# Configuration du logging avancé
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def start(update: Update, context: CallbackContext):
    """Gestion personnalisée de la commande /start"""
    try:
        user = update.effective_user
        logger.info(f"Nouvel utilisateur: {user.id} - {user.full_name}")
        await update.message.reply_html(
            f"🛡️ Bonjour {user.mention_html()}!\n"
            "Je suis votre assistant de détection de spam.\n\n"
            "Envoyez-moi n'importe quel message pour analyser sa légitimité.\n"
            "Exemple : \"Gagnez 1 million € maintenant!\""
        )
    except Exception as e:
        logger.error(f"Erreur dans /start: {str(e)}", exc_info=True)

async def analyser_message(update: Update, context: CallbackContext):
    """Analyse approfondie des messages avec seuil dynamique"""
    try:
        texte = update.message.text
        logger.info(f"Message reçu de {update.effective_user.id}: {texte}")
        
        # Analyse avec seuil adaptatif
        prediction = detector.predict(texte, confidence_threshold=0.6)
        logger.debug(f"Détails prédiction: {prediction}")
        
        # Réponse contextuelle
        if prediction.get('error'):
            raise Exception(prediction['error'])
            
        if prediction['is_spam']:
            response = (
                f"🚨 <b>ALERTE SPAM DÉTECTÉE!</b>\n\n"
                f"Confiance: {prediction['confidence']:.0%}\n"
                f"Type: {prediction['label'].upper()}\n\n"
                "<i>Conseil: Ne cliquez sur aucun lien et ne répondez pas</i>"
            )
        else:
            response = (
                f"✅ <b>MESSAGE SÛR</b>\n\n"
                f"Confiance: {prediction['confidence']:.0%}\n"
                f"Type: {prediction['label'].capitalize()}"
            )
            
        await update.message.reply_html(response, disable_web_page_preview=True)
        
    except Exception as e:
        logger.error(f"Erreur d'analyse: {str(e)}", exc_info=True)
        await update.message.reply_text(
            "⚠️ Une erreur technique est survenue. Veuillez réessayer plus tard.",
            parse_mode='HTML'
        )

def gestion_signaux():
    """Gestion professionnelle des signaux système"""
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: loop.create_task(arret_programme()))

async def arret_programme():
    """Arrêt propre de l'application"""
    logger.info("🚦 Arrêt en cours...")
    await application.stop()
    logger.info("✅ Application arrêtée proprement")
    exit(0)

if __name__ == '__main__':
    # Initialisation de l'application
    application = Application.builder().token(TOKEN).build()
    
    # Enregistrement des handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, analyser_message))
    
    # Gestion des signaux
    gestion_signaux()
    
    # Démarrage
    logger.info("🤖 Démarrage du bot...")
    application.run_polling(
        poll_interval=3.0,
        timeout=15,
        drop_pending_updates=True,
        allowed_updates=Update.ALL_TYPES
    )