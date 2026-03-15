import gettext
import os
from typing import Callable

def setup_i18n(lang: str = 'en') -> Callable[[str], str]:
    """
    Configura el sistema de traducción y devuelve la función de traducción _.
    
    Args:
        lang: Código del idioma (ej. 'es', 'en').
        
    Returns:
        Función gettext instalada para traducir strings.
    """
    locale_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'locales')
    
    # Busca el catálogo 'messages' en la carpeta locales
    translation = gettext.translation(
        'messages', 
        localedir=locale_dir, 
        languages=[lang], 
        fallback=True
    )
    return translation.gettext

