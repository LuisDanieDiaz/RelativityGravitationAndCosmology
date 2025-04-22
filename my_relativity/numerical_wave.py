def wavelength_to_rgb(wavelength, normalize=True):
    '''
    # Tomado de ChatGPT y modificado 04/2025
    Convierte longitud de onda (380-780 nm) a valores RGB (0-255)
    '''
    wavelength = max(380, min(780, wavelength))

    gamma = 0.8

    # Fuera del espectro visible
    if wavelength < 380 or wavelength > 780:
        return (0, 0, 0)

    # Ajuste según la longitud de onda
    if wavelength < 440:
        r = -(wavelength - 440) / (440 - 380)
        g = 0.0
        b = 1.0
    elif wavelength < 490:
        r = 0.0
        g = (wavelength - 440) / (490 - 440)
        b = 1.0
    elif wavelength < 510:
        r = 0.0
        g = 1.0
        b = -(wavelength - 510) / (510 - 490)
    elif wavelength < 580:
        r = (wavelength - 510) / (580 - 510)
        g = 1.0
        b = 0.0
    elif wavelength < 645:
        r = 1.0
        g = -(wavelength - 645) / (645 - 580)
        b = 0.0
    else:
        r = 1.0
        g = 0.0
        b = 0.0

    # Reducción de intensidad en los extremos del espectro visible
    if wavelength < 420:
        factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
    elif wavelength > 700:
        factor = 0.3 + 0.7 * (780 - wavelength) / (780 - 700)
    else:
        factor = 1.0

    if not normalize:
        r = int(255 * (r * factor)**gamma)
        g = int(255 * (g * factor)**gamma)
        b = int(255 * (b * factor)**gamma)
    
    else:
        r = (r * factor)**gamma
        g = (g * factor)**gamma
        b = (b * factor)**gamma

    return (r, g, b)