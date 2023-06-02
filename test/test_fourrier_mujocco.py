import numpy as np

def signal(parameters):
    """
    Crée une fonction qui représente un signal en sommant plusieurs composantes sinusoïdales.
    
    Paramètres:
        parameters: liste de tuples, où chaque tuple contient trois valeurs - amplitude (A), fréquence (F) et phase (P).
        
    Retourne:
        Une fonction qui représente le signal généré.
    """
    def generated_signal(t):
        signal = 0  # Créer un signal initialisé à zéro

        for A, F, P in parameters:
            component = A * np.sin(2*np.pi*F*t + P)  # Générer chaque composante sinusoïdale
            signal += component  # Ajouter la composante au signal total

        return signal

    return generated_signal

# random parameters
size_parameters = 10
# list of tuples (A, F, P)
parameters = [(np.random.rand(), np.random.rand(), np.random.rand()) for _ in range(size_parameters)]
print(parameters)